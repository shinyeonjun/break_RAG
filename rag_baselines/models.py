"""Native Windows adapter for the pinned BERGEN RetroMAE/DeBERTa settings.

This implements the two retrieval paths, not BERGEN's full experiment runner.
"""

from pathlib import Path
import hashlib
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .data import file_sha256, read_json, read_jsonl, write_json
from .store import DocumentStore


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_files(directory):
    """Bind cached embeddings to actual weights and tokenizer, not just names."""
    names = ("config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.txt", "spm.model")
    return {name: file_sha256(directory / name) for name in names if (directory / name).exists()}


class Baselines:
    def __init__(self, root, device="auto", retrieve_k=None, evidence_k=None, raw_log=None):
        self.root = Path(root)
        self.raw_log = raw_log
        self.config = read_json(self.root / "config/baselines.json")
        for key, value in (("retrieve_k", retrieve_k), ("evidence_k", evidence_k)):
            if value is not None:
                self.config[key] = value
        if not 1 <= self.config["evidence_k"] <= self.config["retrieve_k"]:
            raise ValueError("Require 1 <= evidence_k <= retrieve_k.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        torch.manual_seed(self.config["seed"])
        # Avoid TF32 changing the specified dot product on different machines.
        torch.backends.cuda.matmul.allow_tf32 = False
        self.models = {}
        self.tokenizers = {}
        self.file_hashes = {}
        started = time.perf_counter()
        for role, loader in (("retriever", AutoModel), ("reranker", AutoModelForSequenceClassification)):
            directory = self.root / self.config[role]["local_dir"]
            self.file_hashes[role] = model_files(directory)
            self.tokenizers[role] = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
            self.models[role] = loader.from_pretrained(directory, dtype=self.dtype, local_files_only=True,
                                                       trust_remote_code=False).eval().to(self.device)
        if self.models["reranker"].config.num_labels != 1:
            raise ValueError("This baseline requires the pinned single-logit reranker.")
        synchronize(self.device)
        self.load_seconds = time.perf_counter() - started
        self.passages = []
        self.embeddings = None
        self.corpus_sha256 = None
        self.index_info = None
        self.retriever_signature = hashlib.sha256(json.dumps(
            {"config": self.config["retriever"], "files": self.file_hashes["retriever"], "dtype": str(self.dtype)},
            sort_keys=True).encode()).hexdigest()

    def _encode(self, texts, passage_ids=None):
        cfg = self.config["retriever"]
        inputs = self.tokenizers["retriever"](texts, padding="longest", truncation="longest_first",
                                               max_length=cfg["max_length"], return_tensors="pt")
        lengths = [len(ids) for ids in self.tokenizers["retriever"](texts, truncation=False)["input_ids"]]
        token_counts = {"nonpad": int(inputs["attention_mask"].sum()), "padded": int(inputs["input_ids"].numel()),
                        "original_nonpad": sum(lengths), "truncated_sequences": sum(length > cfg["max_length"] for length in lengths)}
        token_counts["dropped_tokens"] = sum(lengths) - token_counts["nonpad"]
        audit = getattr(self, "raw_log", None)
        if audit:
            call_id = audit.model_start("retriever", dict(inputs), {
                "texts": texts, "passage_ids": passage_ids, "tokenization": cfg,
                "model_files": self.file_hashes["retriever"], "tokens": token_counts})
            synchronize(self.device)
            compute_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.models["retriever"](**inputs.to(self.device)).last_hidden_state[:, 0]
        if audit:
            synchronize(self.device)
            compute_seconds = time.perf_counter() - compute_started
            audit.model_end("retriever", call_id, {"cls_embeddings": outputs},
                            compute_seconds=compute_seconds, tokens=token_counts)
        return outputs, token_counts

    def build_index(self, corpus_path, progress=print):
        """Offline document embedding/cache load; never included in online time."""
        started = time.perf_counter()
        corpus_path = Path(corpus_path)
        self.store = DocumentStore(corpus_path)
        self.corpus_sha256 = self.store.sha256
        self.passages = self.store.passages
        ids = [p["id"] for p in self.passages]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Corpus must have nonempty unique passage IDs.")
        signature = {"corpus_sha256": self.corpus_sha256, "retriever": self.config["retriever"],
                     "model_files": self.file_hashes["retriever"], "dtype": str(self.dtype),
                     "count": len(ids)}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
        cache = self.root / "data/indexes"
        cache.mkdir(parents=True, exist_ok=True)
        array_path, meta_path = cache / f"{key}.npy", cache / f"{key}.json"
        cache_hit = array_path.exists() and meta_path.exists()
        if cache_hit:
            metadata = read_json(meta_path)
            if metadata["signature"] != signature or metadata["embedding_sha256"] != file_sha256(array_path):
                raise ValueError("Embedding cache integrity check failed.")
            array = np.load(array_path, allow_pickle=False)
        else:
            arrays = []
            token_counts = {"nonpad": 0, "padded": 0, "original_nonpad": 0, "truncated_sequences": 0, "dropped_tokens": 0}
            batch = self.config["retriever"]["batch_size"]
            for offset in range(0, len(self.passages), batch):
                passages = self.passages[offset:offset + batch]
                embeddings, counts = self._encode([p["content"] for p in passages], [p["id"] for p in passages])
                arrays.append(embeddings.cpu().numpy())
                for name in counts:
                    token_counts[name] += counts[name]
                if progress and (offset // batch % 100 == 0 or offset + batch >= len(self.passages)):
                    progress(f"Index: {min(offset + batch, len(self.passages)):,}/{len(self.passages):,}")
            array = np.concatenate(arrays)
            temporary = array_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(array_path)
            metadata = {"signature": signature, "embedding_sha256": file_sha256(array_path),
                        "document_tokens": token_counts, "build_seconds": time.perf_counter() - started}
            write_json(meta_path, metadata)
        if array.shape != (len(ids), self.models["retriever"].config.hidden_size) or not np.isfinite(array).all():
            raise ValueError("Invalid embedding array shape/values.")
        self.embeddings = torch.from_numpy(array).to(self.device)
        self.id_to_index = self.store.indices
        synchronize(self.device)
        self.index_info = {**metadata, "cache_hit": cache_hit, "prepare_seconds": time.perf_counter() - started}
        if self.raw_log:
            self.raw_log.event("index", "prepared", {**self.index_info, "passage_ids_in_embedding_order": ids})
            self.raw_log.snapshot(array_path)
            self.raw_log.snapshot(meta_path)
        return self.index_info

    def _rerank(self, question, candidate_indices):
        cfg = self.config["reranker"]
        scores = []
        token_counts = {"nonpad": 0, "padded": 0, "original_nonpad": 0, "truncated_sequences": 0, "dropped_tokens": 0}
        for offset in range(0, len(candidate_indices), cfg["batch_size"]):
            indices = candidate_indices[offset:offset + cfg["batch_size"]]
            inputs = self.tokenizers["reranker"]([question] * len(indices),
                                                 [self.passages[i]["content"] for i in indices],
                                                 padding=cfg["padding"], truncation=cfg["truncation"],
                                                 max_length=cfg["max_length"], return_tensors="pt")
            token_counts["nonpad"] += int(inputs["attention_mask"].sum())
            token_counts["padded"] += int(inputs["input_ids"].numel())
            full = self.tokenizers["reranker"]([question] * len(indices),
                                               [self.passages[i]["content"] for i in indices], truncation=False)
            lengths = [len(ids) for ids in full["input_ids"]]
            token_counts["original_nonpad"] += sum(lengths)
            token_counts["truncated_sequences"] += sum(length > cfg["max_length"] for length in lengths)
            audit = getattr(self, "raw_log", None)
            if audit:
                call_id = audit.model_start("reranker", dict(inputs), {
                    "question": question, "passages": [self.passages[i] for i in indices],
                    "tokenization": cfg, "model_files": self.file_hashes["reranker"],
                    "original_lengths": lengths, "nonpad_lengths": inputs["attention_mask"].sum(1).tolist()})
                synchronize(self.device)
                compute_started = time.perf_counter()
            with torch.inference_mode():
                logits = self.models["reranker"](**inputs.to(self.device)).logits[:, 0]
            if audit:
                synchronize(self.device)
                compute_seconds = time.perf_counter() - compute_started
                audit.model_end("reranker", call_id, {"logits": logits}, compute_seconds=compute_seconds)
            scores.extend(logits.float().cpu().tolist())
        token_counts["dropped_tokens"] = token_counts["original_nonpad"] - token_counts["nonpad"]
        return scores, token_counts

    def retrieve(self, question, method):
        """A/B independently encode and search. Takes only natural-language input."""
        if method not in ("A", "B"):
            raise ValueError("method must be A or B.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text.")
        if self.embeddings is None:
            raise RuntimeError("Run build_index before online retrieval.")
        synchronize(self.device)
        started = time.perf_counter()
        query, query_tokens = self._encode([question])
        synchronize(self.device)
        encoded_at = time.perf_counter()
        audit = getattr(self, "raw_log", None)
        if audit:
            call_id = audit.model_start("dense_search", {"query_embedding": query}, {
                "corpus_sha256": self.corpus_sha256, "retrieve_k": self.config["retrieve_k"],
                "score_order": "corpus passage order in the index preparation event", "similarity": "unnormalized dot product"})
            search_started = time.perf_counter()
        with torch.inference_mode():
            all_scores = torch.mm(query, self.embeddings.T)[0].float()
            # Stable corpus-order tie breaking for exact equal scores.
            indices = torch.argsort(all_scores, descending=True, stable=True)[:self.config["retrieve_k"]].cpu().tolist()
            scores = all_scores[indices].cpu().tolist()
        synchronize(self.device)
        if audit:
            compute_seconds = time.perf_counter() - search_started
            audit.model_end("dense_search", call_id, {"all_scores": all_scores, "top_indices": indices},
                            compute_seconds=compute_seconds)
        searched_at = time.perf_counter()
        rerank_scores, rerank_tokens = None, {"nonpad": 0, "padded": 0}
        if method == "B":
            rerank_scores, rerank_tokens = self._rerank(question, indices)
            order = sorted(range(len(indices)), key=lambda i: (-rerank_scores[i], i))
        else:
            order = list(range(len(indices)))
        synchronize(self.device)
        reranked_at = time.perf_counter()
        selected = order[:self.config["evidence_k"]]
        candidates = [{"id": self.passages[index]["id"], "dense_score": scores[i],
                       "rerank_score": rerank_scores[i] if rerank_scores is not None else None}
                      for i, index in enumerate(indices)]
        evidence = [{**self.passages[indices[i]], **candidates[i]} for i in selected]
        result = {"method": method, "question": question, "corpus_sha256": self.corpus_sha256,
                  "corpus_count": len(self.passages), "candidates": candidates, "evidence": evidence,
                  "execution_mode": "online_retrieval", "retrieve_k": self.config["retrieve_k"],
                  "evidence_k": self.config["evidence_k"], "retriever_signature": self.retriever_signature,
                  "seconds": {"query_encode": encoded_at - started, "vector_search": searched_at - encoded_at,
                              "rerank": reranked_at - searched_at if method == "B" else 0.0},
                  "model_tokens": {"retriever_query": query_tokens, "reranker_pairs": rerank_tokens},
                  "api_usage": None, "api_cost_usd": None}
        ended = time.perf_counter()
        result["seconds"]["assemble"] = ended - reranked_at
        result["seconds"]["retrieval_total"] = ended - started
        return result

    def select_cached(self, question, method, cached):
        """Quality evaluation only: share exact saved dense candidates for A/B."""
        if method not in ("A", "B"):
            raise ValueError("method must be A or B.")
        if cached["question"] != question or cached["corpus_sha256"] != self.corpus_sha256 or cached["retriever_signature"] != self.retriever_signature or cached["retrieve_k"] != self.config["retrieve_k"]:
            raise ValueError("Candidate cache does not match question/corpus/model/settings.")
        candidates = cached["candidates"]
        if len(candidates) != min(len(self.passages), self.config["retrieve_k"]) or len({c["id"] for c in candidates}) != len(candidates):
            raise ValueError("Invalid cached candidate count or duplicate IDs.")
        indices = [self.id_to_index[c["id"]] for c in candidates]
        scores, counts = None, {"nonpad": 0, "padded": 0}
        if method == "B":
            scores, counts = self._rerank(question, indices)
        order = sorted(range(len(indices)), key=lambda i: (-scores[i], i)) if scores is not None else list(range(len(indices)))
        candidate_rows = [{"id": c["id"], "dense_score": c["dense_score"], "rerank_score": scores[i] if scores is not None else None}
                          for i, c in enumerate(candidates)]
        evidence = [{**self.passages[indices[i]], **candidate_rows[i]} for i in order[:self.config["evidence_k"]]]
        return {"method": method, "question": question, "corpus_sha256": self.corpus_sha256,
                "corpus_count": len(self.passages), "candidates": candidate_rows, "evidence": evidence,
                "execution_mode": "cached_quality", "retrieve_k": self.config["retrieve_k"], "evidence_k": self.config["evidence_k"],
                "retriever_signature": self.retriever_signature,
                "seconds": {"retrieval_total": None},
                "model_tokens": {"retriever_query": {"nonpad": None, "padded": None}, "reranker_pairs": counts},
                "api_usage": None, "api_cost_usd": None}

    def truncation_audit(self, questions):
        """Verify actual query/doc lengths and question preservation in pair encoding."""
        query_texts = [row["question"] for row in questions]
        audit = {}
        for role in ("retriever", "reranker"):
            tokenizer, limit = self.tokenizers[role], self.config[role]["max_length"]
            lengths = [len(ids) for ids in tokenizer(query_texts, truncation=False)["input_ids"]]
            audit[role] = {"query_count": len(lengths), "query_max_tokens": max(lengths), "query_truncated_count": sum(length > limit for length in lengths)}
            doc_lengths = [len(ids) for ids in tokenizer([p["content"] for p in self.passages], truncation=False)["input_ids"]]
            audit[role].update({"doc_max_tokens": max(doc_lengths), "doc_over_limit_count": sum(length > limit for length in doc_lengths)})
        # only_second must keep the complete question side, including separators.
        tokenizer = self.tokenizers["reranker"]
        for question in query_texts:
            query_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
            if len(query_ids) + tokenizer.num_special_tokens_to_add(pair=True) >= self.config["reranker"]["max_length"]:
                raise ValueError("A question is too long for only_second truncation.")
            encoded = tokenizer(question, self.passages[0]["content"], truncation="only_second",
                                max_length=self.config["reranker"]["max_length"])
            question_side = [tid for tid, sequence in zip(encoded["input_ids"], encoded.sequence_ids()) if sequence == 0]
            if question_side != query_ids:
                raise ValueError("Reranker query was truncated or changed.")
        audit["reranker"]["question_side_verified"] = len(query_texts)
        synthetic_doc = "evidence " * 600
        dense_ids = self.tokenizers["retriever"](synthetic_doc, truncation="longest_first",
                                                 max_length=self.config["retriever"]["max_length"])["input_ids"]
        pair = tokenizer(query_texts[0], synthetic_doc, truncation="only_second",
                         max_length=self.config["reranker"]["max_length"])
        question_side = [tid for tid, sequence in zip(pair["input_ids"], pair.sequence_ids()) if sequence == 0]
        assert len(dense_ids) == self.config["retriever"]["max_length"]
        assert len(pair["input_ids"]) == self.config["reranker"]["max_length"]
        assert question_side == tokenizer(query_texts[0], add_special_tokens=False)["input_ids"]
        audit["synthetic_long_document"] = {"dense_tokens": len(dense_ids), "pair_tokens": len(pair["input_ids"]), "question_preserved": True}
        return audit

    def warmup(self, question):
        # A and B calls are genuinely executed; their results are not measured.
        for method in ("A", "B"):
            self.retrieve(question, method)
