"""Offline label-based metrics and runtime manifests for notebook experiments."""

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
import platform
import random
import sys
import time

import torch

from .data import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from .raw_logging import query_scope


def retrieval_metrics(result, label):
    def has_complete_span(ids):
        selected = set(ids)
        return any(set(group) <= selected for group in label["evidence_groups"])
    evidence_ids = [p["id"] for p in result["evidence"]]
    candidates = [p["id"] for p in result["candidates"]]
    groups = label["evidence_groups"]
    return {"complete_evidence_hit": int(has_complete_span(evidence_ids)),
            "complete_candidate_hit": int(has_complete_span(candidates)),
            "best_group_coverage": max(len(set(group) & set(evidence_ids)) / len(group) for group in groups)}


def runtime_manifest(root, engine):
    root = Path(root)
    if hasattr(engine, "baseline") and hasattr(engine, "decision"):
        manifest = runtime_manifest(root, engine.baseline)
        manifest["c"] = {"signature": engine.decision.signature, "config": engine.decision.config,
                         "preparation": engine.decision.preparation,
                         "deployment": "remote Jev API; uses the shared prepared decision tree"}
        manifest["methods"] = ["A", "B", "C"]
        if engine.local_decision is not None:
            manifest["d"] = {"signature": engine.local_decision.signature,
                              "config": engine.local_decision.config,
                              "runtime": engine.local_decision.client.runtime_info(),
                              "preparation_shared_with": "C"}
            manifest["dependencies"]["laya"] = engine.local_decision.client.runtime_info()["library_version"]
            manifest["methods"].append("D")
        manifest["measurement"] = (
            "warm local A/B and D models; C and D share the same prepared tree; fresh retrieval; "
            "randomized method order; C includes remote API/network, D includes local tokenization and inference"
        )
        return manifest
    files = sorted((root / "rag_baselines").glob("*.py"))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "dependencies": {name: version(name) for name in ("torch", "transformers", "numpy", "huggingface_hub", "nbformat", "nbclient", "openai", "tiktoken", "pyarrow", "httpx", "python-dotenv")},
        "device": str(engine.device), "dtype": str(engine.dtype),
        "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "gpu_bytes": torch.cuda.get_device_properties(engine.device).total_memory if engine.device.type == "cuda" else None,
        "cuda": torch.version.cuda, "config": engine.config,
        "code_sha256": {str(path.relative_to(root)): file_sha256(path) for path in files},
        "model_files": engine.file_hashes, "model_load_seconds": engine.load_seconds,
        "index": engine.index_info,
        "measurement": "warm models, one request at a time, CUDA synchronized, independently executed A/B, randomized method order",
    }


def measure_retrieval(engine, questions, labels, *, repeats=1, seed=20261003, methods=("A", "B"), progress=print, on_result=None, audit=None):
    """Called by ipynb. Records every independent question × method × repeat."""
    if repeats < 1 or not questions:
        raise ValueError("Need positive repeats and nonempty questions.")
    by_id = {row["id"]: row for row in labels}
    if any(row["id"] not in by_id for row in questions):
        raise ValueError("Offline labels are missing for selected questions.")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Methods must be nonempty and distinct.")
    rng = random.Random(seed)
    results = []
    for repeat in range(repeats):
        ordered_questions = list(questions)
        rng.shuffle(ordered_questions)
        for row in ordered_questions:
            ordered_methods = list(methods)
            rng.shuffle(ordered_methods)
            for method in ordered_methods:
                with query_scope(audit, "latency", row, method, repeat):
                    capture_started = audit.capture_seconds if audit else 0
                    # Labels are used only AFTER retrieval has finished.
                    result = engine.retrieve(row["question"], method)
                    if audit:
                        result.update({"raw_log": audit.reference(),
                                       "diagnostic_capture_seconds": audit.capture_seconds - capture_started})
                    result.update({"question_id": row["id"], "task": row.get("task", "nq"), "repeat": repeat})
                    if audit:
                        audit.result("retrieval", result)
                    if hasattr(engine, "store"):
                        engine.store.validate_evidence(result)
                    result["metrics"] = retrieval_metrics(result, by_id[row["id"]])
                    if audit:
                        audit.result("evaluation", result["metrics"], label=by_id[row["id"]])
                    results.append(result)
                    if on_result:
                        on_result(result)
            if progress:
                progress(f"Repeat {repeat + 1}/{repeats}: {row['id']} completed")
    return results


def result_table(results):
    import pandas as pd
    return pd.DataFrame([{
        "method": r["method"], "question_id": r["question_id"], "task": r.get("task", "nq"), "repeat": r["repeat"],
        "execution_mode": r["execution_mode"],
        **{f"{key}_ms": value * 1000 if value is not None else None for key, value in r["seconds"].items()},
        **r["metrics"],
        "retriever_input_tokens": r["model_tokens"].get("retriever_query", {}).get("nonpad"),
        "reranker_input_tokens": r["model_tokens"].get("reranker_pairs", {}).get("nonpad"),
        "jev_input_tokens": r["model_tokens"].get("jev", {}).get("nonpad"),
        "jev_output_tokens": r["model_tokens"].get("jev", {}).get("output"),
        "laya_input_tokens": r["model_tokens"].get("laya", {}).get("nonpad"),
        "laya_output_tokens": r["model_tokens"].get("laya", {}).get("output"),
        "laya_decision_calls": (r.get("local_usage") or {}).get("calls"),
        "laya_truncated_calls": (r.get("local_usage") or {}).get("truncated_calls"),
        "laya_state_tokens_dropped": (r.get("local_usage") or {}).get("state_tokens_dropped"),
        "laya_inference_seconds": (r.get("local_usage") or {}).get("inference_seconds"),
        "retrieval_api_calls": (r.get("api_usage") or {}).get("requests", 0),
        "evidence_count": len(r["evidence"]),
        "evidence_text_tokens": r.get("evidence_text_tokens"),
        "decision_visited_nodes": len(r.get("trace", {}).get("visited_nodes", [])) if r["method"] in ("C", "D") else None,
        "decision_scored_passages": len(r["candidates"]) if r["method"] in ("C", "D") else None,
        "decision_stop_reason": r.get("trace", {}).get("stop_reason"),
        "c_visited_nodes": len(r.get("trace", {}).get("visited_nodes", [])) if r["method"] == "C" else None,
        "c_scored_passages": len(r["candidates"]) if r["method"] == "C" else None,
        "c_stop_reason": r.get("trace", {}).get("stop_reason") if r["method"] == "C" else None,
        "api_cost_usd": r["api_cost_usd"],
    } for r in results])


def save_run(root, results, manifest, notebook_name):
    """Unique directories preserve previous measurements instead of overwriting."""
    from uuid import uuid4
    root = Path(root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = root / "results" / f"{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    notebook = root / "notebook" / notebook_name
    manifest["notebook_source_sha256"] = notebook_source_sha256(notebook)
    write_json(directory / "manifest.json", manifest)
    write_jsonl(directory / "retrieval.jsonl", results)
    result_table(results).to_csv(directory / "retrieval.csv", index=False)
    table = result_table(results)
    summary = table.groupby(["task", "method"]).agg(
        questions=("question_id", "nunique"), records=("question_id", "size"),
        complete_evidence_rate=("complete_evidence_hit", "mean"),
        complete_candidate_rate=("complete_candidate_hit", "mean"),
        retrieval_p50_ms=("retrieval_total_ms", "median"),
        retrieval_p95_ms=("retrieval_total_ms", lambda x: x.quantile(0.95)),
        retrieval_api_calls=("retrieval_api_calls", "sum"),
        api_cost_usd=("api_cost_usd", lambda x: x.sum(min_count=1)),
        evidence_count_mean=("evidence_count", "mean"))
    summary.to_csv(directory / "summary.csv")
    return directory


def notebook_source_sha256(path):
    """Outputs, execution counts and random cell IDs are not notebook source."""
    import hashlib
    import nbformat
    notebook = nbformat.read(path, as_version=4)
    source = [{"type": cell.cell_type, "source": cell.source} for cell in notebook.cells]
    return hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate_cached_quality(engine, questions, labels, *, audit=None):
    """Persist one dense candidate list per question; A/B share it exactly."""
    import hashlib
    signature = {"corpus": engine.corpus_sha256, "retriever": engine.retriever_signature,
                 "retrieve_k": engine.config["retrieve_k"]}
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    path = engine.root / "data/retrieval_cache" / f"{key}.jsonl"
    meta_path = path.with_suffix(".json")
    if path.exists():
        if not meta_path.exists():
            # A legacy cache is regenerated rather than trusted without a hash.
            cache = {}
        else:
            meta = read_json(meta_path)
            if meta["signature"] != signature or meta["sha256"] != file_sha256(path):
                raise ValueError("Candidate cache integrity check failed.")
            cache = {row["question_id"]: row for row in read_jsonl(path)}
    else:
        cache = {}
    by_id = {row["id"]: row for row in labels}
    results = []
    for question in questions:
        qid = question["id"]
        if qid not in cache:
            with query_scope(audit, "quality_candidates", question, "A"):
                raw = engine.retrieve(question["question"], "A")
                if audit:
                    audit.result("retrieval", raw)
                cache[qid] = {key: raw[key] for key in ("question", "corpus_sha256", "retriever_signature", "retrieve_k", "candidates")}
                cache[qid]["question_id"] = qid
                write_jsonl(path, cache.values())
                write_json(meta_path, {"signature": signature, "sha256": file_sha256(path)})
        for method in ("A", "B"):
            with query_scope(audit, "quality", question, method):
                if audit:
                    audit.event("candidates", "cache_reused", {"path": str(path), "signature": signature,
                                "used_row": cache[qid], "dense_search_executed_in_this_scope": False})
                result = engine.select_cached(question["question"], method, cache[qid])
                engine.store.validate_evidence(result)
                result.update({"question_id": qid, "task": question.get("task", "nq"), "repeat": 0,
                               "metrics": retrieval_metrics(result, by_id[qid])})
                if audit:
                    result["raw_log"] = audit.reference()
                    audit.result("retrieval", result, label=by_id[qid])
                results.append(result)
    return results


def evaluate_comparative_quality(engine, questions, labels, *, audit=None):
    """A/B share dense candidates; each decision arm freezes its own tree decisions."""
    by_id = {row["id"]: row for row in labels}
    if any(row["id"] not in by_id for row in questions):
        raise ValueError("Offline labels are missing for selected questions.")
    results = evaluate_cached_quality(engine.baseline, questions, labels, audit=audit)
    for method, decision in engine.decisions.items():
        for question in questions:
            with query_scope(audit, "quality", question, method):
                result = decision.retrieve_cached_quality(question["question"])
                result.update({"question_id": question["id"], "task": question.get("task", "nq"), "repeat": 0,
                               "metrics": retrieval_metrics(result, by_id[question["id"]])})
                if audit:
                    result["raw_log"] = audit.reference()
                    audit.event("candidates", "cache_status", {"cache_hit": result["decision_cache_hit"],
                                "actual_calls": len(result["trace"]["calls"]),
                                "historical_calls": len(result["trace"].get("original_calls", []))})
                    audit.result("retrieval", result, label=by_id[question["id"]])
                results.append(result)
    return results


def measure_system(engine, questions, labels, *, client, generation_config, max_calls, repeats=1,
                   methods=("A", "B"), seed=20261003, progress=print, on_result=None, audit=None):
    """Live retrieval → common API → offline evaluation, with genuine wall time.

    No search cache, no silent SDK retries; the notebook chooses a bounded run.
    """
    from .generation import answer_metrics, generate_answer
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Methods must be nonempty and distinct.")
    needed_calls = len(questions) * len(methods) * repeats
    if needed_calls < 1 or needed_calls > max_calls:
        raise ValueError(f"API call limit: need {needed_calls}, permitted {max_calls}.")
    labels_by_id = {row["id"]: row for row in labels}
    if any(row["id"] not in labels_by_id for row in questions):
        raise ValueError("Offline labels are missing for selected questions.")
    kwargs = {key: generation_config[key] for key in ("model", "max_output_tokens", "encoding_name", "prices", "reasoning_effort", "temperature", "service_tier")}
    rng = random.Random(seed)
    results = []
    for repeat in range(repeats):
        for question in questions:
            ordered_methods = list(methods)
            rng.shuffle(ordered_methods)
            for method in ordered_methods:
                with query_scope(audit, "system", question, method, repeat):
                    capture_started = audit.capture_seconds if audit else 0
                    started = time.perf_counter()
                    retrieved = engine.retrieve(question["question"], method)
                    retrieved.update({"question_id": question["id"], "repeat": repeat, "task": question.get("task", "nq")})
                    if audit:
                        retrieved.update({"raw_log": audit.reference(),
                                          "diagnostic_capture_seconds": audit.capture_seconds - capture_started})
                        # Preserve already-retrieved evidence even if validation or the LLM fails.
                        audit.result("retrieval", retrieved)
                    engine.store.validate_evidence(retrieved)
                    generated = generate_answer(retrieved, client=client, **kwargs)
                    elapsed = time.perf_counter() - started
                    capture_seconds = audit.capture_seconds - capture_started if audit else None
                    retrieved["metrics"] = retrieval_metrics(retrieved, labels_by_id[question["id"]])
                    retrieval_cost = retrieved.get("api_cost_usd")
                    if method in ("A", "B", "D") and retrieved.get("api_usage") is None:
                        retrieval_cost = 0.0  # API-only cost; local GPU cost remains unmeasured.
                    generation_cost = generated["api_cost_usd"]
                    result = {"retrieval": retrieved, "generation": generated, "online_total_seconds": elapsed,
                              "retrieval_api_cost_usd": retrieval_cost, "generation_api_cost_usd": generation_cost,
                              "online_api_cost_usd": retrieval_cost + generation_cost if retrieval_cost is not None and generation_cost is not None else None,
                              "answer_metrics": answer_metrics(generated["answer"], labels_by_id[question["id"]]["answers"])}
                    if audit:
                        result.update({"raw_log": audit.reference(), "diagnostic_capture_seconds": capture_seconds,
                                       "timing_note": "wall time includes diagnostic input/output capture; do not subtract it as a counterfactual latency"})
                        audit.result("system", result, label=labels_by_id[question["id"]])
                    results.append(result)
                    if on_result:
                        on_result(result)
                if progress:
                    progress(f"{question['id']} / {method}: API completed")
    return results
