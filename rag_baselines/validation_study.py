"""Bounded live ablations, with versioned answer audits and separate run arms.

Called from notebook/05_measure_abcd.ipynb. Reuses the prepared
corpus/tree, but never reuses retrieval responses as fresh latency samples.
"""
from copy import deepcopy
from pathlib import Path
import json
import random

import pandas as pd  # Must precede Torch/Transformers on this Windows runtime.
import httpx
from openai import OpenAI, DefaultHttpxClient

from .data import read_json, read_jsonl, write_json, file_sha256
from .store import DocumentStore
from .c_tree import load_tree
from .c_retrieval import JevRetrieval
from .jev import JevClient
from .raw_logging import RawLog
from .evaluation import measure_system, runtime_manifest
from .generation import grounding_audit, component_metrics


def selected_questions(root, split, count):
    if split not in ("dev", "eval") or type(count) is not int or count < 1:
        raise ValueError("Choose dev/eval and a positive questions_per_task.")
    all_questions = read_jsonl(root / f"data/prepared/questions_{split}.jsonl")
    selected = []
    for task in ("nq", "hotpotqa"):
        available = [q for q in all_questions if q["task"] == task]
        if count > len(available):
            raise ValueError(f"Too few prepared {split}/{task} questions.")
        selected.extend(available[:count])
    return selected


def summary_rows(records, arm, answer_audit):
    rows = []
    for record in records:
        retrieval, generated = record["retrieval"], record["generation"]
        qid = retrieval["question_id"]
        grounding = grounding_audit(generated["answer"], generated["citations"],
                                   [p["id"] for p in retrieval["evidence"]], generated["valid_output"])
        components = answer_audit["joint_answers"].get(qid)
        extra = component_metrics(generated["answer"], components["components"]) if components else {}
        rows.append({"arm": arm, "question_id": qid, "task": retrieval["task"], "method": retrieval["method"],
            "repeat": retrieval["repeat"], "answer": generated["answer"], **record["answer_metrics"],
            "reviewed_answer_em": extra.get("components_complete", record["answer_metrics"]["answer_em"]), **extra,
            "grounding_status": grounding["status"], "protocol_compliant": grounding["protocol_compliant"],
            "citation_entailment_verified": False,
            "gold_label_needs_semantic_review": qid in answer_audit["gold_semantics_needing_review"],
            **retrieval["metrics"], "evidence_count": len(retrieval["evidence"]),
            "online_seconds": record["online_total_seconds"], "retrieval_seconds": retrieval["seconds"]["retrieval_total"],
            "generation_seconds": sum(generated["seconds"][k] for k in ("prompt_prepare", "api_request")),
            "online_api_cost_usd": record["online_api_cost_usd"],
            "llm_input_tokens": (generated["api_usage"] or {}).get("input_tokens"),
            "decision_input_tokens": (retrieval.get("model_tokens", {}).get("jev") or
                                      retrieval.get("model_tokens", {}).get("laya") or {}).get("nonpad"),
            "decision_calls": len(retrieval.get("trace", {}).get("calls", [])),
            "coverage_reused": sum(event.get("reused", False) for event in retrieval.get("trace", {}).get("events", [])),
            "stop_reason": retrieval.get("trace", {}).get("stop_reason"),
            "raw_events_path": (record.get("raw_log") or {}).get("events_path")})
    return rows


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run_c_development(root, output, study):
    """Randomize C policies within each dev question; change one policy per arm."""
    root, output = Path(root), Path(output)
    if not study["save_raw_logs"] or study["repeats"] != 1:
        raise ValueError("This first study requires raw logging and one declared repetition.")
    questions = selected_questions(root, "dev", study["questions_per_task"]["dev"])
    variants = study["variants"]
    needed = len(questions) * len(variants)
    if needed > study["max_llm_calls"]:
        raise ValueError("Study would exceed the configured LLM call budget.")
    prepared = read_json(root / "data/prepared/manifest.json")
    entry = prepared["corpora"][str(study["corpus_size"])]
    store = DocumentStore(root / entry["path"])
    if store.sha256 != entry["sha256"]:
        raise ValueError("Prepared corpus integrity check failed.")
    c_config = read_json(root / "config/c.json")
    generation = read_json(root / "config/generation.json")
    answer_audit = read_json(root / "config/answer_audit.json")
    tree, preparation = load_tree(root, store, read_json(root / "config/c_taxonomy.json"), c_config)
    labels = read_jsonl(root / "data/prepared/labels_dev.jsonl")
    audit = RawLog(root, manifest={"notebook": "05_measure_abcd.ipynb", "study": study,
        "phase": "development_C_ablations", "corpus_sha256": store.sha256, "generation": generation},
        prices={"jev": c_config["prices"], "openai": generation["prices"]})
    for path in [root / "notebook/05_measure_abcd.ipynb", root / "config/validation_study.json", root / "config/answer_audit.json",
                 root / "config/c.json", root / "config/generation.json", store.path,
                 root / "data/prepared/questions_dev.jsonl", root / "data/prepared/labels_dev.jsonl",
                 *sorted((root / "rag_baselines").glob("*.py"))]:
        audit.snapshot(path)
    directories, records = {}, {name: [] for name in variants}
    with httpx.Client(timeout=c_config["timeout_seconds"], event_hooks=audit.http_hooks) as jev_http, \
         DefaultHttpxClient(timeout=180, event_hooks=audit.http_hooks) as llm_http, \
         OpenAI(max_retries=0, timeout=180, http_client=llm_http) as generator:
        jev = JevClient(c_config, http_client=jev_http)
        engines = {}
        for name, override in variants.items():
            directory = output / "dev" / name
            directory.mkdir(parents=True, exist_ok=False)  # Reruns get a new study directory.
            directories[name] = directory
            config = deepcopy(c_config)
            config["search"].update(override)
            config["search"]["max_api_calls"] = study["max_c_calls_per_question"]
            engines[name] = JevRetrieval(root, store, tree, preparation, config, client=jev)
            write_json(directory / "manifest.json", {"mode": "live_c_ablation", "split": "dev", "arm": name,
                "study": study, "search_signature": engines[name].signature, "c_config": config,
                "generation": generation, "answer_audit": answer_audit,
                "question_ids": [q["id"] for q in questions], "raw_log": str(audit.directory.relative_to(root)),
                "questions_sha256": file_sha256(root / "data/prepared/questions_dev.jsonl"),
                "labels_sha256": file_sha256(root / "data/prepared/labels_dev.jsonl")})
        rng = random.Random(read_json(root / "config/baselines.json")["seed"])
        for index, question in enumerate(questions):
            ordered = list(variants)
            rng.shuffle(ordered)
            for name in ordered:
                audit.event("study", "arm_selected", {"arm": name, "question_id": question["id"]})
                with audit.scope(phase="system", method="C", arm=name):
                    result = measure_system(engines[name], [question], labels, client=generator,
                        generation_config=generation, max_calls=1, methods=("C",), audit=audit, progress=None,
                        on_result=lambda r, name=name: append_record(directories[name] / "system.jsonl", r))[0]
                records[name].append(result)
                r = result["retrieval"]
                print(f"DEV {index+1}/{len(questions)} {name}: EM={result['answer_metrics']['answer_em']:.0f} "
                      f"evidence={len(r['evidence'])} calls={len(r['trace']['calls'])} "
                      f"wall={result['online_total_seconds']:.2f}s", flush=True)
    all_rows = []
    for name, values in records.items():
        rows = summary_rows(values, name, answer_audit)
        pd.DataFrame(rows).to_csv(directories[name] / "questions.csv", index=False)
        write_json(directories[name] / "completed.json", {"records": len(values), "system_sha256": file_sha256(directories[name] / "system.jsonl")})
        all_rows.extend(rows)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "dev_c_questions.csv", index=False)
    summary = aggregate(frame)
    summary.to_csv(output / "dev_c_summary.csv", index=False)
    choice = summary.sort_values(["answer_em", "complete_evidence_hit", "online_api_cost_usd"],
                                 ascending=[False, False, True]).iloc[0]["arm"]
    freeze = {"selected_arm": choice, "search_overrides": variants[choice], "selection_rule": study["selection_rule"],
              "dev_summary_sha256": file_sha256(output / "dev_c_summary.csv"),
              "c_config": engines[choice].config, "search_signature": engines[choice].signature,
              "note": "Development selection is exploratory; superiority is not established by one repetition."}
    write_json(output / "frozen_c_policy.json", freeze)
    audit.event("session", "development_complete", {"selected_arm": choice, "output": str(output.relative_to(root))})
    return summary, freeze


def aggregate(frame):
    return frame.groupby(["arm", "method"], as_index=False).agg(
        questions=("question_id", "nunique"), answer_em=("answer_em", "mean"), answer_f1=("answer_f1", "mean"),
        reviewed_answer_em=("reviewed_answer_em", "mean"), complete_evidence_hit=("complete_evidence_hit", "mean"),
        complete_candidate_hit=("complete_candidate_hit", "mean"), protocol_compliant=("protocol_compliant", "mean"),
        online_seconds=("online_seconds", "mean"), retrieval_seconds=("retrieval_seconds", "mean"),
        llm_input_tokens=("llm_input_tokens", "mean"), decision_input_tokens=("decision_input_tokens", "mean"),
        decision_calls=("decision_calls", "mean"), evidence_count=("evidence_count", "mean"),
        coverage_reused=("coverage_reused", "sum"),
        online_api_cost_usd=("online_api_cost_usd", lambda values: values.mean(skipna=False)))
