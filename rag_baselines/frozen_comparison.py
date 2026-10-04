"""Notebook 08: fresh end-to-end evaluation of the condition selected in 07.

Reuse original group descriptions and retrieval/generation engines. No new
development sweep, description generation, or test-set policy selection.
"""
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
import json
import math
import random

import pandas as pd  # Arrow must precede Torch on this Windows environment.
import pyarrow
import torch
import httpx
from openai import OpenAI, DefaultHttpxClient

from .c_tree import digest
from .data import read_json, read_jsonl, write_json, file_sha256
from .evaluation import measure_system, retrieval_metrics, runtime_manifest
from .generation import answer_metrics, build_prompt, usage_cost
from .jev import JevClient, call_usage
from .laya_local import LayaDecisionClient
from .models import Baselines
from .parallel_experiment import ParallelComparison, unseen_evaluation_questions
from .parallel_routing import ParallelRouting, routing_quality, validate_groups
from .raw_logging import RawLog, query_scope
from .representation_length import load_length_report
from .store import DocumentStore
from .validation_study import aggregate, append_record, summary_rows


METHODS = ("A", "B", "C-parallel", "D-parallel")
ARM = "frozen_length_heldout"


def validate_frozen_settings(settings):
    for key in ("descriptor_target_tokens", "questions_per_task", "repeats", "max_llm_calls", "max_jev_http_calls"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if type(settings.get("seed")) is not int:
        raise ValueError("Use an integer seed.")
    if 2*settings["questions_per_task"]*len(METHODS)*settings["repeats"] > settings["max_llm_calls"]:
        raise ValueError("Held-out generation exceeds the declared LLM call budget.")


def frozen_reference(root, settings):
    """Select only from audited development results; never read evaluation gold."""
    root = Path(root).resolve()
    validate_frozen_settings(settings)
    source = (root/settings["reference_length_run"]).resolve()
    if not source.is_relative_to(root):
        raise ValueError("Reference must be inside this workspace.")
    _, summary, manifest, preparation = load_length_report(root, source)
    feasible = summary[summary.target_met].sort_values(["decision_api_cost_usd", "descriptor_target_tokens"])
    if feasible.empty or int(feasible.iloc[0].descriptor_target_tokens) != settings["descriptor_target_tokens"]:
        raise ValueError("Choose the lowest-cost feasible tested development condition; do not select on evaluation results.")
    target = str(settings["descriptor_target_tokens"])
    prepared = preparation[target]
    cache = (root/prepared["directory"]).resolve()
    if not cache.is_relative_to(root):
        raise ValueError("Prepared descriptions must be inside this workspace.")
    groups = read_json(cache/"groups.json")
    fixed = read_json(source/"fixed_conditions.json")
    if file_sha256(cache/"groups.json") != prepared["groups_sha256"] or digest(groups) != fixed["groups_sha256"][target]:
        raise ValueError("Selected descriptions differ from the audited length experiment.")
    corpus = read_json(root/"data/prepared/manifest.json")["corpora"][str(manifest["settings"]["corpus_size"])]
    store = DocumentStore(root/corpus["path"])
    if store.sha256 != manifest["corpus_sha256"]:
        raise ValueError("The original corpus changed.")
    validate_groups(groups, store)
    if file_sha256(root/"rag_baselines/parallel_routing.py") != fixed["code_sha256"]["parallel_routing.py"]:
        raise ValueError("Routing code changed after the length experiment; declare a separate treatment.")
    generation, c_config, d_config = (read_json(root/f"config/{name}.json") for name in ("generation", "c", "d_laya"))
    previous = root/manifest["reference_run"]
    prior_manifest, prior_policy = read_json(previous/"manifest.json"), read_json(previous/"frozen_policy.json")
    if generation != manifest["generation"] or c_config != manifest["c_model"] or d_config != prior_policy["d_model"]:
        raise ValueError("Keep generator and decision model settings fixed.")
    baseline_config = read_json(root/"config/baselines.json")
    if baseline_config != prior_manifest["config"] or baseline_config["evidence_k"] != fixed["policy"]["evidence_k"]:
        raise ValueError("Keep baseline revisions and the common final evidence budget fixed.")
    chosen = feasible.iloc[0]
    policy = {"settings": settings, "routing_policy": fixed["policy"],
        "corpus_size": len(store.passages), "corpus_path": corpus["path"], "corpus_sha256": store.sha256,
        "group_size": manifest["group_size"], "groups_directory": str(cache.relative_to(root)),
        "groups_sha256": digest(groups), "groups_file_sha256": file_sha256(cache/"groups.json"),
        "descriptor_target_tokens": int(target), "descriptor_mean_tokens_proxy": prepared["descriptor_mean_tokens_proxy"],
        "development_questions": int(chosen.questions), "development_evidence_rate": float(chosen.complete_evidence_hit),
        "development_target": float(chosen.development_target),
        "selection_note": "Lowest observed judgment API cost among three feasible/tested development lengths; not a global optimum.",
        "d_policy_note": "Same descriptions and routing budget as C; this is an untuned transfer to Laya, not its development optimum.",
        "baseline": baseline_config, "generation": generation, "c_model": c_config, "d_model": d_config,
        "evaluation_labels_loaded": False,
        "source_reference_sha256": {str(p.relative_to(root)): file_sha256(p) for p in (
            source/"completed.json", source/"summary.csv", source/"fixed_conditions.json", source/"preparation.json")},
        "code_sha256": {str(p.relative_to(root)): file_sha256(p) for p in (root/"rag_baselines").glob("*.py")}}
    return policy, groups, store, manifest, prepared


def question_frame(records, answer_audit):
    rows = summary_rows(records, ARM, answer_audit)
    for row, record in zip(rows, records):
        retrieval = record["retrieval"]
        row.update({"generation_api_cost_usd": record["generation_api_cost_usd"],
            "retrieval_api_cost_usd": record["retrieval_api_cost_usd"],
            "evidence_text_tokens": record["generation"]["prompt"]["evidence_text_tokens"],
            "macro_complete": retrieval.get("routing_metrics", {}).get("macro_complete"),
            "input_truncated": retrieval.get("trace", {}).get("input_truncated", False),
            "physical_http_requests": retrieval.get("trace", {}).get("physical_http_requests", 0),
            "local_peak_allocated_bytes": retrieval.get("trace", {}).get("local_peak_allocated_bytes")})
    return pd.DataFrame(rows)


def run_frozen_comparison(root, settings=None, progress=print):
    root = Path(root).resolve()
    settings = read_json(root/"config/frozen_comparison.json") if settings is None else settings
    policy, groups, store, reference, prepared = frozen_reference(root, settings)
    questions, excluded = unseen_evaluation_questions(root, settings["questions_per_task"])
    dev_ids = {q["id"] for q in read_jsonl(root/"data/prepared/questions_dev.jsonl")}
    if dev_ids & {q["id"] for q in questions}:
        raise ValueError("Evaluation overlaps development questions.")
    random.Random(settings["seed"]).shuffle(questions)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 4*1024**3:
        raise RuntimeError("This comparison needs CUDA and 4 GiB free GPU memory.")
    from dotenv import load_dotenv
    load_dotenv(root/".env", override=False)
    directory = root/"results"/f"frozen_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory/"frozen_policy.json", policy)
    write_json(directory/"policy_before_eval_labels.json", {"split": "eval", "question_ids": [q["id"] for q in questions],
        "frozen_policy_sha256": file_sha256(directory/"frozen_policy.json"), "evaluation_labels_loaded": False})
    write_json(directory/"manifest.json", {"mode": "frozen_length_heldout", "split": "eval", "methods": METHODS,
        "settings": settings, "question_ids": [q["id"] for q in questions], "excluded_historical_eval_ids": excluded,
        "development_question_ids": sorted(dev_ids), "corpus_sha256": store.sha256,
        "evaluation_labels_loaded": False, "scope": "Exploratory fresh evaluation on a gold-inclusive 2,000-passage corpus."})
    write_json(directory/"evaluation_questions.json", questions)
    labels_by_id = {r["id"]: r for r in read_jsonl(root/"data/prepared/labels_eval.jsonl")}
    labels = {q["id"]: labels_by_id[q["id"]] for q in questions}
    if any(not any(set(g) <= store.indices.keys() for g in label["evidence_groups"]) for label in labels.values()):
        raise ValueError("A selected question has no complete gold evidence in the common corpus.")
    write_json(directory/"evaluation_labels.json", labels)
    audit = RawLog(root, manifest={"notebook": "08_compare_frozen_routing.ipynb", "settings": settings,
        "question_ids": [q["id"] for q in questions], "frozen_policy_sha256": file_sha256(directory/"frozen_policy.json")},
        prices={"jev": policy["c_model"]["prices"], "openai": policy["generation"]["prices"]})
    lock, attempts = Lock(), {"jev": 0, "openai": 0}
    def request_hook(request):
        provider = "jev" if json.loads(request.content).get("model", "").startswith("jev-") else "openai"
        budget = settings["max_jev_http_calls" if provider == "jev" else "max_llm_calls"]
        with lock:
            if attempts[provider] >= budget:
                raise RuntimeError("Frozen comparison's physical API budget exhausted.")
            attempts[provider] += 1
            audit.http_request(request)
    def response_hook(response):
        with lock:
            audit.http_response(response)
    hooks = {"request": [request_hook], "response": [response_hook]}
    for path in [root/"config/frozen_comparison.json", root/"notebook/08_compare_frozen_routing.ipynb",
                 directory/"frozen_policy.json", store.path, *sorted((root/"rag_baselines").glob("*.py"))]:
        audit.snapshot(path)
    try:
        with ExitStack() as stack:
            generator = stack.enter_context(OpenAI(max_retries=0, timeout=180,
                http_client=stack.enter_context(DefaultHttpxClient(timeout=180, event_hooks=hooks))))
            jev = JevClient(policy["c_model"], http_client=stack.enter_context(
                httpx.Client(timeout=policy["c_model"]["timeout_seconds"], event_hooks=hooks)))
            baseline = Baselines(root, device="cuda", evidence_k=policy["routing_policy"]["evidence_k"], raw_log=audit)
            baseline.build_index(store.path)
            laya = LayaDecisionClient(policy["d_model"], device=baseline.device)
            write_json(directory/"runtime.json", {**runtime_manifest(root, baseline), "laya": laya.runtime_info()})
            write_json(directory/"preparation.json", {"descriptions": prepared, "inherited_classification": reference["inherited_preparation"],
                "actual_new_description_calls": 0, "groups_shared_by": ["C-parallel", "D-parallel"],
                "note": "Historical preparation is reported separately; descriptor generation is not repeated or charged online."})
            write_json(directory/"input_audit.json", baseline.truncation_audit(questions))
            adapters = {}
            for provider, client in (("jev", jev), ("laya", laya)):
                adapter = ParallelRouting(store, groups, client, provider, policy["routing_policy"], directory/"decision_calls.jsonl", audit)
                adapters[adapter.method] = adapter
            engine = ParallelComparison(baseline, adapters)
            warmup_question = read_jsonl(root/"data/prepared/questions_dev.jsonl")[0]
            warmup = []
            for method in METHODS:
                with query_scope(audit, "warmup", warmup_question, method):
                    result = engine.retrieve(warmup_question["question"], method)
                    warmup.append({"method": method, "question_id": warmup_question["id"], "retrieval": result})
            write_json(directory/"warmup.json", warmup)
            if progress:
                progress(f"Frozen target={policy['descriptor_target_tokens']}, g={policy['group_size']}, K={policy['routing_policy']['selection_k']}; {len(questions)} unseen questions")
            def save_record(record):
                r = record["retrieval"]
                if r["method"] in adapters:
                    r["routing_metrics"] = routing_quality(r, groups, labels[r["question_id"]], store)
                append_record(directory/"system.jsonl", record)
            records = measure_system(engine, questions, list(labels.values()), client=generator,
                generation_config=policy["generation"], max_calls=settings["max_llm_calls"]-attempts["openai"],
                repeats=settings["repeats"], methods=METHODS, seed=settings["seed"], audit=audit, progress=progress,
                on_result=save_record)
            answer_audit = read_json(root/"config/answer_audit.json")
            write_json(directory/"answer_audit.json", answer_audit)
            frame = question_frame(records, answer_audit)
            frame.to_csv(directory/"questions.csv", index=False)
            aggregate(frame).to_csv(directory/"summary.csv", index=False)
            manifest = read_json(directory/"manifest.json")
            write_json(directory/"manifest.json", {**manifest, "raw_log": str(audit.directory.relative_to(root)),
                "evaluation_labels_loaded": True, "physical_api_attempts": attempts})
            write_json(directory/"completed.json", {"records": len(records), "physical_api_attempts": attempts,
                "raw_log": str(audit.directory.relative_to(root)),
                "file_sha256": {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()},
                "note": "One exploratory hardware run. Development feasibility does not establish held-out superiority."})
        load_frozen_report(root, directory)
        (root/"results/current_frozen_run.txt").write_text(str(directory.relative_to(root)), encoding="utf-8")
        return directory
    except Exception as error:
        write_json(directory/"failed.json", {"error_type": type(error).__name__, "physical_api_attempts": attempts,
            "raw_log": str(audit.directory.relative_to(root)), "partial_results_preserved": True})
        raise


def load_frozen_report(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if (directory/"failed.json").exists():
        raise ValueError("This comparison failed; use a validated completed run.")
    receipt = read_json(directory/"completed.json")
    for name, expected in receipt["file_sha256"].items():
        if file_sha256(directory/name) != expected:
            raise ValueError(f"Frozen result integrity check failed: {name}")
    policy, before = read_json(directory/"frozen_policy.json"), read_json(directory/"policy_before_eval_labels.json")
    if before["evaluation_labels_loaded"] or file_sha256(directory/"frozen_policy.json") != before["frozen_policy_sha256"]:
        raise ValueError("Policy was not preserved from before evaluation labels were loaded.")
    for relative, expected in policy["source_reference_sha256"].items():
        if file_sha256(root/relative) != expected:
            raise ValueError("The source development experiment changed.")
    groups_path = root/policy["groups_directory"]/"groups.json"
    groups, store = read_json(groups_path), DocumentStore(root/policy["corpus_path"])
    if file_sha256(groups_path) != policy["groups_file_sha256"] or store.sha256 != policy["corpus_sha256"]:
        raise ValueError("Original corpus or frozen descriptions changed.")
    manifest, labels = read_json(directory/"manifest.json"), read_json(directory/"evaluation_labels.json")
    expected_ids = set(before["question_ids"])
    if expected_ids & set(manifest["excluded_historical_eval_ids"]+manifest["development_question_ids"]):
        raise ValueError("Held-out questions overlap an earlier evaluation or development split.")
    records = read_jsonl(directory/"system.jsonl")
    keys = [(r["retrieval"]["question_id"], r["retrieval"]["method"], r["retrieval"]["repeat"]) for r in records]
    expected = {(qid, method, rep) for qid in expected_ids for method in METHODS for rep in range(policy["settings"]["repeats"])}
    if len(records) != receipt["records"] or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unexpected observations.")
    by_key = {}
    for record in records:
        r, generated = record["retrieval"], record["generation"]
        store.validate_evidence(r)
        label = labels[r["question_id"]]
        if r["metrics"] != retrieval_metrics(r, label) or record["answer_metrics"] != answer_metrics(generated["answer"], label["answers"]):
            raise ValueError("Recorded quality disagrees with original evidence/answers.")
        if generated["prompt"] != build_prompt(r, encoding_name=policy["generation"]["encoding_name"]):
            raise ValueError("Final generator prompt differs from the original selected evidence.")
        if record["generation_api_cost_usd"] != usage_cost(generated["api_usage"], generated["prices"]):
            raise ValueError("Generator usage charges disagree with the recorded cost.")
        if r["execution_mode"] != "online_retrieval" or r["seconds"]["retrieval_total"] is None:
            raise ValueError("Held-out latency must use fresh retrieval.")
        if r["method"] in ("C-parallel", "D-parallel"):
            signature = r["trace"]["signature"]
            if signature["groups_sha256"] != policy["groups_sha256"] or signature["policy"] != policy["routing_policy"] or r["trace"]["input_truncated"]:
                raise ValueError("Frozen routing conditions changed or input was clipped.")
            if r["routing_metrics"] != routing_quality(r, groups, label, store):
                raise ValueError("Routing quality disagrees with original group membership.")
            if r["method"] == "C-parallel":
                usage, cost = call_usage(r["trace"]["calls"], policy["c_model"]["prices"])
                if usage != r["api_usage"] or cost != record["retrieval_api_cost_usd"]:
                    raise ValueError("Jev usage/cost does not match its physical responses.")
        total = record["retrieval_api_cost_usd"]+record["generation_api_cost_usd"]
        if not math.isclose(total, record["online_api_cost_usd"], abs_tol=1e-12):
            raise ValueError("Online fee components do not sum to the total.")
        by_key[r["question_id"], r["method"], r["repeat"]] = record
    for qid in expected_ids:
        for rep in range(policy["settings"]["repeats"]):
            a, b = (by_key[qid, m, rep]["retrieval"]["candidates"] for m in ("A", "B"))
            if [(p["id"],p["dense_score"]) for p in a] != [(p["id"],p["dense_score"]) for p in b]:
                raise ValueError("A/B independently executed dense searches returned different candidates.")
    raw = root/receipt["raw_log"]
    ledger = read_jsonl(raw/"api_calls.jsonl")
    if len({r["call_id"] for r in ledger}) != len(ledger) or Counter(r["provider"] for r in ledger) != Counter(receipt["physical_api_attempts"]):
        raise ValueError("Physical API ledger contains missing or duplicate attempts.")
    fees, counts = defaultdict(float), Counter()
    for row in ledger:
        for boundary in ("request", "response"):
            body = row[boundary]
            path = (raw/body["path"]).resolve()
            if not path.is_relative_to(raw) or file_sha256(path) != body["saved_sha256"]:
                raise ValueError("Physical API body integrity check failed.")
            if not body["redacted"] and body["saved_sha256"] != body["source_sha256"]:
                raise ValueError("Unredacted physical body differs from its original hash.")
        response = read_json(raw/row["response"]["path"])
        cost = (call_usage([{"response": response}], policy["c_model"]["prices"])[1] if row["provider"] == "jev"
                else usage_cost(response.get("usage"), policy["generation"]["prices"]))
        if row["http_status"] != 200 or cost is None or not math.isclose(cost, row["api_cost_usd"], abs_tol=1e-12):
            raise ValueError("A physical response has failed or inconsistent billing.")
        if row["phase"] == "system":
            key = row["question_id"],row["method"],row["repeat"]
            fees[key] += cost
            counts[key,row["provider"]] += 1
    for key, record in by_key.items():
        if not math.isclose(fees[key], record["online_api_cost_usd"], abs_tol=1e-12) or counts[key,"openai"] != 1:
            raise ValueError("Question fee disagrees with the physical API ledger.")
        if key[1] == "C-parallel" and counts[key,"jev"] != record["retrieval"]["trace"]["physical_http_requests"]:
            raise ValueError("Jev request count disagrees with the physical API ledger.")
    frame = question_frame(records, read_json(directory/"answer_audit.json"))
    summary = aggregate(frame)
    pd.testing.assert_frame_equal(frame, pd.read_csv(directory/"questions.csv"), check_dtype=False, atol=1e-10)
    pd.testing.assert_frame_equal(summary, pd.read_csv(directory/"summary.csv"), check_dtype=False, atol=1e-10)
    return frame, summary, policy, manifest
