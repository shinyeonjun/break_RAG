"""Development-only representation-length treatment for the fixed Jev router.

No evaluation questions, evaluation labels or answer-generation calls are read.
Preparation is checkpointed; every measured retrieval is freshly executed.
"""
from contextlib import ExitStack
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from threading import Lock
from uuid import uuid4
import json
import math
import platform
import time

import pandas as pd  # Preserve the project's Windows native-library import order.
import pyarrow
import httpx
import tiktoken
from openai import OpenAI, DefaultHttpxClient

from .c_tree import digest, load_tree
from .data import read_json, read_jsonl, write_json, file_sha256
from .evaluation import retrieval_metrics
from .generation import usage_cost
from .jev import JevClient, call_usage
from .parallel_routing import ParallelRouting, prepare_groups, routing_quality
from .raw_logging import RawLog
from .store import DocumentStore
from .validation_study import append_record


def validate_length_settings(settings):
    for key in ("corpus_size", "group_size", "selection_k", "evidence_k", "questions_per_task",
                "descriptor_max_output_tokens", "descriptor_input_tokens_limit", "descriptor_preparation_workers",
                "http_workers", "max_calls_per_retrieval", "max_llm_calls", "max_jev_http_calls"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    targets = settings.get("descriptor_token_targets")
    if (not isinstance(targets, list) or len(targets) < 2 or len(set(targets)) != len(targets)
            or any(type(v) is not int or v < 1 for v in targets)):
        raise ValueError("Need at least two distinct positive description targets.")
    if settings.get("provider") != "jev" or settings.get("descriptor_length_mode") != "target_band":
        raise ValueError("This experiment fixes Jev and the common target-band preparation protocol.")
    groups = math.ceil(settings["corpus_size"]/settings["group_size"])
    if settings["selection_k"] > groups:
        raise ValueError("Selection budget exceeds group count.")
    if len(targets)*groups > settings["max_llm_calls"]:
        raise ValueError("Description preparation exceeds its physical API call budget.")
    if settings["descriptor_preparation_workers"] > 8:
        raise ValueError("Description preparation concurrency must be bounded by 8.")


def length_reference(root, settings):
    """Verify the exact development-only sources needed from the prior completed run."""
    directory = (Path(root)/settings["reference_run"]).resolve()
    receipt = read_json(directory/"completed.json")
    names = ("manifest.json", "frozen_policy.json", "preparation.json",
             "baseline_development_quality.json", "development_labels.json")
    source_hashes = {}
    for name in names:
        actual = file_sha256(directory/name)
        if actual != receipt["file_sha256"][name]:
            raise ValueError(f"Reference development artifact changed: {name}")
        source_hashes[name] = actual
    manifest, frozen = read_json(directory/"manifest.json"), read_json(directory/"frozen_policy.json")
    if frozen["settings"]["corpus_size"] != settings["corpus_size"]:
        raise ValueError("Length treatment must use the reference corpus.")
    ids = set(manifest["dev_question_ids"])
    public = read_jsonl(Path(root)/"data/prepared/questions_dev.jsonl")
    questions = []
    for task in ("nq", "hotpotqa"):
        eligible = [q for q in public if q["id"] in ids and q["task"] == task]
        if len(eligible) < settings["questions_per_task"]:
            raise ValueError("Reference run lacks the requested development count; choose a reference with enough baseline questions.")
        questions.extend(eligible[:settings["questions_per_task"]])
    selected_ids = {q["id"] for q in questions}
    labels = read_json(directory/"development_labels.json")
    labels = {qid: labels[qid] for qid in selected_ids}
    baseline = [r for r in read_json(directory/"baseline_development_quality.json")
                if r["question_id"] in selected_ids]
    b_rows = [r for r in baseline if r["method"] == "B"]
    if {r["question_id"] for r in b_rows} != selected_ids or len(b_rows) != len(questions):
        raise ValueError("Reference B must cover each selected development question once.")
    expected_questions = {q["id"]: (q["task"], q["question"]) for q in questions}
    corpus = read_json(Path(root)/"data/prepared/manifest.json")["corpora"][str(settings["corpus_size"])]
    for row in b_rows:
        if (row["task"], row["question"]) != expected_questions[row["question_id"]]:
            raise ValueError("Reference B and length treatment must use identical question text and task.")
        if row["corpus_sha256"] != corpus["sha256"] or row["corpus_count"] != settings["corpus_size"]:
            raise ValueError("Reference B and length treatment must use the identical original corpus.")
        if row["metrics"] != retrieval_metrics(row, labels[row["question_id"]]):
            raise ValueError("Reference B metrics disagree with its original evidence IDs.")
    target = sum(r["metrics"]["complete_evidence_hit"] for r in b_rows)/len(b_rows)
    return directory, questions, labels, baseline, target, frozen, source_hashes


def length_question_row(record):
    retrieval, metrics = record["retrieval"], record["metrics"]
    return {"question_id": record["question_id"], "task": record["task"],
        "descriptor_target_tokens": record["descriptor_target_tokens"], **metrics,
        "macro_seconds": retrieval["seconds"]["macro"], "micro_seconds": retrieval["seconds"]["micro"],
        "retrieval_seconds": retrieval["seconds"]["retrieval_total"],
        "decision_input_tokens": retrieval["model_tokens"]["jev"]["nonpad"],
        "decision_api_cost_usd": retrieval["api_cost_usd"],
        "physical_http_requests": retrieval["trace"]["physical_http_requests"],
        "input_truncated": retrieval["trace"]["input_truncated"]}


def summarize_lengths(frame, preparation, target):
    rows = []
    for length, group in frame.groupby("descriptor_target_tokens"):
        receipt = preparation[str(int(length))]
        macro_hits = int(group.macro_complete.sum())
        rows.append({"descriptor_target_tokens": int(length), "questions": len(group),
            "descriptor_mean_tokens_proxy": receipt["descriptor_mean_tokens_proxy"],
            "descriptor_min_tokens_proxy": receipt["descriptor_min_tokens_proxy"],
            "descriptor_max_tokens_proxy": receipt["descriptor_max_tokens_proxy"],
            "descriptor_target_band_fraction": receipt["descriptor_target_band_fraction"],
            "macro_complete_rate": group.macro_complete.mean(),
            "complete_evidence_hit": group.complete_evidence_hit.mean(),
            "micro_complete_given_macro": group.loc[group.macro_complete == 1, "complete_evidence_hit"].mean() if macro_hits else None,
            "macro_tpr": group.macro_tp.sum()/group.macro_positive.sum(),
            "macro_fpr": group.macro_fp.sum()/group.macro_negative.sum() if group.macro_negative.sum() else None,
            "development_target": target, "target_met": bool(group.complete_evidence_hit.mean() >= target),
            **{key: group[key].mean() for key in ("macro_seconds", "micro_seconds", "retrieval_seconds",
                   "decision_input_tokens", "decision_api_cost_usd", "physical_http_requests")},
            "input_truncated_count": int(group.input_truncated.sum())})
    return pd.DataFrame(rows)


def run_length_study(root, settings=None, progress=print):
    root = Path(root).resolve()
    settings = read_json(root/"config/representation_length.json") if settings is None else settings
    validate_length_settings(settings)
    reference, questions, labels, baseline, target, frozen, hashes = length_reference(root, settings)
    generation, c_config = read_json(root/"config/generation.json"), read_json(root/"config/c.json")
    if generation != frozen["generation"] or c_config != frozen["c_model"]:
        raise ValueError("Keep generation and Jev versions/settings fixed for this length treatment.")
    entry = read_json(root/"data/prepared/manifest.json")["corpora"][str(settings["corpus_size"])]
    store = DocumentStore(root/entry["path"])
    if store.sha256 != entry["sha256"] or len(store.passages) != settings["corpus_size"]:
        raise ValueError("Common corpus integrity check failed.")
    if any(pid not in store.indices for label in labels.values() for group in label["evidence_groups"] for pid in group):
        raise ValueError("Development evidence is outside the common corpus.")
    from dotenv import load_dotenv
    load_dotenv(root/".env", override=False)
    directory = root/"results"/f"length_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    audit = RawLog(root, manifest={"notebook": "07_mathematical_framework.ipynb", "settings": settings,
        "phase": "representation_length_development", "question_ids": [q["id"] for q in questions]},
        prices={"jev": c_config["prices"], "openai": generation["prices"]})
    lock, attempts = Lock(), {"jev": 0, "openai": 0}
    def request_hook(request):
        provider = "jev" if json.loads(request.content).get("model", "").startswith("jev-") else "openai"
        key = "max_jev_http_calls" if provider == "jev" else "max_llm_calls"
        with lock:
            if attempts[provider] >= settings[key]:
                raise RuntimeError("Length experiment's physical API budget exhausted.")
            attempts[provider] += 1
            audit.http_request(request)
    def response_hook(response):
        with lock:
            audit.http_response(response)
    class PreparationAudit:
        def result(self, stage, payload):
            with lock:
                return audit.result(stage, payload)
    hooks = {"request": [request_hook], "response": [response_hook]}
    for path in [root/"config/representation_length.json", root/"config/c.json", root/"config/generation.json",
                 root/"rag_baselines/representation_length.py", root/"rag_baselines/parallel_routing.py",
                 root/"notebook/07_mathematical_framework.ipynb"]:
        audit.snapshot(path)
    try:
        with ExitStack() as stack:
            generator = stack.enter_context(OpenAI(max_retries=0, timeout=180,
                http_client=stack.enter_context(DefaultHttpxClient(timeout=180, event_hooks=hooks))))
            jev = JevClient(c_config, http_client=stack.enter_context(httpx.Client(timeout=c_config["timeout_seconds"], event_hooks=hooks)))
            tree, inherited = load_tree(root, store, read_json(root/"config/c_taxonomy.json"), c_config)
            write_json(directory/"manifest.json", {"mode": "representation_length_development", "split": "dev",
                "created_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
                "question_ids": [q["id"] for q in questions], "corpus_sha256": store.sha256,
                "group_size": settings["group_size"], "selection_k": settings["selection_k"],
                "generation": generation, "c_model": c_config, "development_target": target,
                "reference_run": str(reference.relative_to(root)), "reference_source_sha256": hashes,
                "python": platform.python_version(), "inherited_preparation": inherited,
                "raw_log": str(audit.directory.relative_to(root)), "evaluation_labels_loaded": False,
                "scope": "All conditions use new common length-band descriptors and fresh randomized development retrieval. No answer generation, no held-out performance claim."})
            write_json(directory/"development_labels.json", labels)
            write_json(directory/"baseline_development_quality.json", baseline)
            preparation, all_groups, members = {}, {}, None
            for length in settings["descriptor_token_targets"]:
                with audit.scope(phase="length_preparation", method="shared", descriptor_target_tokens=length):
                    started, calls_before = time.perf_counter(), attempts["openai"]
                    groups, receipt, cache = prepare_groups(root, store, tree, settings["group_size"],
                        {**settings, "descriptor_target_tokens": length}, generation, generator, PreparationAudit(), progress)
                    membership = [(g["id"], g["passage_ids"]) for g in groups]
                    if members is not None and membership != members:
                        raise ValueError("Group membership changed between length conditions.")
                    members = membership
                    actual = [g["descriptor_text_tokens_proxy"] for g in groups]
                    preparation[str(length)] = {**receipt, "directory": str(cache.relative_to(root)),
                        "actual_new_calls_this_session": attempts["openai"]-calls_before,
                        "prepare_wall_seconds_this_session": time.perf_counter()-started,
                        "descriptor_mean_tokens_proxy": sum(actual)/len(actual),
                        "descriptor_min_tokens_proxy": min(actual), "descriptor_max_tokens_proxy": max(actual),
                        "descriptor_target_band_fraction": sum(.8*length <= n <= 1.2*length for n in actual)/len(actual),
                        "length_band_note": "80–120% is an observed manipulation check, not forced clipping or a quality guarantee."}
                    all_groups[length] = groups
                    write_json(directory/"preparation.json", preparation)
            policy = {key: settings[key] for key in ("selection_k", "evidence_k", "max_calls_per_retrieval", "http_workers")}
            adapters = {length: ParallelRouting(store, groups, jev, "jev", policy, directory/"decision_calls.jsonl", audit)
                        for length, groups in all_groups.items()}
            write_json(directory/"fixed_conditions.json", {"group_membership_sha256": digest(members),
                "groups_sha256": {str(length): digest(groups) for length, groups in all_groups.items()},
                "policy": policy, "c_model": c_config, "generation": generation,
                "code_sha256": {p.name: file_sha256(p) for p in (Path(__file__), root/"rag_baselines/parallel_routing.py")}})
            rng, rows = Random(settings["seed"]), []
            for index, question in enumerate(questions):
                lengths = list(adapters)
                rng.shuffle(lengths)
                for length in lengths:
                    adapter = adapters[length]
                    with audit.scope(phase="length_development", method=adapter.method, descriptor_target_tokens=length,
                            question_id=question["id"], question=question["question"], task=question["task"], repeat=0):
                        retrieval = adapter.retrieve(question["question"])
                        metrics = routing_quality(retrieval, all_groups[length], labels[question["id"]], store)
                        record = {"descriptor_target_tokens": length, "question_id": question["id"],
                                  "task": question["task"], "retrieval": retrieval, "metrics": metrics}
                        append_record(directory/"retrievals.jsonl", record)
                        rows.append(length_question_row(record))
                if progress:
                    progress(f"Fresh length retrieval {index+1}/{len(questions)} questions · {len(rows)} observations")
            frame = pd.DataFrame(rows)
            frame.to_csv(directory/"questions.csv", index=False)
            summary = summarize_lengths(frame, preparation, target)
            summary.to_csv(directory/"summary.csv", index=False)
            write_json(directory/"completed.json", {"records": len(rows), "physical_api_attempts": attempts,
                "raw_log": str(audit.directory.relative_to(root)),
                "file_sha256": {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()},
                "scope": "Exploratory development treatment, one execution per question/target. No evaluation labels or answer-generation calls."})
            (root/"results/current_length_run.txt").write_text(str(directory.relative_to(root)), encoding="utf-8")
            return directory
    except Exception as error:
        write_json(directory/"failed.json", {"error_type": type(error).__name__, "physical_api_attempts": attempts,
            "raw_log": str(audit.directory.relative_to(root)), "partial_results_preserved": True})
        raise


def load_length_report(root, directory):
    root, directory = Path(root), Path(directory)
    receipt, manifest = read_json(directory/"completed.json"), read_json(directory/"manifest.json")
    for name, expected in receipt["file_sha256"].items():
        if file_sha256(directory/name) != expected:
            raise ValueError(f"Length result integrity check failed: {name}")
    store_entry = read_json(root/"data/prepared/manifest.json")["corpora"][str(manifest["settings"]["corpus_size"])]
    store = DocumentStore(root/store_entry["path"])
    if store.sha256 != manifest["corpus_sha256"]:
        raise ValueError("Length report's original corpus changed.")
    labels, preparation = read_json(directory/"development_labels.json"), read_json(directory/"preparation.json")
    groups = {}
    members = None
    for length, prepared in preparation.items():
        path = root/prepared["directory"]/"groups.json"
        if file_sha256(path) != prepared["groups_sha256"]:
            raise ValueError("Length report's prepared groups changed.")
        groups[int(length)] = read_json(path)
        encoder = tiktoken.get_encoding(manifest["generation"]["encoding_name"])
        actual = [len(encoder.encode(json.dumps(g["descriptor"],ensure_ascii=False))) for g in groups[int(length)]]
        expected_length = {"descriptor_mean_tokens_proxy":sum(actual)/len(actual),
            "descriptor_min_tokens_proxy":min(actual),"descriptor_max_tokens_proxy":max(actual),
            "descriptor_target_band_fraction":sum(.8*int(length)<=n<=1.2*int(length) for n in actual)/len(actual)}
        if any(prepared[key]!=value for key,value in expected_length.items()):
            raise ValueError("Prepared descriptor length statistics disagree with the actual text.")
        actual_members = [(g["id"], g["passage_ids"]) for g in groups[int(length)]]
        if members is not None and actual_members != members:
            raise ValueError("Length report compares different memberships.")
        members = actual_members
    fixed = read_json(directory/"fixed_conditions.json")
    if digest(members) != fixed["group_membership_sha256"] or any(digest(value) != fixed["groups_sha256"][str(length)] for length,value in groups.items()):
        raise ValueError("Length report's fixed group conditions changed.")
    records = read_jsonl(directory/"retrievals.jsonl")
    keys = [(r["question_id"], r["descriptor_target_tokens"]) for r in records]
    expected = {(qid, length) for qid in manifest["question_ids"] for length in manifest["settings"]["descriptor_token_targets"]}
    if len(records) != receipt["records"] or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unexpected length observations.")
    for record in records:
        retrieval = record["retrieval"]
        store.validate_evidence(retrieval)
        if routing_quality(retrieval, groups[record["descriptor_target_tokens"]], labels[record["question_id"]], store) != record["metrics"]:
            raise ValueError("Length metrics disagree with original evidence.")
        usage, cost = call_usage(retrieval["trace"]["calls"], manifest["c_model"]["prices"])
        if usage != retrieval["api_usage"] or cost != retrieval["api_cost_usd"]:
            raise ValueError("Length usage/cost disagrees with the saved provider calls.")
        if retrieval["execution_mode"] != "online_retrieval" or retrieval["seconds"]["retrieval_total"] is None:
            raise ValueError("Length timing must use fresh retrieval.")
        signature = retrieval["trace"]["signature"]
        if signature["policy"] != fixed["policy"] or signature["groups_sha256"] != fixed["groups_sha256"][str(record["descriptor_target_tokens"])]:
            raise ValueError("Length retrieval used different fixed conditions.")
    raw_directory = (root/manifest["raw_log"]).resolve()
    ledger = read_jsonl(raw_directory/"api_calls.jsonl")
    if len({x["call_id"] for x in ledger}) != len(ledger) or Counter(x["provider"] for x in ledger) != Counter(receipt["physical_api_attempts"]):
        raise ValueError("Physical length API ledger has duplicate or missing attempts.")
    costs, requests = defaultdict(float), Counter()
    for row in ledger:
        for boundary in ("request", "response"):
            body = row[boundary]
            path = (raw_directory/body["path"]).resolve()
            if not path.is_relative_to(raw_directory) or file_sha256(path) != body["saved_sha256"]:
                raise ValueError("Physical API body integrity check failed.")
            if not body["redacted"] and body["saved_sha256"] != body["source_sha256"]:
                raise ValueError("Unredacted API body disagrees with its original hash.")
        response = read_json(raw_directory/row["response"]["path"])
        response_cost = (call_usage([{"response": response}], manifest["c_model"]["prices"])[1]
                         if row["provider"] == "jev" else usage_cost(response.get("usage"), manifest["generation"]["prices"]))
        if response_cost is None or row["api_cost_usd"] is None or not math.isclose(response_cost, row["api_cost_usd"], abs_tol=1e-12):
            raise ValueError("Physical API charges disagree with its raw response.")
        if row["phase"] == "length_development":
            if row["provider"] != "jev" or row["api_cost_usd"] is None or row["http_status"] != 200:
                raise ValueError("Measured length judgment has an unknown or failed API charge.")
            key = row["question_id"],row["descriptor_target_tokens"]
            costs[key] += row["api_cost_usd"]
            requests[key] += 1
    for record in records:
        key = record["question_id"],record["descriptor_target_tokens"]
        if not math.isclose(costs[key],record["retrieval"]["api_cost_usd"],abs_tol=1e-12) or requests[key] != record["retrieval"]["trace"]["physical_http_requests"]:
            raise ValueError("Physical API charges disagree with the question's recorded retrieval.")
    frame = pd.DataFrame([length_question_row(r) for r in records])
    summary = summarize_lengths(frame, preparation, manifest["development_target"])
    pd.testing.assert_frame_equal(frame, pd.read_csv(directory/"questions.csv"), check_dtype=False, atol=1e-10)
    pd.testing.assert_frame_equal(summary, pd.read_csv(directory/"summary.csv"), check_dtype=False, atol=1e-10)
    return frame, summary, manifest, preparation
