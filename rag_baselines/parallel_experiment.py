"""Notebook 07's bounded development sweep and fresh held-out comparison."""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4
import json
import time

import pandas as pd  # Windows native-DLL order: import Arrow before Torch.
import pyarrow
import torch
import httpx
from openai import OpenAI, DefaultHttpxClient

from .c_tree import digest, load_tree
from .data import read_json, read_jsonl, write_json, file_sha256
from .evaluation import evaluate_cached_quality, measure_system, runtime_manifest, retrieval_metrics
from .generation import answer_metrics, usage_cost
from .jev import JevClient
from .laya_local import LayaDecisionClient
from .models import Baselines
from .parallel_routing import ParallelRouting, prepare_groups, routing_quality
from .raw_logging import RawLog, query_scope
from .validation_study import selected_questions, append_record, summary_rows, aggregate


def validate_settings(settings):
    for key in ("corpus_size", "evidence_k", "descriptor_target_tokens", "descriptor_max_output_tokens",
                "descriptor_input_tokens_limit", "http_workers", "max_calls_per_retrieval", "max_llm_calls", "max_jev_http_calls"):
        if type(settings.get(key)) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("group_sizes", "selection_budgets"):
        values = settings.get(key)
        if not isinstance(values, list) or not values or len(values) != len(set(values)) or any(type(v) is not int or v < 1 for v in values):
            raise ValueError(f"{key} must contain distinct positive integers.")
    if max(settings["selection_budgets"]) > min((settings["corpus_size"] + g - 1)//g for g in settings["group_sizes"]):
        raise ValueError("Selection budget exceeds the number of groups.")
    overrides = settings.get("descriptor_max_output_tokens_by_group_size", {})
    if (not isinstance(overrides, dict) or any(str(g) not in {str(s) for s in settings["group_sizes"]}
            or type(v) is not int or v < 1 for g, v in overrides.items())):
        raise ValueError("Description output overrides must use declared group sizes and positive budgets.")
    counts = settings["questions_per_task"]
    if set(counts) != {"dev", "eval"} or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("Declare positive dev/eval questions_per_task.")
    required_llm = sum((settings["corpus_size"] + g - 1)//g for g in settings["group_sizes"]) + 8*counts["eval"]
    if required_llm > settings["max_llm_calls"]:
        raise ValueError("Description preparation plus held-out generation exceeds the configured call budget.")


def unseen_evaluation_questions(root, count):
    """Read public questions and previous execution IDs, never gold labels."""
    root = Path(root)
    historical = set()
    paths = list((root / "results").glob("staged_*/eval/*/manifest.json"))
    paths += list((root / "results").glob("*/manifest.json"))
    paths += list((root / "results").glob("parallel_*/policy_before_eval_labels.json"))
    for path in paths:
        manifest = read_json(path)
        if manifest.get("split") == "eval":
            historical.update(manifest.get("question_ids", []))
    questions = read_jsonl(root / "data/prepared/questions_eval.jsonl")
    selected = []
    for task in ("nq", "hotpotqa"):
        eligible = [q for q in questions if q["task"] == task and q["id"] not in historical]
        if len(eligible) < count:
            raise ValueError("Not enough previously unevaluated questions; declare a separate exploratory re-evaluation.")
        selected.extend(eligible[:count])
    return selected, sorted(historical)


def summarize_quality(frame):
    rows = []
    for (provider, size, k), group in frame.groupby(["provider", "group_size", "selection_k"]):
        macro_hits = int(group.macro_complete.sum())
        rows.append({"provider": provider, "group_size": int(size), "selection_k": int(k),
            "questions": len(group), "macro_complete_rate": group.macro_complete.mean(),
            "complete_evidence_hit": group.complete_evidence_hit.mean(),
            "micro_complete_given_macro": group.loc[group.macro_complete == 1, "complete_evidence_hit"].mean() if macro_hits else None,
            "macro_tpr": group.macro_tp.sum()/group.macro_positive.sum(),
            "macro_fpr": group.macro_fp.sum()/group.macro_negative.sum() if group.macro_negative.sum() else None,
            "input_proxy_tokens": group.input_proxy_tokens.mean()})
    return pd.DataFrame(rows)


def choose_policy(summary, provider, target):
    table = summary[summary.provider == provider].copy()
    feasible = table[table.complete_evidence_hit >= target]
    if not feasible.empty:
        chosen = feasible.sort_values(["input_proxy_tokens", "group_size", "selection_k"]).iloc[0]
    else:
        chosen = table.sort_values(["complete_evidence_hit", "input_proxy_tokens", "group_size", "selection_k"],
                                   ascending=[False, True, True, True]).iloc[0]
    return {"group_size": int(chosen.group_size), "selection_k": int(chosen.selection_k),
            "development_target": float(target), "development_evidence_rate": float(chosen.complete_evidence_hit),
            "target_met": not feasible.empty, "selection_input_proxy": float(chosen.input_proxy_tokens)}


class ParallelComparison:
    def __init__(self, baseline, adapters):
        self.baseline, self.adapters, self.store = baseline, adapters, baseline.store

    def retrieve(self, query, method):
        return self.baseline.retrieve(query, method) if method in ("A", "B") else self.adapters[method].retrieve(query, method)


def run_parallel_experiment(root, settings=None, progress=print):
    """Live calls are opt-in at the notebook boundary; every run gets new receipts."""
    root = Path(root).resolve()
    settings = read_json(root / "config/parallel_routing.json") if settings is None else settings
    validate_settings(settings)
    if not torch.cuda.is_available():
        raise RuntimeError("This local comparison requires CUDA.")
    if torch.cuda.mem_get_info()[0] < 4*1024**3:
        raise RuntimeError("Close inactive project notebook kernels: comparison needs 4 GiB free GPU memory.")
    from dotenv import load_dotenv
    load_dotenv(root / ".env", override=False)
    generation, c_config, d_config = (read_json(root / f"config/{name}.json") for name in ("generation", "c", "d_laya"))
    dev_questions = selected_questions(root, "dev", settings["questions_per_task"]["dev"])
    eval_questions, excluded_ids = unseen_evaluation_questions(root, settings["questions_per_task"]["eval"])
    directory = root / "results" / f"parallel_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    audit = RawLog(root, manifest={"notebook": "07_mathematical_framework.ipynb", "settings": settings,
        "phase": "parallel_routing", "eval_question_ids": [q["id"] for q in eval_questions]},
        prices={"jev": c_config["prices"], "openai": generation["prices"]})
    audit_lock = Lock()
    attempts = {"jev": 0, "openai": 0}
    def request_hook(request):
        provider = "jev" if json.loads(request.content).get("model", "").startswith("jev-") else "openai"
        key = "max_jev_http_calls" if provider == "jev" else "max_llm_calls"
        with audit_lock:
            if attempts[provider] >= settings[key]:
                raise RuntimeError("Declared physical API request budget exhausted.")
            attempts[provider] += 1
            audit.http_request(request)
    def response_hook(response):
        with audit_lock:
            audit.http_response(response)
    hooks = {"request": [request_hook], "response": [response_hook]}
    for path in [root/"config/parallel_routing.json", root/"config/c.json", root/"config/d_laya.json",
                 root/"config/generation.json", root/"notebook/07_mathematical_framework.ipynb",
                 *sorted((root/"rag_baselines").glob("*.py"))]:
        audit.snapshot(path)
    try:
        with ExitStack() as stack:
            llm_http = stack.enter_context(DefaultHttpxClient(timeout=180, event_hooks=hooks))
            generator = stack.enter_context(OpenAI(max_retries=0, timeout=180, http_client=llm_http))
            jev_http = stack.enter_context(httpx.Client(timeout=c_config["timeout_seconds"], event_hooks=hooks))
            jev = JevClient(c_config, http_client=jev_http)
            baseline = Baselines(root, device="cuda", evidence_k=settings["evidence_k"], raw_log=audit)
            entry = read_json(root/"data/prepared/manifest.json")["corpora"][str(settings["corpus_size"])]
            baseline.build_index(root/entry["path"])
            if baseline.store.sha256 != entry["sha256"]:
                raise ValueError("Prepared corpus integrity check failed.")
            tree, inherited = load_tree(root, baseline.store, read_json(root/"config/c_taxonomy.json"), c_config)
            base_manifest = runtime_manifest(root, baseline)
            write_json(directory/"manifest.json", {**base_manifest, "mode": "parallel_routing_study", "settings": settings,
                "generation": generation, "inherited_preparation": inherited, "raw_log": str(audit.directory.relative_to(root)),
                "dev_question_ids": [q["id"] for q in dev_questions], "planned_eval_question_ids": [q["id"] for q in eval_questions],
                "excluded_historical_eval_ids": excluded_ids, "evaluation_labels_loaded": False,
                "scope": "Development quality uses replayed judgment subsets; latency uses fresh held-out retrieval."})
            all_groups, preparation = {}, {}
            for size in settings["group_sizes"]:
                with audit.scope(phase="group_preparation", method="shared", group_size=size):
                    group_settings = {**settings, "descriptor_max_output_tokens": settings.get(
                        "descriptor_max_output_tokens_by_group_size", {}).get(str(size), settings["descriptor_max_output_tokens"])}
                    started = time.perf_counter()
                    calls_before = attempts["openai"]
                    groups, receipt, path = prepare_groups(root, baseline.store, tree, size, group_settings, generation, generator, audit, progress)
                    all_groups[size], preparation[size] = groups, {**receipt, "directory": str(path.relative_to(root)),
                        "prepare_wall_seconds_this_session": time.perf_counter()-started,
                        "actual_new_calls_this_session": attempts["openai"]-calls_before}
            write_json(directory/"preparation.json", preparation)
            laya = LayaDecisionClient(d_config, device=baseline.device)
            clients = {"jev": jev, "laya": laya}
            write_json(directory/"laya_runtime.json", laya.runtime_info())
            labels = {row["id"]: row for row in read_jsonl(root/"data/prepared/labels_dev.jsonl")}
            write_json(directory/"development_labels.json", {q["id"]: labels[q["id"]] for q in dev_questions})
            baseline_quality = evaluate_cached_quality(baseline, dev_questions, list(labels.values()), audit=audit)
            b_target = sum(r["metrics"]["complete_evidence_hit"] for r in baseline_quality if r["method"] == "B")/len(dev_questions)
            write_json(directory/"baseline_development_quality.json", baseline_quality)
            quality_rows = []
            policy_base = {key: settings[key] for key in ("evidence_k", "max_calls_per_retrieval", "http_workers")}
            for provider, client in clients.items():
                for size, groups in all_groups.items():
                    policy = {**policy_base, "selection_k": max(settings["selection_budgets"])}
                    adapter = ParallelRouting(baseline.store, groups, client, provider, policy, directory/"decision_calls.jsonl", audit)
                    for index, question in enumerate(dev_questions):
                        with query_scope(audit, "development_quality", question, adapter.method):
                            judged = adapter.judge(question["question"])
                            append_record(directory/"development_judgments.jsonl", {"provider": provider, "group_size": size,
                                "question_id": question["id"], "task": question["task"], "judgments": judged})
                            for k in settings["selection_budgets"]:
                                result = adapter.assemble(judged, k, quality_only=True)
                                result.update({"question_id": question["id"], "task": question["task"]})
                                metrics = routing_quality(result, groups, labels[question["id"]], baseline.store)
                                row = {"provider": provider, "group_size": size, "selection_k": k,
                                       "question_id": question["id"], "task": question["task"], **metrics}
                                quality_rows.append(row)
                                append_record(directory/"development_quality.jsonl", {"summary": row, "retrieval": result})
                        if progress:
                            progress(f"Development {provider}, g={size}: {index + 1}/{len(dev_questions)}")
            quality = pd.DataFrame(quality_rows)
            quality.to_csv(directory/"development_questions.csv", index=False)
            summary = summarize_quality(quality)
            summary.to_csv(directory/"development_summary.csv", index=False)
            selected = {provider: choose_policy(summary, provider, b_target) for provider in clients}
            frozen = {"selected": selected, "settings": settings, "generation": generation,
                      "development_summary_sha256": file_sha256(directory/"development_summary.csv"),
                      "groups_sha256": {str(size): digest(groups) for size, groups in all_groups.items()},
                      "c_model": c_config, "d_model": d_config, "evaluation_labels_loaded": False,
                      "code_sha256": {p.name: file_sha256(p) for p in (root/"rag_baselines").glob("parallel_*.py")}}
            write_json(directory/"frozen_policy.json", frozen)
            write_json(directory/"policy_before_eval_labels.json", {"split": "eval", "question_ids": [q["id"] for q in eval_questions],
                "frozen_policy_sha256": file_sha256(directory/"frozen_policy.json"), "evaluation_labels_loaded": False})
            if progress:
                progress(f"Policies frozen: {selected}; running fresh held-out retrieval and generation")
            eval_labels = read_jsonl(root/"data/prepared/labels_eval.jsonl")
            eval_by_id = {row["id"]: row for row in eval_labels}
            write_json(directory/"evaluation_labels.json", {q["id"]: eval_by_id[q["id"]] for q in eval_questions})
            adapters = {}
            for provider, chosen in selected.items():
                adapter = ParallelRouting(baseline.store, all_groups[chosen["group_size"]], clients[provider], provider,
                    {**policy_base, "selection_k": chosen["selection_k"]}, directory/"decision_calls.jsonl", audit)
                adapters[adapter.method] = adapter
            engine = ParallelComparison(baseline, adapters)
            records = measure_system(engine, eval_questions, eval_labels, client=generator, generation_config=generation,
                max_calls=settings["max_llm_calls"]-attempts["openai"], methods=("A", "B", "C-parallel", "D-parallel"),
                seed=settings["seed"], audit=audit, progress=progress,
                on_result=lambda row: append_record(directory/"system.jsonl", row))
            answer_audit = read_json(root/"config/answer_audit.json")
            write_json(directory/"answer_audit.json", answer_audit)
            questions_frame = pd.DataFrame(summary_rows(records, "parallel_heldout", answer_audit))
            questions_frame.to_csv(directory/"questions.csv", index=False)
            aggregate(questions_frame).to_csv(directory/"summary.csv", index=False)
            write_json(directory/"completed.json", {"records": len(records), "physical_api_attempts": attempts,
                "raw_log": str(audit.directory.relative_to(root)),
                "file_sha256": {p.name: file_sha256(p) for p in directory.iterdir() if p.is_file()},
                "note": "Single hardware run, exploratory held-out results; no constant-time or accuracy guarantee."})
            (root/"results/current_parallel_run.txt").write_text(str(directory.relative_to(root)), encoding="utf-8")
            return directory
    except Exception as error:
        write_json(directory/"failed.json", {"error_type": type(error).__name__, "physical_api_attempts": attempts,
                   "raw_log": str(audit.directory.relative_to(root)), "partial_results_preserved": True})
        raise


def load_parallel_report(directory):
    directory = Path(directory)
    receipt = read_json(directory/"completed.json")
    for name, digest_value in receipt["file_sha256"].items():
        if file_sha256(directory/name) != digest_value:
            raise ValueError(f"Parallel result integrity check failed: {name}")
    records = read_jsonl(directory/"system.jsonl")
    if len(records) != receipt["records"]:
        raise ValueError("Parallel result count mismatch.")
    keys = [(r["retrieval"]["question_id"], r["retrieval"]["method"], r["retrieval"]["repeat"]) for r in records]
    expected_ids = set(read_json(directory/"policy_before_eval_labels.json")["question_ids"])
    expected_keys = {(qid, method, 0) for qid in expected_ids for method in ("A", "B", "C-parallel", "D-parallel")}
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise ValueError("Missing, repeated or unexpected held-out observations.")
    labels = read_json(directory/"evaluation_labels.json")
    for record in records:
        retrieval, generated = record["retrieval"], record["generation"]
        label = labels[retrieval["question_id"]]
        if retrieval["metrics"] != retrieval_metrics(retrieval, label):
            raise ValueError("Stored retrieval metrics disagree with original evidence IDs.")
        if record["answer_metrics"] != answer_metrics(generated["answer"], label["answers"]):
            raise ValueError("Stored answer metrics disagree with generated answers.")
        actual_cost = usage_cost(generated["api_usage"], generated["prices"])
        if actual_cost != record["generation_api_cost_usd"]:
            raise ValueError("Generation cost disagrees with recorded provider usage.")
        if retrieval["execution_mode"] != "online_retrieval" or retrieval["seconds"]["retrieval_total"] is None:
            raise ValueError("Held-out latency must come from fresh retrieval.")
    frame = pd.DataFrame(summary_rows(records, "parallel_heldout", read_json(directory/"answer_audit.json")))
    restored = aggregate(frame)
    saved = pd.read_csv(directory/"summary.csv")
    pd.testing.assert_frame_equal(restored, saved, check_dtype=False, atol=1e-10)
    quality = pd.read_csv(directory/"development_questions.csv")
    dev_summary = summarize_quality(quality)
    pd.testing.assert_frame_equal(dev_summary, pd.read_csv(directory/"development_summary.csv"), check_dtype=False, atol=1e-10)
    frozen = read_json(directory/"frozen_policy.json")
    if file_sha256(directory/"frozen_policy.json") != read_json(directory/"policy_before_eval_labels.json")["frozen_policy_sha256"]:
        raise ValueError("Evaluation policy was modified after its freeze receipt.")
    return dev_summary, restored, frozen
