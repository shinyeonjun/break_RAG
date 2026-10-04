"""Reconcile a completed A/B/C/D run before plotting; never makes API calls."""
import json
import math
from pathlib import Path

from .data import file_sha256, read_json, read_jsonl
from .evaluation import retrieval_metrics
from .generation import answer_metrics, build_prompt, usage_cost
from .jev import call_usage
from .store import DocumentStore


def _same_number(actual, expected, name):
    if expected is None:
        valid = actual is None
    else:
        valid = type(actual) in (int, float) and math.isfinite(actual) and math.isclose(actual, expected, abs_tol=1e-10)
    if not valid:
        raise ValueError(f"Stored {name} disagrees with its source data.")


def resolve_comparison_run(root, system_run=None):
    """Select a completed system run, including a staged study's evaluation."""
    root = Path(root)

    def completed(directory):
        if not all((directory / name).is_file() for name in ("manifest.json", "system.jsonl")):
            return False
        if read_json(directory / "manifest.json").get("mode") not in ("live_system_abc", "live_system_abcd"):
            return False
        if directory.parent.name in ("dev", "eval"):
            return all((directory / name).is_file() for name in ("completed.json", "summary.csv"))
        return (directory / "system_summary.csv").is_file()

    if system_run is not None:
        directory = Path(system_run)
        if not directory.is_absolute():
            directory = root / "results" / directory
        if not (directory / "manifest.json").is_file():
            directory = directory / "eval" / "ABCD_frozen"
        if not completed(directory):
            raise ValueError("Choose a completed system run or a study with completed evaluation.")
    else:
        sources = list((root / "results").glob("20*/manifest.json"))
        sources += list((root / "results").glob("staged_*/eval/ABCD_frozen/manifest.json"))
        candidates = [source.parent for source in sources if completed(source.parent)]
        if not candidates:
            raise RuntimeError("No completed comparison run. Execute notebook 05 first.")
        directory = max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime_ns)
    study = directory.parent.parent if directory.parent.name in ("dev", "eval") else None
    return directory, study


def load_comparison(root, directory):
    root, directory = Path(root), Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["mode"] not in ("live_system_abc", "live_system_abcd"):
        raise ValueError("Choose a completed live-system run, not a quality cache or partial run.")
    expected_methods = manifest.get("methods", ["A", "B", "C"])
    if not expected_methods or len(expected_methods) != len(set(expected_methods)):
        raise ValueError("Run manifest has an invalid methods list.")
    records = read_jsonl(directory / "system.jsonl")
    if not records:
        raise ValueError("No system observations.")
    receipt_path = directory / "completed.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["records"] != len(records) or receipt["system_sha256"] != file_sha256(directory / "system.jsonl"):
            raise ValueError("System results differ from their completion receipt.")
    prepared = read_json(root / "data/prepared/manifest.json")
    split = manifest["split"]
    question_path = root / f"data/prepared/questions_{split}.jsonl"
    label_path = root / f"data/prepared/labels_{split}.jsonl"
    if (file_sha256(question_path) != manifest["questions_file_sha256"]
            or file_sha256(label_path) != prepared[f"labels_{split}_sha256"]):
        raise ValueError("Question/label files differ from the recorded source.")
    questions = {q["id"]: q for q in read_jsonl(question_path)}
    labels = {q["id"]: q for q in read_jsonl(label_path)}
    size = records[0]["retrieval"]["corpus_count"]
    entry = prepared["corpora"][str(size)]
    store = DocumentStore(root / entry["path"])
    if store.sha256 != entry["sha256"]:
        raise ValueError("Prepared corpus hash mismatch.")
    keys, groups, rows = set(), {}, []
    for record in records:
        retrieved, generated = record["retrieval"], record["generation"]
        qid, method, repeat = retrieved["question_id"], retrieved["method"], retrieved["repeat"]
        key = qid, method, repeat
        if key in keys:
            raise ValueError("Duplicate question/method/repeat observation.")
        keys.add(key)
        groups.setdefault((qid, repeat), {})[method] = retrieved
        if (retrieved["execution_mode"] != "online_retrieval" or retrieved["corpus_count"] != size
                or retrieved["question"] != questions[qid]["question"] or retrieved["task"] != questions[qid]["task"]):
            raise ValueError("Observations must use the same live corpus and question source.")
        store.validate_evidence(retrieved)
        if retrieved["metrics"] != retrieval_metrics(retrieved, labels[qid]):
            raise ValueError("Stored retrieval metrics do not match offline labels.")
        if record["answer_metrics"] != answer_metrics(generated["answer"], labels[qid]["answers"]):
            raise ValueError("Stored answer metrics do not match offline references.")
        if generated["requested_model"] != manifest["generation"]["model"]:
            raise ValueError("Methods must use the recorded common generator.")
        if generated["valid_output"]:
            parsed = json.loads(generated["output_text"])
            if parsed["answer"] != generated["answer"] or parsed["citations"] != generated["citations"]:
                raise ValueError("Parsed answer differs from the raw output text.")
            if any(pid not in {p["id"] for p in retrieved["evidence"]} for pid in parsed["citations"]):
                raise ValueError("Citation is outside the supplied originals.")
        prompt = build_prompt(retrieved, encoding_name=manifest["generation"]["encoding_name"])
        if generated["prompt"] != prompt:
            raise ValueError("Recorded prompt differs from full original evidence/common budget.")
        generation_cost = usage_cost(generated["api_usage"], generated["prices"])
        if generated["prices"] != manifest["generation"]["prices"]:
            raise ValueError("Generation prices must match the run snapshot.")
        retrieval_cost = 0.0
        local = {}
        laya_tokens = None
        if method == "C":
            usage, retrieval_cost = call_usage(retrieved["trace"]["calls"], manifest["c"]["config"]["prices"])
            if usage != retrieved["api_usage"]:
                raise ValueError("Jev usage disagrees with request/response records.")
            _same_number(retrieved["api_cost_usd"], retrieval_cost, "Jev API cost")
        elif method == "D":
            usage, _ = call_usage(retrieved["trace"]["calls"], None)
            local = retrieved.get("local_usage") or {}
            laya_tokens = retrieved.get("model_tokens", {}).get("laya", {}).get("nonpad")
            if (retrieved.get("api_usage") is not None or retrieved.get("api_cost_usd") != 0.0
                    or not local.get("usage_complete") or usage["input_tokens"] != laya_tokens
                    or usage["output_tokens"] != retrieved["model_tokens"]["laya"]["output"]
                    or local.get("calls") != usage["requests"]
                    or local.get("truncated_calls") != 0 or local.get("state_tokens_dropped") != 0):
                raise ValueError("D local usage or truncation records disagree with decision-call responses.")
            retrieval_cost = 0.0  # API fee only; local GPU cost is not priced here.
        elif method not in ("A", "B"):
            raise ValueError("Unknown comparison method.")
        total_cost = retrieval_cost + generation_cost if retrieval_cost is not None and generation_cost is not None else None
        for name, expected in (("retrieval_api_cost_usd", retrieval_cost), ("generation_api_cost_usd", generation_cost),
                               ("online_api_cost_usd", total_cost)):
            _same_number(record[name], expected, name)
        _same_number(generated["api_cost_usd"], generation_cost, "generation API cost")
        retrieval_seconds = retrieved["seconds"]["retrieval_total"]
        generation_seconds = sum(generated["seconds"][k] for k in ("prompt_prepare", "api_request"))
        total_seconds = record["online_total_seconds"]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
               for v in (retrieval_seconds, generation_seconds, total_seconds)) or total_seconds + 1e-6 < retrieval_seconds + generation_seconds:
            raise ValueError("Invalid live timing or incompatible stage sums.")
        usage = generated["api_usage"] if isinstance(generated["api_usage"], dict) else {}
        decision_usage = retrieved.get("api_usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        rows.append({"question_id": qid, "question": retrieved["question"], "task": retrieved["task"], "method": method,
                     "repeat": repeat, "evidence_count": len(retrieved["evidence"]), "retrieval_seconds": retrieval_seconds,
                     "generation_seconds": generation_seconds, "online_seconds": total_seconds,
                     "diagnostic_capture_seconds": record.get("diagnostic_capture_seconds"),
                     "raw_events_path": (record.get("raw_log") or {}).get("events_path"),
                     "llm_input_tokens": input_tokens if type(input_tokens) is int and input_tokens >= 0 else None,
                     "llm_output_tokens": output_tokens if type(output_tokens) is int and output_tokens >= 0 else None,
                     "jev_input_tokens": decision_usage.get("input_tokens") if decision_usage.get("usage_complete") else None,
                     "jev_requests": decision_usage.get("requests"),
                     "laya_input_tokens": laya_tokens if method == "D" else None,
                     "laya_decision_calls": local.get("calls") if method == "D" else None,
                     "laya_truncated_calls": local.get("truncated_calls") if method == "D" else None,
                     "laya_state_tokens_dropped": local.get("state_tokens_dropped") if method == "D" else None,
                     "laya_inference_seconds": local.get("inference_seconds") if method == "D" else None,
                     "decision_model_calls": len(retrieved.get("trace", {}).get("calls", [])) if method in ("C", "D") else None,
                     "decision_stop_reason": retrieved.get("trace", {}).get("stop_reason"),
                     "retrieval_api_cost_usd": retrieval_cost, "generation_api_cost_usd": generation_cost,
                     "online_api_cost_usd": total_cost, "answer": generated["answer"], **record["answer_metrics"],
                     **retrieved["metrics"],
                     "c_stop_reason": retrieved.get("trace", {}).get("stop_reason") if method == "C" else None})
    repeats = {}
    for (qid, repeat), methods in groups.items():
        if set(methods) != set(expected_methods):
            raise ValueError("Each question/repeat must contain exactly the methods in the run manifest.")
        repeats.setdefault(qid, set()).add(repeat)
        a, b = methods["A"], methods["B"]
        if [(p["id"], p["dense_score"]) for p in a["candidates"]] != [(p["id"], p["dense_score"]) for p in b["candidates"]]:
            raise ValueError("A/B dense candidate IDs/scores differ.")
    if any(value != next(iter(repeats.values())) for value in repeats.values()):
        raise ValueError("Questions have unbalanced repetition counts.")
    if "question_ids" in manifest and set(repeats) != set(manifest["question_ids"]):
        raise ValueError("System observations do not cover the recorded question IDs.")
    if "repeats" in manifest and any(value != set(range(manifest["repeats"])) for value in repeats.values()):
        raise ValueError("System observations do not cover the recorded repetitions.")
    provenance = {str(p.relative_to(root)): file_sha256(p) for p in
                  (directory / "manifest.json", directory / "system.jsonl", question_path, label_path, store.path)}
    return manifest, rows, provenance
