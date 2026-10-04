import copy
import importlib
import json
from pathlib import Path
import sys

import pytest

from rag_baselines.comparison import load_comparison, resolve_comparison_run
from rag_baselines.data import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from rag_baselines.generation import build_prompt, usage_cost
from rag_baselines.jev import call_usage


@pytest.fixture
def run(tmp_path):
    corpus = tmp_path / "data/prepared/corpus_1.jsonl"
    passage = {"id": "7:0", "wikipedia_id": "7", "title": "Paris", "content": "Paris is the capital of France.", "source_spans": []}
    write_jsonl(corpus, [passage])
    questions = tmp_path / "data/prepared/questions_dev.jsonl"
    labels = tmp_path / "data/prepared/labels_dev.jsonl"
    write_jsonl(questions, [{"id": "q", "question": "Capital of France?", "task": "nq"}])
    write_jsonl(labels, [{"id": "q", "answers": ["Paris"], "evidence_groups": [["7:0"]]}])
    write_json(tmp_path / "data/prepared/manifest.json", {
        "corpora": {"1": {"path": "data/prepared/corpus_1.jsonl", "sha256": file_sha256(corpus)}},
        "labels_dev_sha256": file_sha256(labels)})
    prices = {"input_usd_per_million": .1, "cached_input_usd_per_million": .01, "output_usd_per_million": .5}
    c_prices = {"input_usd_per_million": .042, "output_usd_per_million": 0}
    directory = tmp_path / "results/fixture"
    write_json(directory / "manifest.json", {"mode": "live_system_abc", "split": "dev",
        "questions_file_sha256": file_sha256(questions),
        "generation": {"model": "gpt-6-luna", "encoding_name": "o200k_base", "prices": prices},
        "c": {"config": {"prices": c_prices}}})
    records = []
    metrics = {"complete_evidence_hit": 1, "complete_candidate_hit": 1, "best_group_coverage": 1.0}
    for method in ("A", "B", "C"):
        retrieved = {"question_id": "q", "question": "Capital of France?", "task": "nq", "method": method,
                     "repeat": 0, "execution_mode": "online_retrieval", "corpus_count": 1,
                     "corpus_sha256": file_sha256(corpus), "evidence": [passage],
                     "candidates": [{"id": "7:0", "dense_score": 1.0}], "metrics": metrics,
                     "seconds": {"retrieval_total": .01}}
        retrieval_cost = 0.0
        if method == "C":
            calls = [{"response": {"usage": {"input_tokens": 10, "output_tokens": 1}}}]
            usage, retrieval_cost = call_usage(calls, c_prices)
            retrieved.update({"trace": {"calls": calls}, "api_usage": usage, "api_cost_usd": retrieval_cost})
        usage = {"input_tokens": 100, "output_tokens": 10}
        cost = usage_cost(usage, prices)
        generated = {"requested_model": "gpt-6-luna", "valid_output": True, "answer": "Paris", "citations": ["7:0"],
                     "output_text": '{"answer":"Paris","citations":["7:0"]}', "prices": prices, "api_usage": usage,
                     "api_cost_usd": cost, "prompt": build_prompt(retrieved), "seconds": {"prompt_prepare": .001, "api_request": .02}}
        records.append({"retrieval": retrieved, "generation": generated, "online_total_seconds": .032,
                        "answer_metrics": {"answer_em": 1.0, "answer_f1": 1.0},
                        "retrieval_api_cost_usd": retrieval_cost, "generation_api_cost_usd": cost,
                        "online_api_cost_usd": cost + retrieval_cost})
    write_jsonl(directory / "system.jsonl", records)
    return tmp_path, directory


def test_comparison_reconciles_originals_metrics_tokens_cost_and_provenance(run):
    root, directory = run
    _, rows, hashes = load_comparison(root, directory)
    assert len(rows) == 3 and len(hashes) == 5
    assert all(r["answer_em"] == 1 and r["complete_evidence_hit"] == 1 for r in rows)
    assert rows[2]["jev_input_tokens"] == 10 and rows[0]["jev_input_tokens"] is None
    assert rows[2]["online_api_cost_usd"] > rows[0]["online_api_cost_usd"]


@pytest.mark.parametrize("corruption", ["duplicate", "prompt", "cost", "missing_method", "candidate"])
def test_comparison_rejects_misleading_or_incomplete_plot_inputs(run, corruption):
    root, directory = run
    records = read_jsonl(directory / "system.jsonl")
    if corruption == "duplicate": records.append(copy.deepcopy(records[0]))
    if corruption == "prompt": records[0]["generation"]["prompt"]["text"] += "forged"
    if corruption == "cost": records[0]["online_api_cost_usd"] = -1
    if corruption == "missing_method": records.pop()
    if corruption == "candidate": records[1]["retrieval"]["candidates"][0]["dense_score"] = 9
    write_jsonl(directory / "system.jsonl", records)
    with pytest.raises(ValueError): load_comparison(root, directory)


def test_comparison_preserves_unknown_usage_and_cost_instead_of_zero(run):
    root, directory = run
    records = read_jsonl(directory / "system.jsonl")
    records[0]["generation"].update({"api_usage": None, "api_cost_usd": None})
    records[0].update({"generation_api_cost_usd": None, "online_api_cost_usd": None})
    write_jsonl(directory / "system.jsonl", records)
    _, rows, _ = load_comparison(root, directory)
    assert rows[0]["llm_input_tokens"] is None and rows[0]["online_api_cost_usd"] is None


def test_staged_run_selection_uses_complete_evaluation_and_ignores_partial_runs(run):
    root, original = run
    study = root / "results/staged_20261004T000000_completed"
    evaluation = study / "eval/ABCD_frozen"
    for name in ("manifest.json", "system.jsonl"):
        evaluation.mkdir(parents=True, exist_ok=True)
        (evaluation / name).write_bytes((original / name).read_bytes())
    (evaluation / "summary.csv").write_text("method\nA\nB\nC\n", encoding="utf-8")
    write_json(evaluation / "completed.json", {
        "records": 3, "system_sha256": file_sha256(evaluation / "system.jsonl")})
    partial = root / "results/staged_20261004T000001_partial/eval/ABCD_frozen"
    partial.mkdir(parents=True)
    for name in ("manifest.json", "system.jsonl", "summary.csv"):
        (partial / name).write_bytes((evaluation / name).read_bytes())
    assert resolve_comparison_run(root) == (evaluation, study)
    assert resolve_comparison_run(root, study.name) == (evaluation, study)
    assert resolve_comparison_run(root, evaluation) == (evaluation, study)
    with pytest.raises(ValueError, match="completed"):
        resolve_comparison_run(root, partial)


def test_comparison_rejects_a_modified_completed_staged_result(run):
    root, directory = run
    write_json(directory / "completed.json", {
        "records": 3, "system_sha256": file_sha256(directory / "system.jsonl")})
    records = read_jsonl(directory / "system.jsonl")
    records[0]["generation"]["returned_model"] = "edited-after-completion"
    write_jsonl(directory / "system.jsonl", records)
    with pytest.raises(ValueError, match="completion receipt"):
        load_comparison(root, directory)


@pytest.mark.parametrize("missing", ["question", "repeat"])
def test_comparison_rejects_missing_declared_observations_even_when_remaining_rows_are_balanced(run, missing):
    root, directory = run
    manifest = read_json(directory / "manifest.json")
    manifest.update(question_ids=["q"], repeats=1)
    if missing == "question":
        manifest["question_ids"].append("q-declared-but-absent")
    else:
        manifest["repeats"] = 2
    write_json(directory / "manifest.json", manifest)
    with pytest.raises(ValueError, match="recorded question IDs|recorded repetitions"):
        load_comparison(root, directory)


def test_comparison_notebook_refreshes_legacy_loader_and_selects_completed_abcd(run, monkeypatch):
    """Replay an ABC-only loader cached by an already-running notebook kernel."""
    root, original = run
    manifest = read_json(original / "manifest.json")
    manifest.update(mode="live_system_abcd", methods=["A", "B", "C", "D"])
    records = read_jsonl(original / "system.jsonl")
    local = copy.deepcopy(records[2])
    local["retrieval"].update(method="D", api_usage=None, api_cost_usd=0.0,
        local_usage={"usage_complete": True, "calls": 1, "truncated_calls": 0,
                     "state_tokens_dropped": 0, "inference_seconds": .005},
        model_tokens={"laya": {"nonpad": 10, "output": 0}})
    local["retrieval"]["trace"]["calls"][0]["response"]["usage"]["output_tokens"] = 0
    local.update(retrieval_api_cost_usd=0.0,
                 online_api_cost_usd=local["generation_api_cost_usd"])
    records.append(local)
    directory = root / "results/20261004T000000_completed"
    write_json(directory / "manifest.json", manifest)
    write_jsonl(directory / "system.jsonl", records)
    (directory / "system_summary.csv").write_text("method\nA\nB\nC\nD\n", encoding="utf-8")
    # Even a newer quality directory with system-named files must not be selected.
    quality = root / "results/20261004T000001_quality"
    write_json(quality / "manifest.json", dict(manifest, mode="cached_quality_abcd"))
    write_jsonl(quality / "system.jsonl", records)
    (quality / "system_summary.csv").write_text("method\n", encoding="utf-8")

    comparison = importlib.import_module("rag_baselines.comparison")
    current_loader = comparison.load_comparison

    def legacy_loader(root, directory):
        if read_json(directory / "manifest.json")["mode"] != "live_system_abc":
            raise ValueError("Choose a completed live_system_abc run, not a quality cache or partial run.")
        return current_loader(root, directory)

    monkeypatch.setattr(comparison, "load_comparison", legacy_loader)
    monkeypatch.setattr(sys, "path", list(sys.path))
    (root / "rag_baselines").mkdir()
    monkeypatch.chdir(root)
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                           "notebook/06_compare_results.ipynb").read_text(encoding="utf-8"))
    setup = "".join(next(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"))
    namespace = {}
    exec(compile(setup, "06_compare_results.ipynb:setup", "exec"), namespace)
    loaded, rows, _ = namespace["load_comparison"](namespace["ROOT"], namespace["RUN_DIR"])
    assert namespace["RUN_DIR"] == directory
    assert loaded["methods"] == ["A", "B", "C", "D"]
    assert {row["method"] for row in rows} == {"A", "B", "C", "D"}
    assert len(rows) == 4
    with pytest.raises(ValueError, match="Choose a completed live-system run"):
        namespace["load_comparison"](root, quality)
