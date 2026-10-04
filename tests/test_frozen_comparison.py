from pathlib import Path
import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import httpx

import rag_baselines.frozen_comparison as frozen
from rag_baselines.c_tree import digest
from rag_baselines.data import read_json, read_jsonl, write_json, write_jsonl, file_sha256
from rag_baselines.parallel_experiment import unseen_evaluation_questions
from rag_baselines.store import DocumentStore


ROOT = Path(__file__).resolve().parents[1]


def test_generation_budget_and_integer_counts_checked_before_live_calls():
    settings = read_json(ROOT/"config/frozen_comparison.json")
    frozen.validate_frozen_settings(settings)
    with pytest.raises(ValueError, match="LLM call budget"):
        frozen.validate_frozen_settings({**settings, "max_llm_calls": 1})
    with pytest.raises(ValueError, match="positive integer"):
        frozen.validate_frozen_settings({**settings, "questions_per_task": True})


def test_later_holdout_excludes_completed_and_partially_started_frozen_evaluations(tmp_path):
    write_jsonl(tmp_path/"data/prepared/questions_eval.jsonl", [
        {"id": f"{task}{i}", "task": task, "question": "query"}
        for task in ("nq", "hotpotqa") for i in range(4)])
    write_json(tmp_path/"results/frozen_first/manifest.json", {"split": "eval", "question_ids": ["nq0", "hotpotqa0"]})
    write_json(tmp_path/"results/frozen_partial/manifest.json", {"split": "eval", "question_ids": ["nq1", "hotpotqa1"]})
    questions, excluded = unseen_evaluation_questions(tmp_path, 1)
    assert {q["id"] for q in questions} == {"nq2", "hotpotqa2"}
    assert set(excluded) == {"nq0", "nq1", "hotpotqa0", "hotpotqa1"}


@pytest.fixture
def source(tmp_path, monkeypatch):
    directory = tmp_path/"results/length_source"
    cache = tmp_path/"data/parallel_groups/chosen"
    write_jsonl(tmp_path/"corpus.jsonl", [{"id": "a", "title": "Facts", "content": "Original fact.",
        "source_spans": [], "wikipedia_id": "a"}])
    store = DocumentStore(tmp_path/"corpus.jsonl")
    groups = [{"id": "group", "passage_ids": ["a"], "descriptor": {"summary": "Original fact.", "entities": [], "qa": []}}]
    write_json(cache/"groups.json", groups)
    prepared = {"directory": str(cache.relative_to(tmp_path)), "groups_sha256": file_sha256(cache/"groups.json"),
        "descriptor_mean_tokens_proxy": 10}
    policy = {"selection_k": 1, "evidence_k": 5, "max_calls_per_retrieval": 32, "http_workers": 4}
    routing_source = tmp_path/"rag_baselines/parallel_routing.py"
    routing_source.parent.mkdir()
    routing_source.write_text("frozen routing code", encoding="utf-8")
    fixed = {"policy": policy, "groups_sha256": {"720": digest(groups)},
        "code_sha256": {"parallel_routing.py": file_sha256(routing_source)}}
    manifest = {"settings": {"corpus_size": 1}, "corpus_sha256": store.sha256, "group_size": 1,
        "generation": {"model": "pinned"}, "c_model": {"model": "jev-pinned"}, "reference_run": "results/parallel_source"}
    for name in ("completed.json", "summary.csv", "preparation.json"):
        write_json(directory/name, {})
    write_json(directory/"fixed_conditions.json", fixed)
    write_json(tmp_path/"config/generation.json", manifest["generation"])
    write_json(tmp_path/"config/c.json", manifest["c_model"])
    write_json(tmp_path/"config/d_laya.json", {"model": {"revision": "pinned"}})
    write_json(tmp_path/"config/baselines.json", {"evidence_k": 5, "revision": "pinned"})
    write_json(tmp_path/"results/parallel_source/manifest.json", {"config": read_json(tmp_path/"config/baselines.json")})
    write_json(tmp_path/"results/parallel_source/frozen_policy.json", {"d_model": read_json(tmp_path/"config/d_laya.json")})
    write_json(tmp_path/"data/prepared/manifest.json", {"corpora": {"1": {"path": "corpus.jsonl"}}})
    summary = pd.DataFrame([{"target_met": False, "descriptor_target_tokens": 180, "decision_api_cost_usd": .001,
        "questions": 40, "complete_evidence_hit": .8, "development_target": .875},
        {"target_met": True, "descriptor_target_tokens": 720, "decision_api_cost_usd": .004,
        "questions": 40, "complete_evidence_hit": .925, "development_target": .875}])
    monkeypatch.setattr(frozen, "load_length_report", lambda *args: (None, summary, manifest, {"720": prepared}))
    settings = {**read_json(ROOT/"config/frozen_comparison.json"), "reference_length_run": "results/length_source"}
    return tmp_path, settings, groups


def test_freeze_reuses_exact_descriptors_and_requires_feasible_development_selection(source):
    root, settings, groups = source
    policy, reused, *_ = frozen.frozen_reference(root, settings)
    assert reused == groups
    assert policy["development_evidence_rate"] == .925
    assert policy["routing_policy"]["evidence_k"] == 5
    assert policy["evaluation_labels_loaded"] is False
    # No evaluation labels or API credentials exist in this fixture.
    with pytest.raises(ValueError, match="feasible tested development"):
        frozen.frozen_reference(root, {**settings, "descriptor_target_tokens": 180})


def test_changed_description_or_routing_code_rejected_before_spending(source):
    root, settings, groups = source
    path = root/"data/parallel_groups/chosen/groups.json"
    changed = copy.deepcopy(groups)
    changed[0]["descriptor"]["summary"] = "Different information"
    write_json(path, changed)
    with pytest.raises(ValueError, match="descriptions differ"):
        frozen.frozen_reference(root, settings)
    write_json(path, groups)
    (root/"rag_baselines/parallel_routing.py").write_text("changed routing", encoding="utf-8")
    with pytest.raises(ValueError, match="Routing code changed"):
        frozen.frozen_reference(root, settings)


def test_baseline_generation_and_final_evidence_budget_cannot_silently_drift(source):
    root, settings, _ = source
    write_json(root/"config/generation.json", {"model": "different"})
    with pytest.raises(ValueError, match="model settings fixed"):
        frozen.frozen_reference(root, settings)
    write_json(root/"config/generation.json", {"model": "pinned"})
    write_json(root/"config/baselines.json", {"evidence_k": 2, "revision": "pinned"})
    with pytest.raises(ValueError, match="final evidence budget fixed"):
        frozen.frozen_reference(root, settings)


@pytest.mark.parametrize("fail_validation", [False, True])
def test_live_runner_reconciles_physical_mock_calls_and_publishes_only_validated_results(source, monkeypatch, fail_validation):
    root, settings, groups = source
    policy, _, store, reference, prepared = frozen.frozen_reference(root, settings)
    settings = {**settings, "questions_per_task": 1}
    policy["settings"] = settings
    policy["generation"] = read_json(ROOT/"config/generation.json")
    policy["c_model"] = read_json(ROOT/"config/c.json")
    policy["d_model"] = read_json(ROOT/"config/d_laya.json")
    reference["inherited_preparation"] = {}
    monkeypatch.setattr(frozen, "frozen_reference", lambda *a: (policy, groups, store, reference, prepared))
    write_json(root/"config/frozen_comparison.json", settings)
    write_json(root/"config/answer_audit.json", read_json(ROOT/"config/answer_audit.json"))
    write_json(root/"notebook/08_compare_frozen_routing.ipynb", {})
    write_jsonl(root/"data/prepared/questions_dev.jsonl", [{"id": "dev", "task": "nq", "question": "warmup"}])
    write_jsonl(root/"data/prepared/questions_eval.jsonl", [
        {"id": task, "task": task, "question": "What is the fact?"} for task in ("nq", "hotpotqa")])
    write_jsonl(root/"data/prepared/labels_eval.jsonl", [
        {"id": task, "answers": ["fact"], "evidence_groups": [["a"]]} for task in ("nq", "hotpotqa")])
    pointer = root/"results/current_frozen_run.txt"
    pointer.write_text("previous-completed-run", encoding="utf-8")
    monkeypatch.setattr(frozen.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(frozen.torch.cuda, "mem_get_info", lambda: (8*1024**3, 8*1024**3))
    monkeypatch.setattr(frozen.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(frozen.torch.cuda, "max_memory_allocated", lambda *a: 1024)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-only-key")
    def responses(body):
        return {qid: ({"type": "noul", "noul": .9} if q["type"] == "noul" else
            {"type": "score", "score": 3, "confidence": 1, "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1}})
            for qid, q in body["questions"].items()}
    def handle(request):
        body = json.loads(request.content)
        if body["model"].startswith("jev-"):
            return httpx.Response(200, json={"model": body["model"], "answers": responses(body),
                "usage": {"input_tokens": 100, "output_tokens": 0}})
        return httpx.Response(200, json={"model": body["model"], "usage": {"input_tokens": 10, "output_tokens": 10},
            "output_text": json.dumps({"answer": "fact", "citations": ["a"]})})
    client_class = httpx.Client
    def http_factory(**kwargs):
        return client_class(transport=httpx.MockTransport(handle), **kwargs)
    monkeypatch.setattr(frozen.httpx, "Client", http_factory)
    monkeypatch.setattr(frozen, "DefaultHttpxClient", http_factory)
    class Generator:
        def __init__(self, **kwargs):
            self.http, self.responses = kwargs["http_client"], self
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def create(self, **kwargs):
            response = self.http.post("https://api.openai.com/v1/responses", json=kwargs).json()
            return SimpleNamespace(output_text=response["output_text"], model_dump=lambda **k: response)
    monkeypatch.setattr(frozen, "OpenAI", Generator)
    class Baseline:
        def __init__(self, *a, **k): self.store, self.device = store, "cpu"
        def build_index(self, *a): pass
        def truncation_audit(self, *a): return {}
        def retrieve(self, query, method):
            return {"method": method, "question": query, "corpus_sha256": store.sha256, "corpus_count": 1,
                "evidence": store.passages, "candidates": [{"id": "a", "dense_score": 1}],
                "execution_mode": "online_retrieval", "seconds": {"retrieval_total": .01},
                "api_usage": None, "api_cost_usd": None, "model_tokens": {}}
    monkeypatch.setattr(frozen, "Baselines", Baseline)
    monkeypatch.setattr(frozen, "runtime_manifest", lambda *a: {})
    class Laya:
        def __init__(self, config, **kwargs):
            self.config, self.model, self.revision = config, "local-model", "pinned"
            self.actual_device, self.batch_size = "cpu", 8
        def runtime_info(self): return {"library_version": "fixture"}
        def decide(self, state, questions, *, calls, stage, on_record, **kwargs):
            row = {"request": {"questions": questions}, "response": {"usage": {
                "input_tokens": 100, "output_tokens": 0, "truncated": False}}, "inference_seconds": .01}
            calls.append(row); on_record(row)
            return responses({"questions": questions})
    monkeypatch.setattr(frozen, "LayaDecisionClient", Laya)
    if fail_validation:
        def reject(*a): raise ValueError("simulated report integrity rejection")
        monkeypatch.setattr(frozen, "load_frozen_report", reject)
        with pytest.raises(ValueError, match="integrity rejection"):
            frozen.run_frozen_comparison(root, settings, progress=None)
        assert pointer.read_text(encoding="utf-8") == "previous-completed-run"
        directory = next((root/"results").glob("frozen_*"))
        assert read_json(directory/"failed.json")["partial_results_preserved"]
        assert len(read_jsonl(directory/"system.jsonl")) == 8
    else:
        directory = frozen.run_frozen_comparison(root, settings, progress=None)
        frame, summary, *_ = frozen.load_frozen_report(root, directory)
        assert len(frame) == 8 and set(summary.method) == set(frozen.METHODS)
        assert read_json(directory/"completed.json")["physical_api_attempts"] == {"jev": 6, "openai": 8}
        assert len(read_json(directory/"warmup.json")) == 4
        assert pointer.read_text(encoding="utf-8") == str(directory.relative_to(root))
        assert frame[frame.method == "D-parallel"].retrieval_api_cost_usd.eq(0).all()
        write_json(directory/"failed.json", {"fixture": True})
        with pytest.raises(ValueError, match="comparison failed"):
            frozen.load_frozen_report(root, directory)
