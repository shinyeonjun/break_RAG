import copy
import json
from pathlib import Path

import httpx
import pytest

from rag_baselines.c_retrieval import ComparativeRetrieval, JevRetrieval
from rag_baselines.c_tree import assemble_tree, load_tree, prepare_tree, validate_tree
from rag_baselines.data import read_json, write_jsonl
from rag_baselines.evaluation import measure_retrieval, measure_system, retrieval_metrics
from rag_baselines.jev import JevBudgetExceeded, JevClient, call_usage
from rag_baselines.store import DocumentStore


QUERY = "How did advertising spend and sales revenue change from August to September?"


@pytest.fixture
def setup(tmp_path):
    config = read_json(Path(__file__).resolve().parents[1] / "config/c.json")
    config["prepare"].update({"group_size": 1, "alternative_placement_probability": 0.2})
    config["search"]["frontier_width"] = 1
    taxonomy = {"id": "root", "title": "Business", "description": "Business records", "children": [
        {"id": "ads", "title": "Advertising", "description": "Campaign spending and performance"},
        {"id": "sales", "title": "Sales", "description": "Sales revenue and payment records"},
        {"id": "other", "title": "Other", "description": "Unrelated business records"}]}
    passages = [
        {"id": "ad:0", "title": "Advertising", "content": "Advertising spend was 10 in August and 20 in September.", "wikipedia_id": "ad", "source_spans": []},
        {"id": "sale:0", "title": "Sales", "content": "Sales revenue was 100 in August and 120 in September.", "wikipedia_id": "sale", "source_spans": []},
        {"id": "hr:0", "title": "Payroll", "content": "Payroll salary schedules and vacation rules.", "wikipedia_id": "hr", "source_spans": []}]
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, passages)
    store = DocumentStore(corpus)
    return tmp_path, config, taxonomy, corpus, store


def decision_server(config, captured, *, recover_sales=False, fail_call=None):
    def handler(request):
        assert str(request.url) == JevClient.endpoint
        assert request.headers["Authorization"] == "Bearer fixture-key"
        body = json.loads(request.content)
        captured.append(body)
        if fail_call == len(captured):
            return httpx.Response(429, json={"error": "secret must not be recorded: fixture-key"})
        answers = {}
        for key, question in body["questions"].items():
            kind, instructions = question["type"], question["instructions"]
            if kind == "choice":
                title = instructions["passage"]["title"]
                choice = "ads" if title == "Advertising" else "sales" if title == "Sales" else "other"
                answers[key] = {"type": "choice", "choice": choice, "confidence": 1.0,
                                "probabilities": {option: float(option == choice) for option in question["criteria"]}}
            elif kind == "score":
                relevant = instructions["passage"]["title"] in ("Advertising", "Sales")
                score = 3 if relevant else 0
                answers[key] = {"type": "score", "score": float(score), "confidence": 1.0,
                                "probabilities": {str(i): float(i == score) for i in range(4)}}
            elif key == "complete":
                texts = " ".join(item["text"] for item in body["state"]["evidence"])
                complete = "Advertising spend" in texts and "Sales revenue" in texts
                answers[key] = {"type": "noul", "noul": 0.99 if complete else 0.01}
            else:
                title = instructions["branch"]["title"]
                probability = 0.9 if title.startswith(("Advertising", "Sales")) else 0.01
                if recover_sales and title == "Sales":
                    probability = 0.2
                answers[key] = {"type": "noul", "noul": probability}
        return httpx.Response(200, json={"model": config["model"], "answers": answers,
                                       "usage": {"input_tokens": 100, "output_tokens": 10}})
    return httpx.Client(transport=httpx.MockTransport(handler))


def make_engine(setup, **server_options):
    root, config, taxonomy, _, store = setup
    captured = []
    client = JevClient(config, api_key="fixture-key", http_client=decision_server(config, captured, **server_options))
    tree = assemble_tree(store, taxonomy, {"ad:0": ["ads"], "sale:0": ["sales"], "hr:0": ["other"]}, config["prepare"])
    engine = JevRetrieval(root, store, tree, {"fixture": True}, config, client=client)
    return engine, captured


def test_c_collects_both_independent_branches_and_preserves_originals(setup):
    engine, captured = make_engine(setup)
    result = engine.retrieve(QUERY)
    assert {p["id"] for p in result["evidence"]} == {"ad:0", "sale:0"}
    assert result["trace"]["stop_reason"] == "coverage_satisfied"
    assert any(e["kind"] == "coverage" and e["noul"] < .5 for e in result["trace"]["events"])
    assert result["evidence_text_tokens"] <= engine.config["search"]["evidence_text_tokens"]
    engine.store.validate_evidence(result)
    assert result["api_usage"]["requests"] == len(captured)
    assert result["api_cost_usd"] == pytest.approx(len(captured) * .0000042)
    assert retrieval_metrics(result, {"evidence_groups": [["ad:0", "sale:0"]]})["complete_evidence_hit"] == 1
    assert not any(q["type"] == "choice" for r in captured for q in r["questions"].values())


def test_c_recovers_a_low_probability_branch_when_coverage_is_incomplete(setup):
    engine, _ = make_engine(setup, recover_sales=True)
    result = engine.retrieve(QUERY)
    assert {p["id"] for p in result["evidence"]} == {"ad:0", "sale:0"}
    assert result["trace"]["recoveries"] >= 1
    assert any(e["kind"] == "recovery" for e in result["trace"]["events"])


def test_top_k_keeps_ranked_candidates_below_the_legacy_cutoff(setup):
    setup[1]["search"].update(selection_mode="top_k", selection_k=2)
    engine, _ = make_engine(setup)
    evidence, _, _ = engine._evidence({pid: {"id": pid, "jev_score": score}
                                      for pid, score in (("ad:0", 1.4), ("sale:0", 1.3), ("hr:0", 0.1))})
    assert [p["id"] for p in evidence] == ["ad:0", "sale:0"]


def test_coverage_reuse_skips_only_identical_query_evidence_inputs(setup):
    setup[1]["search"]["coverage_threshold"] = .9999
    legacy, captured_legacy = make_engine(setup)
    original = legacy.retrieve(QUERY)
    setup[1]["search"]["reuse_coverage"] = True
    reused, captured_reused = make_engine(setup)
    result = reused.retrieve(QUERY)
    assert result["evidence"] == original["evidence"]
    assert result["candidates"] == original["candidates"]
    assert any(event.get("reused") for event in result["trace"]["events"])
    count = lambda captured: sum("complete" in call["questions"] for call in captured)
    assert count(captured_reused) < count(captured_legacy)
    assert result["api_usage"]["requests"] == len(captured_reused)
    assert result["api_cost_usd"] < original["api_cost_usd"]


def test_duplicate_source_references_are_scored_and_returned_once(setup):
    root, config, taxonomy, _, store = setup
    captured = []
    client = JevClient(config, api_key="fixture-key", http_client=decision_server(config, captured))
    tree = assemble_tree(store, taxonomy, {"ad:0": ["ads", "sales"], "sale:0": ["sales"], "hr:0": ["other"]}, config["prepare"])
    engine = JevRetrieval(root, store, tree, {}, config, client=client)
    result = engine.retrieve(QUERY)
    assert len(result["evidence"]) == len({p["id"] for p in result["evidence"]})
    scored_ids = [key for r in captured for key, q in r["questions"].items() if q["type"] == "score"]
    assert scored_ids.count("ad:0") == 1
    corrupt = copy.deepcopy(tree)
    corrupt["nodes"][tree["root"]]["children"].append(tree["root"])
    with pytest.raises(ValueError, match="cycle"):
        validate_tree(corrupt, store)


def test_c_call_budget_returns_partial_evidence_without_overrunning(setup):
    engine, captured = make_engine(setup)
    engine.config["search"]["max_api_calls"] = 1
    result = engine.retrieve(QUERY)
    assert len(captured) == 1
    assert result["trace"]["stop_reason"] == "api_call_budget"
    assert result["evidence"] == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_changed_call_budget_is_validated_before_making_requests(setup, limit):
    engine, captured = make_engine(setup)
    engine.config["search"]["max_api_calls"] = limit
    with pytest.raises(ValueError, match="decision-call budget"):
        engine.retrieve(QUERY)
    assert captured == []


def test_d_call_budget_uses_current_local_policy(setup):
    root, config, taxonomy, _, store = setup
    tree = assemble_tree(store, taxonomy, {"ad:0": ["ads"], "sale:0": ["sales"], "hr:0": ["other"]}, config["prepare"])
    local_config = {"model": {"repo": "fixture-local"}, "search": copy.deepcopy(config["search"])}
    local_config["search"].pop("max_api_calls")
    local_config["search"]["max_decision_calls"] = 32
    received_limits = []

    class LocalClient:
        model = "fixture-local"
        revision = "fixture-revision"

        def decide(self, state, questions, *, calls, max_calls, stage, on_record):
            received_limits.append(max_calls)
            if len(calls) >= max_calls:
                raise JevBudgetExceeded("Local call budget exceeded")
            record = {"stage": stage, "seconds": 0, "response": {"usage": {"input_tokens": 1, "output_tokens": 0}}}
            calls.append(record)
            on_record(record)
            return {key: {"noul": .9} for key in questions}

    engine = JevRetrieval(root, store, tree, {}, local_config, client=LocalClient(), method="D", provider="laya")
    engine.config["search"]["max_decision_calls"] = 1
    result = engine.retrieve(QUERY)
    assert received_limits and all(limit == 1 for limit in received_limits)
    assert result["local_usage"]["calls"] == 1
    assert result["trace"]["stop_reason"] == "decision_call_budget"
    assert result["api_usage"] is None


def test_quality_cache_is_frozen_and_online_search_makes_fresh_calls(setup):
    engine, captured = make_engine(setup)
    first = engine.retrieve_cached_quality(QUERY)
    before = len(captured)
    replay = engine.retrieve_cached_quality(QUERY)
    assert len(captured) == before
    assert not first["decision_cache_hit"] and replay["decision_cache_hit"]
    assert replay["seconds"]["retrieval_total"] is None
    assert replay["api_cost_usd"] == 0 and replay["api_usage"]["requests"] == 0
    assert replay["original_api_usage"]["requests"] > 0
    engine.retrieve(QUERY)
    assert len(captured) > before
    cache = next((engine.root / "data/c_retrieval_cache").glob("*.json"))
    if cache.name.endswith(".hash.json"):
        cache = next(p for p in cache.parent.glob("*.json") if not p.name.endswith(".hash.json"))
    cache.write_text(cache.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        engine.retrieve_cached_quality(QUERY)


def test_preparation_resumes_completed_placements_without_question_labels(setup):
    root, config, taxonomy, corpus, store = setup
    config["questions_per_request"] = 1
    captured = []
    failing = JevClient(config, api_key="fixture-key", http_client=decision_server(config, captured, fail_call=2))
    with pytest.raises(RuntimeError, match="429"):
        prepare_tree(root, corpus, taxonomy, config, client=failing, progress=None)
    captured_again = []
    client = JevClient(config, api_key="fixture-key", http_client=decision_server(config, captured_again))
    tree, metadata = prepare_tree(root, corpus, taxonomy, config, client=client, progress=None)
    validate_tree(tree, store)
    assert len(captured_again) == 2
    assert "ad:0" not in {key for call in captured_again for key in call["questions"]}
    assert metadata["api_usage"]["requests"] == 4
    assert not metadata["api_usage"]["usage_complete"] and metadata["api_cost_usd"] is None
    assert QUERY not in json.dumps(captured + captured_again)
    assert "fixture-key" not in (Path(metadata["directory"]) / "checkpoint.json").read_text()
    assert load_tree(root, store, taxonomy, config)[1]["cache_hit"]


def test_request_guards_split_without_truncation_and_validate_outputs(setup):
    _, config, _, _, _ = setup
    config["questions_per_request"] = 1
    captured = []
    client = JevClient(config, api_key="fixture-key", http_client=decision_server(config, captured))
    questions = {"ads": {"type": "noul", "instructions": {"branch": {"title": "Advertising"}}},
                 "sales": {"type": "noul", "instructions": {"branch": {"title": "Sales"}}}}
    calls = []
    with pytest.raises(JevBudgetExceeded):
        client.decide({"query": QUERY}, questions, calls=calls, max_calls=1, stage="navigation")
    assert captured == [] and calls == []
    answers = client.decide({"query": QUERY}, questions, calls=calls, max_calls=2, stage="navigation")
    assert set(answers) == set(questions) and len(captured) == 2
    assert all(r["state"]["query"] == QUERY for r in captured)
    with pytest.raises(ValueError, match="no text was truncated"):
        client.batches("x" * config["state_question_max_bytes"], questions)
    invalid = {"model": config["model"], "usage": {"input_tokens": 1, "output_tokens": 1},
               "answers": {"ads": {"type": "noul", "noul": float("nan")}}}
    with pytest.raises(ValueError, match="numeric"):
        client.validate_response(invalid, {"ads": questions["ads"]})


def test_c_integrates_shared_measurement_generator_and_incremental_results(setup):
    engine, _ = make_engine(setup)
    store = engine.store

    class AB:
        def __init__(self):
            self.store = store
            self.calls = []

        def retrieve(self, question, method):
            self.calls.append(method)
            return {"method": method, "question": question, "corpus_sha256": store.sha256,
                    "evidence": [store.get("ad:0"), store.get("sale:0")], "candidates": [{"id": "ad:0"}, {"id": "sale:0"}],
                    "seconds": {"retrieval_total": 0.01}, "api_usage": None, "api_cost_usd": None}

    ab = AB()
    combined = ComparativeRetrieval(ab, engine)
    questions = [{"id": "q", "question": QUERY, "task": "multi"}]
    labels = [{"id": "q", "answers": ["10 and 20"], "evidence_groups": [["ad:0", "sale:0"]]}]
    retrieval_checkpoints = []
    measured = measure_retrieval(combined, questions, labels, methods=("A", "B", "C"), progress=None,
                                 on_result=retrieval_checkpoints.append)
    assert len(retrieval_checkpoints) == 3
    assert {r["method"] for r in measured} == {"A", "B", "C"}
    assert all(r["metrics"]["complete_evidence_hit"] == 1 for r in measured)

    from types import SimpleNamespace
    class Responses:
        def create(self, **kwargs):
            assert "secret_gold" not in kwargs["input"]
            return SimpleNamespace(output_text='{"answer":"10 and 20","citations":["ad:0","sale:0"]}',
                model_dump=lambda **unused: {"model": "gpt-6-luna", "usage": {"input_tokens": 100, "output_tokens": 10}})
    config = {"model": "gpt-6-luna", "max_output_tokens": 256, "encoding_name": "o200k_base",
              "prices": {"input_usd_per_million": .1, "cached_input_usd_per_million": .01, "output_usd_per_million": .5},
              "reasoning_effort": "none", "temperature": 0, "service_tier": "default"}
    checkpoints = []
    results = measure_system(combined, questions, labels, client=SimpleNamespace(responses=Responses()),
                             generation_config=config, max_calls=3, methods=("A", "B", "C"),
                             progress=None, on_result=checkpoints.append)
    assert len(checkpoints) == 3 and all(r["answer_metrics"]["answer_em"] == 1 for r in results)
    c = next(r for r in results if r["retrieval"]["method"] == "C")
    assert c["online_api_cost_usd"] == pytest.approx(c["retrieval_api_cost_usd"] + c["generation_api_cost_usd"])


def test_evidence_budget_keeps_whole_originals_and_reports_exclusions(setup):
    engine, _ = make_engine(setup)
    passage = engine.store.get("ad:0")
    one_cost = len(engine.encoder.encode(f"[{passage['id']}] {passage['content']}"))
    engine.config["search"]["evidence_text_tokens"] = one_cost
    result = engine.retrieve(QUERY)
    assert [p["id"] for p in result["evidence"]] == ["ad:0"]
    assert result["evidence"][0]["content"] == passage["content"]
    assert "sale:0" in result["trace"]["budget_excluded_ids"]
    assert result["evidence_text_tokens"] <= one_cost
    assert result["trace"]["stop_reason"] != "coverage_satisfied"
    tampered = {**result, "evidence": [{**result["evidence"][0], "wikipedia_id": "wrong-source"}]}
    with pytest.raises(ValueError, match="original"):
        engine.store.validate_evidence(tampered)


def test_invalid_usage_does_not_hide_original_failure_or_claim_zero_cost():
    for invalid in (None, "bad", [], {"input_tokens": "bad", "output_tokens": 1}, {"input_tokens": 1, "output_tokens": -1}):
        calls = [{"status": "failed", "response": {"usage": invalid}}]
        usage, cost = call_usage(calls, {"input_usd_per_million": .042, "output_usd_per_million": 0})
        assert not usage["usage_complete"] and cost is None


def test_live_probability_rounding_is_accepted_without_normalizing(setup):
    _, config, _, _, _ = setup
    client = JevClient(config, api_key="fixture-key", http_client=decision_server(config, []))
    question = {"type": "choice", "criteria": {"a": "A", "b": "B", "c": "C"}}
    body = {"model": config["model"], "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"q": {"type": "choice", "choice": "a", "confidence": .51,
                              "probabilities": {"a": .55, "b": .26, "c": .18}}}}
    original = copy.deepcopy(body)
    client.validate_response(body, {"q": question})
    assert body == original and sum(body["answers"]["q"]["probabilities"].values()) == pytest.approx(.99)
    body["answers"]["q"]["probabilities"]["a"] = .40
    with pytest.raises(ValueError, match="rounding"):
        client.validate_response(body, {"q": question})
    score_question = {"type": "score", "criteria": ["none", "background", "partial", "direct"]}
    body["answers"]["q"] = {"type": "score", "score": 2.92, "confidence": .97,
                              "probabilities": {"0": .01, "1": .01, "2": .01, "3": .97}}
    client.validate_response(body, {"q": score_question})
    body["answers"]["q"]["score"] = 2.80
    with pytest.raises(ValueError, match="disagrees"):
        client.validate_response(body, {"q": score_question})


def test_jev_invalid_prices_are_rejected_before_requests(setup):
    _, config, _, _, _ = setup
    captured = []
    config["prices"]["input_usd_per_million"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        JevClient(config, api_key="fixture-key", http_client=decision_server(config, captured))
    assert captured == []
