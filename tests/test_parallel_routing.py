import copy
import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from rag_baselines.c_tree import assemble_tree
from rag_baselines.data import read_json, write_json, write_jsonl
from rag_baselines.jev import JevClient
from rag_baselines.parallel_experiment import choose_policy, summarize_quality, unseen_evaluation_questions, validate_settings
from rag_baselines.parallel_routing import ParallelRouting, balanced_groups, routing_quality, validate_groups, normalize_descriptor, parse_descriptor_text, reuse_description_checkpoints
from rag_baselines.c_tree import digest
from rag_baselines.store import DocumentStore


@pytest.fixture
def corpus(tmp_path):
    passages = [{"id": key, "title": title, "content": text, "source_spans": [], "wikipedia_id": key}
                for key, title, text in [("a", "Advertising", "Advertising spend was 20 in September."),
                                          ("b", "Sales", "Sales revenue was 120 in September."),
                                          ("c", "Payroll", "Payroll vacation rules.")]]
    write_jsonl(tmp_path/"corpus.jsonl", passages)
    store = DocumentStore(tmp_path/"corpus.jsonl")
    taxonomy = {"id": "root", "title": "Records", "description": "Records", "children": [
        {"id": key, "title": key, "description": key} for key in ("ads", "sales", "other")]}
    tree = assemble_tree(store, taxonomy, {"a": ["ads", "sales"], "b": ["sales"], "c": ["other"]},
                         {"group_size": 1, "descriptor_words": 8})
    groups = balanced_groups(store, tree, 1)
    for g in groups:
        g["descriptor"] = {"summary": store.get(g["passage_ids"][0])["title"]}
    return store, tree, groups


def adapter(tmp_path, corpus, *, calls_limit=16):
    store, tree, groups = corpus
    config = read_json(Path(__file__).resolve().parents[1]/"config/c.json")
    config["questions_per_request"] = 1
    captured = []
    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        answers = {}
        for qid, q in body["questions"].items():
            if q["type"] == "noul":
                title = q["instructions"]["collection"]["summary"]
                answers[qid] = {"type": "noul", "noul": {"Advertising": .9, "Sales": .8, "Payroll": .01}[title]}
            else:
                score = 3 if qid in ("a", "b") else 0
                answers[qid] = {"type": "score", "score": score, "confidence": 1.,
                                "probabilities": {str(i): float(i == score) for i in range(4)}}
        return httpx.Response(200, json={"model": config["model"], "answers": answers,
                                        "usage": {"input_tokens": 100, "output_tokens": 0}})
    client = JevClient(config, api_key="fixture-key", http_client=httpx.Client(transport=httpx.MockTransport(handle)))
    policy = {"selection_k": 2, "evidence_k": 2, "http_workers": 2, "max_calls_per_retrieval": calls_limit}
    return ParallelRouting(store, groups, client, "jev", policy, tmp_path/"calls.jsonl"), captured


def test_balanced_groups_preserve_every_original_once_despite_multiple_placements(corpus):
    store, tree, _ = corpus
    groups = balanced_groups(store, tree, 2)
    ids = [pid for g in groups for pid in g["passage_ids"]]
    assert sorted(ids) == ["a", "b", "c"]
    assert [len(g["passage_ids"]) for g in groups] == [2, 1]


def test_two_stages_have_distinct_physical_requests_and_fresh_latency(tmp_path, corpus):
    engine, captured = adapter(tmp_path, corpus)
    result = engine.retrieve("Compare advertising spend and sales revenue.")
    assert {p["id"] for p in result["evidence"]} == {"a", "b"}
    assert result["trace"]["logical_stages"] == 2
    assert result["trace"]["physical_http_requests"] == 5
    assert result["api_usage"]["requests"] == 5
    assert result["model_tokens"]["jev"]["nonpad"] == 500
    assert all("evidence_groups" not in json.dumps(body) for body in captured)
    engine.retrieve("Compare advertising spend and sales revenue.")
    assert len(captured) == 10
    assert result["seconds"]["retrieval_total"] > 0


def test_quality_subset_does_not_claim_cached_latency_or_double_bill(tmp_path, corpus):
    engine, captured = adapter(tmp_path, corpus)
    judged = engine.judge("Compare advertising spend and sales revenue.")
    result = engine.assemble(judged, 1, quality_only=True)
    label = {"evidence_groups": [["a", "b"]]}
    metrics = routing_quality(result, corpus[2], label, corpus[0])
    assert metrics["macro_complete"] == metrics["complete_evidence_hit"] == 0
    assert metrics["macro_tp"] == 1
    assert result["seconds"]["retrieval_total"] is None
    assert result["api_usage"] is None and result["api_cost_usd"] is None
    assert result["trace"]["calls"] == [] and len(captured) == 5
    complete = engine.assemble(judged, 2, quality_only=True)
    assert routing_quality(complete, corpus[2], label, corpus[0])["complete_evidence_hit"] == 1


def test_request_budget_fails_before_spending_on_the_next_stage(tmp_path, corpus):
    engine, captured = adapter(tmp_path, corpus, calls_limit=3)
    with pytest.raises(ValueError, match="budget"):
        engine.retrieve("Compare advertising spend and sales revenue.")
    assert len(captured) == 3
    assert len((tmp_path/"calls.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_duplicate_or_missing_group_reference_is_rejected(corpus):
    groups = copy.deepcopy(corpus[2])
    groups[-1]["passage_ids"] = ["a"]
    with pytest.raises(ValueError, match="exactly once"):
        validate_groups(groups, corpus[0])


def test_policy_selection_honors_quality_constraint_and_reports_infeasibility():
    table = pd.DataFrame([{"provider": "jev", "group_size": 20, "selection_k": 1,
                           "complete_evidence_hit": .8, "input_proxy_tokens": 100},
                          {"provider": "jev", "group_size": 50, "selection_k": 3,
                           "complete_evidence_hit": .95, "input_proxy_tokens": 300}])
    assert choose_policy(table, "jev", .9)["selection_k"] == 3
    assert choose_policy(table, "jev", .99)["target_met"] is False


def test_holdout_excludes_ids_from_previous_evaluation_without_reading_labels(tmp_path):
    prepared = tmp_path/"data/prepared"
    prepared.mkdir(parents=True)
    write_jsonl(prepared/"questions_eval.jsonl", [
        {"id": f"{task}{i}", "task": task, "question": "query"}
        for task in ("nq", "hotpotqa") for i in range(3)])
    old = tmp_path/"results/staged_old/eval/ABCD_frozen"
    old.mkdir(parents=True)
    write_json(old/"manifest.json", {"split": "eval", "question_ids": ["nq0", "hotpotqa0"]})
    questions, excluded = unseen_evaluation_questions(tmp_path, 1)
    assert [q["id"] for q in questions] == ["nq1", "hotpotqa1"]
    assert excluded == ["hotpotqa0", "nq0"]


def test_settings_bound_description_and_generation_calls_before_live_work():
    settings = read_json(Path(__file__).resolve().parents[1]/"config/parallel_routing.json")
    validate_settings(settings)
    settings["max_llm_calls"] = 1
    with pytest.raises(ValueError, match="call budget"):
        validate_settings(settings)


def test_description_compression_records_overflow_and_non_original_facts(corpus):
    store = corpus[0]
    group = {"passage_ids": ["a", "b"]}
    valid = {"question": "What was sales revenue?", "answer": "120", "passage_id": "b"}
    raw = {"summary": "Monthly metrics", "entities": ["Advertising"]*14 + ["Invented Company"],
           "qa": [valid, {"question": "What was sales revenue?", "answer": "999", "passage_id": "b"}]}
    descriptor, excluded = normalize_descriptor(raw, group, store)
    assert len(descriptor["entities"]) == 12
    assert descriptor["qa"] == [valid]
    assert len(excluded) == 4
    assert {e["reason"] for e in excluded} == {"declared_cap_12", "not_verbatim_original"}


def test_description_schema_failure_preserves_the_need_for_a_real_summary(corpus):
    with pytest.raises(ValueError, match="schema"):
        normalize_descriptor({"entities": [], "qa": []}, {"passage_ids": ["a"]}, corpus[0])


def test_provider_text_wrapper_is_recorded_without_synthesizing_json():
    value = {"summary": "Monthly metrics", "entities": [], "qa": []}
    parsed, wrapper = parse_descriptor_text('Explanatory prefix.\n```json\n'+json.dumps(value)+'\n```')
    assert parsed == value
    assert wrapper["prefix_characters"] > 0 and wrapper["suffix_characters"] > 0
    with pytest.raises(ValueError, match="exactly one"):
        parse_descriptor_text('{"summary": "Unfinished"')
    with pytest.raises(ValueError, match="exactly one"):
        parse_descriptor_text(json.dumps(value)+'\n'+json.dumps(value))


def test_checkpoint_reuse_validates_inputs_and_original_references(tmp_path, corpus):
    store = corpus[0]
    prior, target = tmp_path/'cache/old', tmp_path/'cache/new'
    prior.mkdir(parents=True);target.mkdir()
    group = {'id': 'group_0000', 'passage_ids': ['a']}
    signature = {'corpus_sha256':store.sha256, 'tree_sha256':'tree', 'group_size':1,
                 'description_policy':{'target':180}, 'generation':{'model':'pinned'}}
    descriptor = {'summary':'Advertising', 'entities':['Advertising'], 'qa':[]}
    saved = {'signature':{**signature,'code_sha256':'previous-online-code'}, 'passage_ids':['a'],
             'descriptor':descriptor,'descriptor_sha256':digest(descriptor)}
    write_json(prior/'group_0000.json',saved)
    reuse_description_checkpoints(target,signature,[group],store)
    copied=read_json(target/'group_0000.json')
    assert copied['descriptor']==descriptor and copied['signature']==signature
    assert copied['reused_checkpoint']['sha256']
    saved['descriptor']['entities']=['Invented Company']
    saved['descriptor_sha256']=digest(saved['descriptor'])
    write_json(prior/'group_0001.json',saved)
    reuse_description_checkpoints(target,signature,[{'id':'group_0001','passage_ids':['a']}],store)
    assert not (target/'group_0001.json').exists()
