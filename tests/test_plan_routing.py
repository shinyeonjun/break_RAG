from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow
import pytest

from rag_baselines.data import read_json, read_jsonl, write_json, write_jsonl
from rag_baselines.plan_routing import PlanIndex, PlanRetrieval, validated_facts
from rag_baselines.plan_experiment import validate_settings, extraction_format, packing_variability
from rag_baselines.store import DocumentStore


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def index(tmp_path):
    passages = [
        {"id":"film","title":"Film Alpha","content":"Film Alpha was directed by Person Beta. It was released in 2000.","source_spans":[],"wikipedia_id":"1"},
        {"id":"person","title":"Person Beta","content":"Person Beta was born in City Gamma. Person Beta later lived in City Delta.","source_spans":[],"wikipedia_id":"2"},
        {"id":"wrong","title":"Film Alpha reviews","content":"Film Alpha reviews praised its soundtrack.","source_spans":[],"wikipedia_id":"3"}]
    write_jsonl(tmp_path/"corpus.jsonl",passages)
    raw = {"passages":[
        {"id":"film","facts":[{"subject":"Film Alpha","relation":"directed by","object":"Person Beta","quote":"Film Alpha was directed by Person Beta."}]},
        {"id":"person","facts":[{"subject":"Person Beta","relation":"born in","object":"City Gamma","quote":"Person Beta was born in City Gamma."}]},
        {"id":"wrong","facts":[{"subject":"Film Alpha","relation":"review praised","object":"soundtrack","quote":"Film Alpha reviews praised its soundtrack."}]}]}
    facts,rejected = validated_facts(raw,passages,8)
    assert not rejected
    policy = {**read_json(ROOT/"config/plan_routing.json")["search"],"quote_context_characters":1,"selected_plans":1}
    return PlanIndex(DocumentStore(tmp_path/"corpus.jsonl"),facts,policy)


def test_two_edge_join_is_source_grounded_and_does_not_use_gold(index):
    candidates,diagnostics = index.candidates("Where was the director of Film Alpha born?")
    assert diagnostics["seed_mode"] == "exact_entity"
    assert any(set(p["passage_ids"]) == {"film","person"} for p in candidates)
    assert set(diagnostics["candidate_passage_ids"]) == {"film","person","wrong"}


def test_fabricated_or_mislocated_source_is_rejected(index):
    passage = index.store.get("person")
    body = {"passages":[{"id":"person","facts":[{"subject":"Person Beta","relation":"born in","object":"City Fake","quote":"Person Beta was born in City Fake."}]}]}
    facts,rejected = validated_facts(body,[passage],8)
    assert not facts and rejected[0]["reason"] == "quote_not_verbatim"
    broken = [dict(next(iter(index.facts.values())),start=3)]
    with pytest.raises(ValueError,match="coordinates"):
        PlanIndex(index.store,broken,index.policy)


def test_multiple_paths_and_packing_keep_original_retrieval_contract(index):
    def decide(state,questions,*,calls,max_calls,stage,on_record):
        answers = {key:{"noul":0.9 if len(q["instructions"]["source_backed_plan"]) == 2 else 0.1} for key,q in questions.items()}
        calls.append({"response":{"usage":{"input_tokens":30,"output_tokens":0}}})
        return answers
    adapter = PlanRetrieval(index,SimpleNamespace(decide=decide),provider="jev",
                            prices={"input_usd_per_million":0.042,"output_usd_per_million":0})
    result = adapter.retrieve("Where was the director of Film Alpha born?","C_pack")
    assert {p["id"] for p in result["evidence"]} == {"film","person"}
    index.store.validate_evidence(result)
    packed = index.packed_view(result)
    assert result["evidence"][0]["content"] == index.store.get(result["evidence"][0]["id"])["content"]
    for trace in packed["packing_trace"]:
        source = index.store.get(trace["passage_id"])["content"]
        assert [source[a:b] for a,b in trace["passage_character_spans"]] == trace["quotes"]
    assert result["decision_usage"]["input_tokens"] == 30
    assert result["api_cost_usd"] == pytest.approx(30*0.042/1e6)


def test_unknown_entity_and_caps_are_visible(index):
    plans,diagnostics = index.candidates("Unrelated ZetaUnknownThing")
    assert not plans and diagnostics["seed_mode"] == "no_seed"
    index.policy["max_candidate_plans"] = 1
    plans,diagnostics = index.candidates("Film Alpha")
    assert len(plans) == 1 and diagnostics["candidate_cap_active"]


def test_budgets_checked_before_generation():
    settings = read_json(ROOT/"config/plan_routing.json")
    validate_settings(settings)
    with pytest.raises(ValueError,match="budget"):
        validate_settings({**settings,"max_generation_calls":1})
    with pytest.raises(ValueError,match="development"):
        validate_settings({**settings,"split":"eval"})
    with pytest.raises(ValueError,match="positive integer"):
        validate_settings({**settings,"questions_per_task":True})


def test_extraction_budget_overflow_is_explicitly_rejected_not_lost(index):
    p=index.store.get("person")
    f={"subject":"Person Beta","relation":"born in","object":"City Gamma","quote":"Person Beta was born in City Gamma."}
    facts,rejected=validated_facts({"passages":[{"id":"person","facts":[f,f]}]},[p],1)
    assert len(facts)==1 and rejected[0]["reason"]=="exceeds_explicit_fact_budget"
    schema=extraction_format([p],1)
    assert schema["strict"] is True
    row=schema["schema"]["properties"]["passages"]["items"]
    assert row["properties"]["facts"]["maxItems"]==1
    assert row["properties"]["id"]["enum"]==["person"]


def test_punctuation_only_value_is_an_explicit_index_rejection():
    p={"id":"a","title":"Alpha","content":"Alpha value is —.","source_spans":[],"wikipedia_id":"a"}
    fact={"subject":"Alpha","relation":"value","object":"—","quote":p["content"]}
    facts,rejected=validated_facts({"passages":[{"id":"a","facts":[fact]}]},[p],8)
    assert not facts and rejected[0]["reason"]=="empty_normalized_entity"


def test_plan_adapter_completes_the_existing_generator_contract(index):
    import json
    from rag_baselines.generation import generate_answer
    result=PlanRetrieval(index).retrieve("Where was Film Alpha's director born?","P_code")
    result.update({"question_id":"sample","repeat":0})
    view=index.packed_view(result)
    payload={"answer":"City Gamma","citations":[p["id"] for p in view["evidence"]]}
    usage={"input_tokens":100,"output_tokens":20,"input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0}}
    response=SimpleNamespace(output_text=json.dumps(payload),model_dump=lambda **_:{"model":"mock","usage":usage})
    client=SimpleNamespace(responses=SimpleNamespace(create=lambda **_:response))
    generation=read_json(ROOT/"config/generation.json")
    keys=("model","max_output_tokens","encoding_name","prices","reasoning_effort","temperature","service_tier")
    generated=generate_answer(view,client=client,**{k:generation[k] for k in keys})
    assert generated["answer"]=="City Gamma" and generated["valid_output"]
    assert generated["prompt"]["passage_ids"]==[p["id"] for p in result["evidence"]]
    assert generated["seconds"]["retrieval_plus_generation"]>=result["seconds"]["retrieval_total"]


def test_live_selector_variability_is_reported_not_misread_as_packing(index):
    from copy import deepcopy
    r=PlanRetrieval(index).retrieve("Film Alpha","C_plan")
    r.update({"question_id":"q","repeat":0})
    packed=deepcopy(r)
    packed["method"]="C_pack"
    assert packing_variability([{"retrieval":r},{"retrieval":packed}])[0]["packing_controlled"]
    packed["selected_fact_ids"]=[]
    observed=packing_variability([{"retrieval":r},{"retrieval":packed}])[0]
    assert observed["same_plan_order"] and not observed["packing_controlled"]


def test_failed_batch_still_saves_other_running_success_and_closes_clients(tmp_path,monkeypatch):
    import json
    from threading import Barrier
    import rag_baselines.plan_experiment as experiment
    for name in ("generation","c"):
        write_json(tmp_path/f"config/{name}.json",read_json(ROOT/f"config/{name}.json"))
    settings=read_json(ROOT/"config/plan_routing.json")
    settings["extraction"]={**settings["extraction"],"passages_per_request":1,"workers":2}
    passages=[{"id":key,"title":"Alpha","content":"Alpha founded Beta.","source_spans":[],"wikipedia_id":key}
              for key in ("bad","good")]
    write_jsonl(tmp_path/settings["corpus"],passages)
    barrier,clients=Barrier(2),[]
    class Client:
        def __init__(self,**kwargs):
            self.transport=kwargs["http_client"]
            self.responses=SimpleNamespace(create=self.create)
            self.closed=False
            clients.append(self)
        def create(self,**kwargs):
            data=json.loads(kwargs["input"].split("PASSAGES:\n")[1])
            key=data[0]["id"]
            barrier.wait(timeout=5)
            payload={"passages":[{"id":key,"facts":[{"subject":"Alpha","relation":"founded","object":"Beta",
                "qualifiers":"","quote":"Alpha founded Beta."}]}]}
            raw={"model":"mock","status":"completed" if key=="good" else "incomplete",
                 "incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":1,"output_tokens":1}}
            return SimpleNamespace(output_text=json.dumps(payload),model_dump=lambda **_:raw)
        def close(self):
            self.closed=True
            self.transport.close()
    monkeypatch.setattr(experiment,"OpenAI",Client)
    with pytest.raises(ValueError,match="incomplete"):
        experiment.prepare_fact_index(tmp_path,settings,progress=None)
    checkpoint=next((tmp_path/"data/plan_index").glob("*/batches.jsonl"))
    rows=read_jsonl(checkpoint)
    assert len(rows)==1 and rows[0]["passage_ids"]==["good"]
    assert all(c.closed for c in clients)
    assert next((tmp_path/"data/plan_index").glob("*/failures.jsonl")).exists()
