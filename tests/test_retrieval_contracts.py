import json
from pathlib import Path

import pytest

from rag_baselines.data import (build_common_corpora, chunk_article, map_provenance,
                                read_jsonl, write_json)
from rag_baselines.evaluation import measure_retrieval, retrieval_metrics
from rag_baselines.generation import answer_metrics, build_prompt, usage_cost, grounding_audit, component_metrics


def provenance(start, end, paragraph=0):
    return {"wikipedia_id": "7", "bleu_score": 1.0,
            "start_paragraph_id": paragraph, "end_paragraph_id": paragraph,
            "start_character": start, "end_character": end}


def test_chunk_text_matches_bergen_and_keeps_original_coordinates():
    article = {"wikipedia_id": "7", "wikipedia_title": "Title",
               "text": ["  alpha beta\n", "BULLET::::discard this", "Section::::Body. gamma delta"]}
    passages = chunk_article(article, 3)
    assert [p["content"] for p in passages] == ["Title. alpha beta Section:Body.", "Title. gamma delta"]
    assert passages[0]["source_spans"][0] == {"paragraph_id": 0, "start_character": 2, "end_character": 12}
    assert map_provenance(provenance(0, 6, 1), article, passages) is None


def test_span_crossing_chunks_requires_both_and_rejects_bad_offsets():
    article = {"wikipedia_id": "7", "wikipedia_title": "Title", "text": ["첫째 alpha beta gamma"]}
    passages = chunk_article(article, 2)
    assert map_provenance(provenance(3, 13), article, passages) == ["7:0", "7:1"]
    assert map_provenance(provenance(0, 100), article, passages) is None
    result = {"evidence": [{"id": "7:0"}], "candidates": [{"id": "7:0"}, {"id": "7:1"}]}
    metrics = retrieval_metrics(result, {"evidence_groups": [["7:0", "7:1"]]})
    assert metrics["complete_evidence_hit"] == 0
    assert metrics["complete_candidate_hit"] == 1
    assert metrics["best_group_coverage"] == 0.5


def test_independent_a_b_execution_and_offline_label_use():
    class RecordingEngine:
        def __init__(self):
            self.calls = []

        def retrieve(self, question, method):
            self.calls.append((question, method))
            return {"method": method, "evidence": [{"id": "7:0"}], "candidates": [{"id": "7:0"}]}

    engine = RecordingEngine()
    results = measure_retrieval(engine, [{"id": "q", "question": "Which?"}],
                               [{"id": "q", "answers": ["secret label"], "evidence_groups": [["7:0"]]}],
                               repeats=3, progress=None)
    assert len(engine.calls) == 6
    assert sum(method == "A" for _, method in engine.calls) == 3
    assert sum(method == "B" for _, method in engine.calls) == 3
    assert all(question == "Which?" for question, _ in engine.calls)
    assert all(r["metrics"]["complete_evidence_hit"] == 1 for r in results)
    with pytest.raises(ValueError, match="positive"):
        measure_retrieval(engine, [], [], repeats=0)


def test_nested_common_corpus_is_reproducible_and_labels_are_separate(tmp_path):
    config = {"seed": 42, "qa": {"dev_count": 1, "eval_count": 1}, "words_per_passage": 1}
    write_json(tmp_path / "config/baselines.json", config)
    article = {"wikipedia_id": "7", "wikipedia_title": "Title", "text": [" ".join(f"word{i}" for i in range(20010))]}
    question = {"id": "dev-q", "input": "Find the first word.",
                "output": [{"answer": "word0", "provenance": [provenance(0, 5)]}]}
    eval_question = {**question, "id": "eval-q"}
    candidates = {"dev": [question], "eval": [eval_question]}
    first = build_common_corpora(tmp_path, candidates, [article], {"source": "fixture"})
    second = build_common_corpora(tmp_path, candidates, [article], {"source": "fixture"})
    assert first == second
    sets = []
    for size in (1000, 5000, 20000):
        rows = read_jsonl(tmp_path / f"data/prepared/corpus_{size}.jsonl")
        sets.append({row["id"] for row in rows})
        assert len(rows) == size
        assert "7:0" in sets[-1]
        assert all("answers" not in row and "evidence_groups" not in row for row in rows)
    assert sets[0] < sets[1] < sets[2]
    online = read_jsonl(tmp_path / "data/prepared/questions_dev.jsonl")[0]
    assert set(online) == {"id", "question", "task"}


def test_generation_budget_cost_and_answer_metrics_without_api_calls():
    result = {"question": "What?", "evidence": [{"id": "7:0", "content": "alpha beta"}]}
    assert build_prompt(result)["passage_ids"] == ["7:0"]
    with pytest.raises(ValueError, match="budget"):
        build_prompt(result, evidence_tokens=1)
    assert usage_cost(None, None) is None
    assert usage_cost({"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200}, "output_tokens": 100},
                      {"input_usd_per_million": 2, "cached_input_usd_per_million": 1, "output_usd_per_million": 4}) == pytest.approx(0.0022)
    assert answer_metrics("The Red Fox!", ["red fox"])["answer_em"] == 1.0
    assert answer_metrics(None, ["red fox"])["answer_f1"] == 0.0


def test_invalid_usage_is_unknown_and_bad_prices_fail_before_paid_request():
    from types import SimpleNamespace
    from rag_baselines.generation import generate_answer
    prices = {"input_usd_per_million": .1, "cached_input_usd_per_million": .01, "output_usd_per_million": .5}
    for invalid in ("bad", {"input_tokens": 0, "output_tokens": -10},
                    {"input_tokens": 1, "output_tokens": 1, "input_tokens_details": []},
                    {"input_tokens": 1, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 2}}):
        assert usage_cost(invalid, prices) is None
    assert usage_cost({"input_tokens": 1, "output_tokens": 1,
                       "input_tokens_details": {"cache_write_tokens": 1}}, prices) is None
    calls = []
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs)))
    result = {"question": "What?", "evidence": []}
    for invalid_price in (float("nan"), float("inf"), -1, "bad"):
        with pytest.raises(ValueError, match="finite"):
            generate_answer(result, client=client, model="gpt-6-luna",
                            prices={**prices, "output_usd_per_million": invalid_price})
    assert not calls


def test_no_evidence_lucky_answer_is_not_protocol_compliant():
    assert grounding_audit("Spike Slawson", [], [])["status"] == "unsupported_no_evidence"
    assert not grounding_audit("Spike Slawson", [], [])["protocol_compliant"]
    assert grounding_audit("I do not know", [], [])["protocol_compliant"]
    cited = grounding_audit("Spike Slawson", ["7:0"], ["7:0"])
    assert cited["protocol_compliant"] and not cited["citation_entailment_verified"]


def test_joint_answer_audit_does_not_turn_one_component_into_a_complete_answer():
    components = ["English language", "history", "traditional and national symbols", "Australian values"]
    assert component_metrics("English language", components) == {"component_coverage": .25, "components_complete": 0.0}
    complete = "; ".join(components)
    assert component_metrics(complete, components)["components_complete"] == 1
    assert answer_metrics(complete, components)["answer_em"] == 0  # preserve benchmark EM


def test_multi_document_annotation_requires_all_pages(tmp_path):
    from rag_baselines.data import provenance_options
    question = {"output": [{"answer": "yes", "provenance": [provenance(0, 5), {**provenance(0, 5), "wikipedia_id": "8"}]}]}
    assert len(provenance_options(question)[0]) == 2
    label = {"evidence_groups": [["7:0", "8:0"]]}
    result = {"evidence": [{"id": "7:0"}], "candidates": [{"id": "7:0"}, {"id": "8:0"}]}
    assert retrieval_metrics(result, label)["complete_evidence_hit"] == 0
    result["evidence"].append({"id": "8:0"})
    assert retrieval_metrics(result, label)["complete_evidence_hit"] == 1


def test_luna_http_request_response_usage_and_cache_write_cost(tmp_path):
    import httpx
    from openai import OpenAI
    from rag_baselines.generation import generate_answer
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 1,
            "status": "completed", "model": "gpt-6-luna",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "annotations": [], "text": '{"answer":"Paris","citations":["7:0"]}'}]}],
            "usage": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 200, "cache_write_tokens": 300},
                      "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1100},
        })

    client = OpenAI(api_key="test-key", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = {"question_id": "q", "question": "Capital?", "method": "A", "repeat": 0,
              "seconds": {"retrieval_total": 0.01}, "evidence": [{"id": "7:0", "content": "Paris."}]}
    output = generate_answer(result, client=client, model="gpt-6-luna", prices={
        "input_usd_per_million": .10, "cached_input_usd_per_million": .01,
        "cache_write_usd_per_million": .125, "output_usd_per_million": .50})
    assert captured[0]["model"] == "gpt-6-luna"
    assert captured[0]["reasoning"] == {"effort": "none"}
    assert captured[0]["temperature"] == 0
    assert captured[0]["max_output_tokens"] == 256
    assert output["answer"] == "Paris" and output["valid_output"]
    assert output["api_usage"]["input_tokens_details"]["cache_write_tokens"] == 300
    assert output["api_cost_usd"] == pytest.approx(0.0001395)
    quality_result = {**result, "seconds": {"retrieval_total": None}}
    quality_output = generate_answer(quality_result, client=client, model="gpt-6-luna")
    assert quality_output["answer"] == "Paris"
    assert quality_output["seconds"]["retrieval_plus_generation"] is None
    # Exercise the complete live-system orchestration through a real SDK/mock HTTP.
    from rag_baselines.data import write_jsonl
    from rag_baselines.evaluation import measure_system
    from rag_baselines.store import DocumentStore
    passage = {"id": "7:0", "content": "Paris.", "title": "Paris", "wikipedia_id": "7", "source_spans": []}
    write_jsonl(tmp_path / "corpus.jsonl", [passage])

    class FixtureAdapter:
        store = DocumentStore(tmp_path / "corpus.jsonl")

        def retrieve(self, question, method):
            return {**result, "question": question, "method": method,
                    "corpus_sha256": self.store.sha256, "evidence": [passage], "candidates": [{"id": "7:0"}]}

    config = {"model": "gpt-6-luna", "max_output_tokens": 256, "encoding_name": "o200k_base",
              "prices": None, "reasoning_effort": "none", "temperature": 0, "service_tier": "default"}
    system = measure_system(FixtureAdapter(), [{"id": "q", "question": "Capital?"}],
                            [{"id": "q", "answers": ["Paris"], "evidence_groups": [["7:0"]]}],
                            client=client, generation_config=config, max_calls=2, progress=None)
    assert {row["retrieval"]["method"] for row in system} == {"A", "B"}
    assert all(row["answer_metrics"]["answer_em"] == 1 for row in system)
    assert all(row["online_total_seconds"] > 0 for row in system)
    with pytest.raises(ValueError, match="call limit"):
        measure_system(FixtureAdapter(), [{"id": "q", "question": "Capital?"}], [],
                       client=client, generation_config=config, max_calls=1, progress=None)


def test_candidate_cache_hash_and_common_evidence_contract(tmp_path):
    from rag_baselines.data import write_jsonl
    from rag_baselines.evaluation import evaluate_cached_quality
    from rag_baselines.store import DocumentStore
    from rag_baselines.models import Baselines
    path = tmp_path / "corpus.jsonl"
    passage = {"id": "7:0", "content": "Title. Paris", "source_spans": [], "wikipedia_id": "7", "title": "Title"}
    write_jsonl(path, [passage])
    store = DocumentStore(path)

    class CacheEngine:
        select_cached = Baselines.select_cached

        def __init__(self):
            self.root, self.store = tmp_path, store
            self.passages, self.id_to_index = store.passages, store.indices
            self.corpus_sha256, self.retriever_signature = store.sha256, "fixture-model"
            self.config = {"retrieve_k": 50, "evidence_k": 5}
            self.search_calls = 0

        def retrieve(self, question, method):
            self.search_calls += 1
            return {"question": question, "corpus_sha256": self.corpus_sha256,
                    "retriever_signature": self.retriever_signature, "retrieve_k": 50,
                    "candidates": [{"id": "7:0", "dense_score": 1.0}]}

        def _rerank(self, question, indices):
            assert indices == [0]
            return [2.0], {"nonpad": 5, "padded": 256}

    engine = CacheEngine()
    questions = [{"id": "q", "question": "Capital?"}]
    labels = [{"id": "q", "evidence_groups": [["7:0"]]}]
    first = evaluate_cached_quality(engine, questions, labels)
    second = evaluate_cached_quality(engine, questions, labels)
    assert engine.search_calls == 1
    assert [r["evidence"][0]["id"] for r in first] == ["7:0", "7:0"]
    assert all(r["seconds"]["retrieval_total"] is None for r in second)
    modified = {**first[0], "evidence": [{**passage, "content": "invented summary"}]}
    with pytest.raises(ValueError, match="original"):
        store.validate_evidence(modified)
    cache_path = next((tmp_path / "data/retrieval_cache").glob("*.jsonl"))
    cache_path.write_text(cache_path.read_text().replace('"dense_score":1.0', '"dense_score":9.0'))
    with pytest.raises(ValueError, match="integrity"):
        evaluate_cached_quality(engine, questions, labels)


def test_notebook_source_hash_ignores_execution_outputs(tmp_path):
    import nbformat
    from rag_baselines.evaluation import notebook_source_sha256
    path = tmp_path / "measure.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("RETRIEVE_K = 50")])
    nbformat.write(notebook, path)
    original = notebook_source_sha256(path)
    notebook.cells[0].execution_count = 1
    notebook.cells[0].outputs = [nbformat.v4.new_output("stream", name="stdout", text="measured result")]
    nbformat.write(notebook, path)
    assert notebook_source_sha256(path) == original
    notebook.cells[0].source = "RETRIEVE_K = 100"
    nbformat.write(notebook, path)
    assert notebook_source_sha256(path) != original
