import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import torch
from openai import OpenAI, APIConnectionError, RateLimitError
from transformers import BatchEncoding

from rag_baselines.data import read_json, read_jsonl, write_json, write_jsonl
from rag_baselines.evaluation import (evaluate_cached_quality, measure_retrieval,
                                     measure_system)
from rag_baselines.models import Baselines
from rag_baselines.raw_logging import RawLog, query_scope
from rag_baselines.store import DocumentStore
from test_c_retrieval import setup, make_engine, QUERY


PRICES = {'openai': {'input_usd_per_million': .1, 'cached_input_usd_per_million': .01,
                     'output_usd_per_million': .5},
          'jev': {'input_usd_per_million': .042, 'output_usd_per_million': 0}}
QUESTION = {'id': 'q:with/windows?chars', 'question': 'Capital?', 'task': 'nq'}
LABEL = {'id': QUESTION['id'], 'answers': ['Paris', 'OFFLINE_ONLY_ANSWER'], 'evidence_groups': [['7:0']]}
GENERATION = {'model': 'gpt-6-luna', 'max_output_tokens': 256, 'encoding_name': 'o200k_base',
              'prices': PRICES['openai'], 'reasoning_effort': 'none', 'temperature': 0, 'service_tier': 'default'}


def log_at(root):
    return RawLog(root, manifest={'fixture': True}, prices=PRICES, secrets=('fixture-key',))


def scope_events(log, reference):
    return read_jsonl(log.root / reference['events_path'])


def test_exact_bodies_saved_before_request_and_one_ledger_entry_per_attempt(tmp_path):
    log = log_at(tmp_path)
    request_bytes = b'{ "model":"gpt-6-luna", "input":"data" }'
    response_bytes = b'{ "usage":{"input_tokens":20,"output_tokens":3}, "custom": [1,2] }'

    def handler(request):
        # Evidence of a request survives a crash or timeout inside the transport.
        receipt = scope_events(log, log.reference())[-1]['payload']['request']
        assert (log.directory / receipt['path']).read_bytes() == request.content == request_bytes
        return httpx.Response(200, content=response_bytes, headers={'x-request-id': 'req-1', 'set-cookie': 'fixture-key'})

    with query_scope(log, 'system', QUESTION, 'A', 2) as reference:
        with httpx.Client(transport=httpx.MockTransport(handler), event_hooks=log.http_hooks) as client:
            client.post('https://example.test/v1/responses', content=request_bytes,
                        headers={'Authorization': 'Bearer fixture-key'})
        log.result('copied_response', json.loads(response_bytes))
    ledger = read_jsonl(log.directory / 'api_calls.jsonl')
    assert len(ledger) == 1
    row = ledger[0]
    assert (row['question_id'], row['task'], row['phase'], row['method'], row['repeat']) == (QUESTION['id'], 'nq', 'system', 'A', 2)
    assert (log.directory / row['response']['path']).read_bytes() == response_bytes
    assert row['response']['source_sha256'] == hashlib.sha256(response_bytes).hexdigest()
    assert row['api_cost_usd'] == pytest.approx(.0000035)
    events = scope_events(log, reference)
    assert events[-1]['kind'] == 'completed'
    response_event = next(e for e in events if e['kind'] == 'http_response')
    assert response_event['payload']['safe_headers'] == {'x-request-id': 'req-1'}
    assert 'fixture-key' not in '\n'.join(p.read_text(encoding='utf-8') for p in log.directory.rglob('*.json*'))


@pytest.mark.parametrize('network_failure', [False, True])
def test_failed_llm_keeps_retrieved_evidence_request_and_unknown_cost(tmp_path, network_failure):
    path = tmp_path / 'corpus.jsonl'
    passage = {'id': '7:0', 'content': 'Paris.', 'title': 'Capital', 'wikipedia_id': '7', 'source_spans': []}
    write_jsonl(path, [passage])
    store = DocumentStore(path)
    engine = SimpleNamespace(store=store, retrieve=lambda q, m: {
        'method': m, 'question': q, 'evidence': [passage], 'candidates': [{'id': '7:0'}],
        'corpus_sha256': store.sha256,
        'seconds': {'retrieval_total': .001}, 'api_usage': None, 'api_cost_usd': None})
    log = log_at(tmp_path)

    def handler(request):
        assert 'OFFLINE_ONLY_ANSWER' not in request.content.decode()
        if network_failure:
            raise httpx.ConnectError('fixture-key should never appear in exception logs', request=request)
        return httpx.Response(429, json={'error': {'message': 'echo fixture-key', 'type': 'rate_limit'}})

    with httpx.Client(transport=httpx.MockTransport(handler), event_hooks=log.http_hooks) as http:
        with OpenAI(api_key='fixture-key', max_retries=0, http_client=http) as client:
            with pytest.raises(APIConnectionError if network_failure else RateLimitError):
                measure_system(engine, [QUESTION], [LABEL], client=client, generation_config=GENERATION,
                               methods=('A',), max_calls=1, audit=log, progress=None)
    row = read_jsonl(log.directory / 'api_calls.jsonl')[0]
    assert row['http_status'] == (None if network_failure else 429)
    assert row['api_cost_usd'] is None
    request = log.directory / row['request']['path']
    assert json.loads(request.read_bytes())['input'].endswith('[7:0] Paris.')
    retrieved = read_json(request.parent / 'retrieval.json')
    assert retrieved['evidence'] == [passage]
    assert 'metrics' not in retrieved  # Generation failure does not invent an evaluated result.
    assert not (request.parent / 'system.json').exists()
    assert any(e['kind'] == 'failed' for e in read_jsonl(request.parent / 'events.jsonl'))
    assert 'fixture-key' not in '\n'.join(p.read_text(encoding='utf-8') for p in log.directory.rglob('*.json*'))
    assert not log.pending


@pytest.mark.parametrize('body', [{'usage': {'input_tokens': -1, 'output_tokens': 2}},
                                  {'usage': {'input_tokens': 1, 'output_tokens': float('nan')}},
                                  {'usage': {'input_tokens': 10**1000, 'output_tokens': 1}},
                                  {'usage': []}])
def test_invalid_provider_usage_stays_unknown_without_losing_raw_body(tmp_path, body):
    log = log_at(tmp_path)
    with query_scope(log, 'system', QUESTION, 'A'):
        with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps(body).encode())),
                          event_hooks=log.http_hooks) as client:
            client.post('https://example.test/responses', json={'model': 'gpt-6-luna'})
    row = read_jsonl(log.directory / 'api_calls.jsonl')[0]
    assert row['api_cost_usd'] is None
    assert (log.directory / row['response']['path']).read_bytes() == json.dumps(body).encode()


def test_snapshot_integrity_secret_redaction_and_no_env_or_outside_workspace(tmp_path):
    log = log_at(tmp_path)
    path = tmp_path / 'source.json'
    write_json(path, {'echo': 'fixture-key'})
    receipt = log.snapshot(path)
    assert receipt['redacted']
    assert receipt['sha256'] != receipt['saved_sha256']
    assert 'fixture-key' not in (log.directory / receipt['snapshot']).read_text()
    assert log.snapshot(path) == receipt
    (log.directory / receipt['snapshot']).write_text('tampered')
    with pytest.raises(ValueError, match='changed'):
        log.snapshot(path)
    for forbidden in (tmp_path / '.env', tmp_path / '.ENV', tmp_path.parent / 'outside.json'):
        with pytest.raises(ValueError, match='never .env'):
            log.snapshot(forbidden)
    with pytest.raises(ValueError, match='directory labels'):
        with log.scope(phase='../bad'):
            pass


def test_keyboard_interrupt_preserves_pending_attempt_and_failure_context(tmp_path):
    log = log_at(tmp_path)

    def handler(request):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        with query_scope(log, 'latency', QUESTION, 'C', 1) as reference:
            with httpx.Client(transport=httpx.MockTransport(handler), event_hooks=log.http_hooks) as client:
                client.post('https://example.test/systemone', json={'model': 'jev-1.13.0', 'questions': {}})
    row = read_jsonl(log.directory / 'api_calls.jsonl')[0]
    assert row['error_type'] == 'KeyboardInterrupt' and row['repeat'] == 1
    assert row['api_cost_usd'] is None
    assert any(e['kind'] == 'failed' for e in scope_events(log, reference))
    assert log.current['phase'] == 'preparation' and not log.pending


def test_successful_system_records_evaluation_separately_and_persistent_links(tmp_path):
    path = tmp_path / 'corpus.jsonl'
    passage = {'id': '7:0', 'content': 'Paris.', 'title': 'Capital', 'wikipedia_id': '7', 'source_spans': []}
    write_jsonl(path, [passage])
    store = DocumentStore(path)
    engine = SimpleNamespace(store=store, retrieve=lambda q, m: {
        'method': m, 'question': q, 'corpus_sha256': store.sha256,
        'evidence': [passage], 'candidates': [{'id': '7:0'}],
        'seconds': {'retrieval_total': 0}, 'api_usage': None, 'api_cost_usd': None})
    log = log_at(tmp_path)

    def handler(request):
        assert 'OFFLINE_ONLY_ANSWER' not in request.content.decode()
        return httpx.Response(200, json={
            'id': 'resp_fixture', 'object': 'response', 'created_at': 1, 'status': 'completed', 'model': 'gpt-6-luna',
            'output': [{'id': 'msg_fixture', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'annotations': [], 'text': '{"answer":"Paris","citations":["7:0"]}'}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 5}})

    with httpx.Client(transport=httpx.MockTransport(handler), event_hooks=log.http_hooks) as http:
        with OpenAI(api_key='fixture-key', max_retries=0, http_client=http) as client:
            results = measure_system(engine, [QUESTION], [LABEL], client=client, generation_config=GENERATION,
                                     methods=('A', 'B'), repeats=2, max_calls=4, audit=log, progress=None)
    assert len({r['raw_log']['scope_id'] for r in results}) == 4
    ledger = read_jsonl(log.directory / 'api_calls.jsonl')
    assert len(ledger) == 4 and {r['repeat'] for r in ledger} == {0, 1}
    for r in results:
        assert r['online_api_cost_usd'] == pytest.approx(.0000125)
        assert r['answer_metrics']['answer_em'] == 1
        assert 0 < r['diagnostic_capture_seconds'] < r['online_total_seconds']
        events = scope_events(log, r['raw_log'])
        assert any(e['stage'] == 'offline_evaluation' and e['kind'] == 'label' for e in events)
        saved = read_json((log.root / r['raw_log']['events_path']).parent / 'system.json')
        assert saved == r


class TokenizerFixture:
    def __call__(self, texts, pairs=None, return_tensors=None, truncation=False, max_length=None, **kwargs):
        ids = [[len(text), len(pairs[i]) if pairs else 7, 8, 9] for i, text in enumerate(texts)]
        if truncation:
            ids = [row[:max_length] for row in ids]
        if not return_tensors:
            return {'input_ids': ids}
        return BatchEncoding({'input_ids': torch.tensor(ids), 'attention_mask': torch.ones(len(ids), len(ids[0]), dtype=torch.int64)})


def local_engine(root, log):
    engine = Baselines.__new__(Baselines)
    engine.root, engine.raw_log, engine.device, engine.dtype = root, log, torch.device('cpu'), torch.float32
    engine.config = {'retrieve_k': 2, 'evidence_k': 1,
                     'retriever': {'max_length': 3, 'batch_size': 2},
                     'reranker': {'max_length': 3, 'batch_size': 2, 'padding': 'max_length', 'truncation': 'only_second'}}
    engine.file_hashes = {'retriever': {'weights': 'fixture'}, 'reranker': {'weights': 'fixture'}}
    engine.retriever_signature = 'fixture-retriever'
    engine.tokenizers = {role: TokenizerFixture() for role in ('retriever', 'reranker')}

    class Encoder:
        config = SimpleNamespace(hidden_size=2)

        def __call__(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=torch.stack((input_ids.float(), torch.ones_like(input_ids).float()), dim=2))

    engine.models = {'retriever': Encoder(), 'reranker': lambda input_ids, attention_mask: SimpleNamespace(logits=input_ids[:, 1:2].float())}
    return engine


def test_local_token_ids_vectors_dense_scores_rerank_logits_and_cache_parity(tmp_path):
    path = tmp_path / 'corpus.jsonl'
    passages = [{'id': f'{i}:0', 'content': text, 'title': 'Fixture', 'wikipedia_id': str(i), 'source_spans': []}
                for i, text in enumerate(('short', 'a much longer text', 'middle text'))]
    write_jsonl(path, passages)
    log = log_at(tmp_path)
    engine = local_engine(tmp_path, log)
    engine.build_index(path, progress=None)
    labels = [{**LABEL, 'evidence_groups': [['1:0']]}]
    results = measure_retrieval(engine, [QUESTION], labels, methods=('A', 'B'), audit=log, progress=None)
    assert [p['id'] for p in results[0]['candidates']] == [p['id'] for p in results[1]['candidates']]
    for result in results:
        events = scope_events(log, result['raw_log'])
        inputs = [e['payload'] for e in events if e['kind'] == 'model_request']
        outputs = [e['payload'] for e in events if e['kind'] == 'model_response']
        encoder_inputs = np.load(log.directory / inputs[0]['inputs']['path'], allow_pickle=False)
        assert encoder_inputs['input_ids'].tolist() == [[len(QUESTION['question']), 7, 8]]
        assert encoder_inputs['attention_mask'].tolist() == [[1, 1, 1]]
        encoder_output = np.load(log.directory / outputs[0]['outputs']['path'], allow_pickle=False)['cls_embeddings']
        scores = np.load(log.directory / outputs[1]['outputs']['path'], allow_pickle=False)
        np.testing.assert_allclose(scores['all_scores'], (encoder_output @ engine.embeddings.numpy().T)[0])
        assert scores['top_indices'].tolist() == [1, 2]
        if result['method'] == 'B':
            logits = np.load(log.directory / outputs[2]['outputs']['path'], allow_pickle=False)['logits']
            assert logits.tolist() == [len(passages[1]['content']), len(passages[2]['content'])]
        engine.raw_log = None
        unlogged = engine.retrieve(QUESTION['question'], result['method'])
        engine.raw_log = log
        assert unlogged['candidates'] == result['candidates']
        assert unlogged['evidence'] == result['evidence']
        assert unlogged['model_tokens'] == result['model_tokens']
        assert result['diagnostic_capture_seconds'] > 0
    engine.build_index(path, progress=None)
    assert engine.index_info['cache_hit']
    first = evaluate_cached_quality(engine, [QUESTION], labels, audit=log)
    second = evaluate_cached_quality(engine, [QUESTION], labels, audit=log)
    assert first[0]['candidates'] == second[0]['candidates']
    for result in second:
        events = scope_events(log, result['raw_log'])
        assert any(e['kind'] == 'cache_reused' for e in events)
        assert not any(e['stage'] in ('retriever', 'dense_search') for e in events)


@pytest.mark.parametrize('failed', [False, True])
def test_c_cache_and_failure_journals_do_not_duplicate_physical_calls(setup, failed):
    engine, captured = make_engine(setup, fail_call=2 if failed else None)
    log = log_at(engine.root)
    engine.client.http.event_hooks = log.http_hooks
    question = {'id': 'multi', 'question': QUERY, 'task': 'hotpotqa'}
    if failed:
        with pytest.raises(RuntimeError):
            with query_scope(log, 'quality', question, 'C'):
                engine.retrieve_cached_quality(QUERY)
    else:
        with query_scope(log, 'quality', question, 'C'):
            fresh = engine.retrieve_cached_quality(QUERY)
            log.result('retrieval', fresh)
    count = len(read_jsonl(log.directory / 'api_calls.jsonl'))
    assert count == len(captured)
    if failed:
        assert len(list((log.directory / 'sources').glob('*.failure.json'))) == 1
        failure = read_json(next((log.directory / 'sources').glob('*.failure.json')))
        assert failure['events']
    else:
        with query_scope(log, 'quality', question, 'C'):
            cached = engine.retrieve_cached_quality(QUERY)
            log.result('retrieval', cached)
        assert cached['decision_cache_hit'] and cached['trace']['original_calls']
        assert cached['api_cost_usd'] == 0
        assert len(read_jsonl(log.directory / 'api_calls.jsonl')) == count
        assert {r['role'] for r in read_jsonl(log.directory / 'api_calls.jsonl')} == {'navigation', 'scoring', 'coverage'}
    assert 'fixture-key' not in '\n'.join(p.read_text(encoding='utf-8') for p in log.directory.rglob('*.json*'))
    engine.client.http.close()


@pytest.mark.parametrize('failed', [False, True])
def test_local_d_journals_are_snapshotted_even_on_failure_without_api_ledger_calls(tmp_path, failed):
    log = log_at(tmp_path)
    journals = tmp_path/'results/d_requests'
    journals.mkdir(parents=True)
    def work():
        with query_scope(log, 'system', QUESTION, 'D'):
            write_json(journals/'local.meta.json', {'provider':'laya-local'})
            (journals/'local.jsonl').write_text('{"state":"fixture-key","input_ids":[1,2,3]}\n',encoding='utf-8')
            if failed:
                write_json(journals/'local.failure.json', {'error_type':'ValueError'})
                raise ValueError('invalid input')
    if failed:
        with pytest.raises(ValueError): work()
    else:
        work()
    sources = list((log.directory/'sources').glob('*local*'))
    assert len(sources) == (3 if failed else 2)
    assert not read_jsonl(log.directory/'api_calls.jsonl')
    assert 'fixture-key' not in '\n'.join(p.read_text(encoding='utf-8') for p in sources)
