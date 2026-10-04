"""Synchronous experiment receipts: exact model boundaries and HTTP bodies.

The API ledger counts physical HTTP attempts, never copies of response records.
Credentials and auth headers are excluded; redacted bodies retain both hashes.
Diagnostic capture inside a measurement is included in its observed wall time.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import math
import os
import shutil
import time
import traceback

import numpy as np

from .data import file_sha256, write_json


def query_scope(audit, phase, question, method, repeat=0):
    return audit.scope(phase=phase, method=method, question_id=question['id'],
                       question=question['question'], task=question.get('task'), repeat=repeat) if audit else nullcontext()


class RawLog:
    def __init__(self, root, *, manifest, prices, secrets=()):
        from .generation import usage_cost
        from .jev import call_usage
        usage_cost({'input_tokens': 0, 'output_tokens': 0}, prices.get('openai'))
        call_usage([], prices.get('jev'))
        self.root = Path(root).resolve()
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
        self.directory = self.root / 'results/raw' / f'{stamp}_{uuid4().hex[:8]}'
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / 'api_calls.jsonl').touch()
        self.prices = deepcopy(prices)
        self._secrets = {value for value in (*secrets, os.environ.get('OPENAI_API_KEY'), os.environ.get('TYPESAFE_API_KEY')) if value}
        self.current = {'phase': 'preparation', 'method': 'shared', 'scope_id': uuid4().hex}
        self.capture_seconds = 0.0
        self.pending = {}
        write_json(self.directory / 'manifest.json', {
            'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'experiment': self._redact(manifest), 'prices': self._redact(prices),
            'timing': 'capture within the retrieval/API interval is included in wall time; final receipts are written afterward',
            'accounting': 'api_calls.jsonl counts each physical HTTP attempt once; cached/historical response copies are not new calls',
            'bodies': 'httpx request content and decoded HTTP response body; no auth headers; secret replacement is marked',
            'local_models': 'token IDs/masks, encoder CLS vectors, full dense scores and reranker logits; not hidden-layer activations',
        })

    def _path(self):
        context = self.current
        # User/dataset IDs never become Windows filenames.
        scope = context['scope_id']
        path = self.directory / context['phase'] / context['method'] / scope
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _redact(self, value):
        if isinstance(value, dict):
            return {k: '[REDACTED]' if k.casefold() in {'authorization', 'api_key', 'access_token', 'cookie', 'set-cookie'}
                    else self._redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(v) for v in value]
        if isinstance(value, str):
            for secret in self._secrets:
                value = value.replace(secret, '[REDACTED]')
        if isinstance(value, float) and not math.isfinite(value):
            return {'nonfinite': str(value)}
        return value

    @staticmethod
    def _append(path, record):
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def event(self, stage, kind, payload):
        started = time.perf_counter()
        path = self._path() / 'events.jsonl'
        row = {'event_id': uuid4().hex, 'utc': datetime.now(timezone.utc).isoformat(),
               **self.current, 'stage': stage, 'kind': kind, 'payload': self._redact(payload)}
        self._append(path, row)
        self._append(self.directory / 'events.jsonl', {
            key: row[key] for key in ('event_id', 'utc', 'phase', 'method', 'scope_id', 'stage', 'kind')
        } | {'events_path': str(path.relative_to(self.directory))})
        self.capture_seconds += time.perf_counter() - started
        return row['event_id']

    def reference(self):
        return {'session': str(self.directory.relative_to(self.root)),
                'scope_id': self.current['scope_id'],
                'events_path': str((self._path() / 'events.jsonl').relative_to(self.root))}

    @contextmanager
    def scope(self, **context):
        for key in ('phase', 'method'):
            value = context.get(key, self.current[key])
            if not isinstance(value, str) or not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789-' for c in value):
                raise ValueError('Raw-log phase/method must be simple directory labels.')
        previous = self.current
        self.current = {**previous, **context, 'scope_id': uuid4().hex}
        journals = self.root / f"results/{self.current['method'].lower()}_requests"
        before = set(journals.glob('*.meta.json')) if self.current['method'] in ('C', 'D') else None
        try:
            self.event('scope', 'started', {})
            yield self.reference()
        except BaseException as error:
            self._finish_pending(type(error).__name__)
            self.event('scope', 'failed', {'error_type': type(error).__name__, 'message': str(error),
                                         'traceback': traceback.format_exc()})
            raise
        else:
            self.event('scope', 'completed', {})
        finally:
            try:
                if before is not None:
                    # Copy only journals created in this scope, including partial failure state.
                    for meta in set(journals.glob('*.meta.json')) - before:
                        base = meta.name.removesuffix('.meta.json')
                        for suffix in ('.meta.json', '.jsonl', '.failure.json'):
                            path = meta.parent / f'{base}{suffix}'
                            if path.exists():
                                self.snapshot(path)
            finally:
                self.current = previous

    def snapshot(self, path):
        started = time.perf_counter()
        path = Path(path).resolve()
        if not path.is_relative_to(self.root) or path.name.casefold().startswith('.env'):
            raise ValueError('Snapshot must be a named experiment artifact inside the workspace; never .env.')
        source_sha = file_sha256(path)
        destination = self.directory / 'sources' / f'{source_sha[:16]}_{path.name}'
        destination.parent.mkdir(exist_ok=True)
        saved = None
        if path.suffix in {'.json', '.jsonl', '.py', '.ipynb', '.md', '.csv'}:
            saved = path.read_bytes()
            for secret in self._secrets:
                saved = saved.replace(secret.encode(), b'[REDACTED]')
        expected_sha = hashlib.sha256(saved).hexdigest() if saved is not None else source_sha
        if not destination.exists():
            if saved is not None:
                destination.write_bytes(saved)
            else:
                shutil.copyfile(path, destination)
        saved_sha = file_sha256(destination)
        redacted = saved_sha != source_sha
        if saved_sha != expected_sha or file_sha256(path) != source_sha:
            raise ValueError('Artifact changed while taking its raw-log snapshot.')
        self.capture_seconds += time.perf_counter() - started
        record = {'source': str(path.relative_to(self.root)), 'snapshot': str(destination.relative_to(self.directory)),
                  'sha256': source_sha, 'saved_sha256': saved_sha, 'redacted': redacted, 'bytes': destination.stat().st_size}
        self.event('preparation', 'artifact', record)
        return record

    def result(self, stage, payload, *, label=None):
        started = time.perf_counter()
        path = self._path() / f'{stage}.json'
        write_json(path, self._redact(payload))
        receipt = {'path': str(path.relative_to(self.directory)), 'sha256': file_sha256(path)}
        self.capture_seconds += time.perf_counter() - started
        self.event(stage, 'result', receipt)
        if label is not None:
            self.event('offline_evaluation', 'label', label)

    def _arrays(self, name, arrays):
        started = time.perf_counter()
        values = {k: v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v) for k, v in arrays.items()}
        if any(v.dtype.hasobject for v in values.values()):
            raise ValueError('Raw tensor files must not require pickle.')
        path = self._path() / f'{name}.npz'
        np.savez_compressed(path, **values)
        record = {'path': str(path.relative_to(self.directory)), 'sha256': file_sha256(path),
                  'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in values.items()}}
        self.capture_seconds += time.perf_counter() - started
        return record

    def model_start(self, role, inputs, metadata):
        call_id = uuid4().hex
        arrays = self._arrays(f'{call_id}_input', inputs)
        self.event(role, 'model_request', {'call_id': call_id, 'inputs': arrays, **metadata})
        return call_id

    def model_end(self, role, call_id, outputs, *, compute_seconds, tokens=None):
        arrays = self._arrays(f'{call_id}_output', outputs)
        self.event(role, 'model_response', {'call_id': call_id, 'outputs': arrays,
                   'compute_seconds': compute_seconds, 'tokens': tokens})

    def _body(self, name, content):
        started = time.perf_counter()
        saved = content
        for secret in self._secrets:
            saved = saved.replace(secret.encode(), b'[REDACTED]')
        path = self._path() / name
        path.write_bytes(saved)
        record = {'path': str(path.relative_to(self.directory)), 'source_bytes': len(content),
                  'source_sha256': hashlib.sha256(content).hexdigest(), 'saved_sha256': hashlib.sha256(saved).hexdigest(),
                  'redacted': content != saved}
        self.capture_seconds += time.perf_counter() - started
        return record

    @property
    def http_hooks(self):
        return {'request': [self.http_request], 'response': [self.http_response]}

    def http_request(self, request):
        call_id = uuid4().hex
        content = request.content
        body = json.loads(content)
        provider = 'jev' if body.get('model', '').startswith('jev-') else 'openai'
        kinds = {q['type'] for q in body.get('questions', {}).values()}
        role = 'generation' if provider == 'openai' else ('placement' if kinds == {'choice'} else
            'scoring' if kinds == {'score'} else 'coverage' if set(body.get('questions', {})) == {'complete'} else 'navigation')
        receipt = self._body(f'{call_id}_request.json', content)
        row = {'call_id': call_id, 'provider': provider, 'model': body.get('model'), 'role': role,
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'request': receipt, 'endpoint': {'scheme': request.url.scheme, 'host': request.url.host, 'path': request.url.path}}
        self.event(role, 'http_request', row)
        self.pending[call_id] = {**row, 'context': dict(self.current), 'started': time.perf_counter()}
        request.extensions['raw_log_call_id'] = call_id

    def http_response(self, response):
        call_id = response.request.extensions['raw_log_call_id']
        attempt = self.pending[call_id]
        # Reading a normal nonstreaming SDK response is network work, not log IO.
        content = response.read()
        elapsed = time.perf_counter() - attempt['started']
        receipt = self._body(f'{call_id}_response.json', content)
        try:
            body = json.loads(content)
        except (ValueError, UnicodeDecodeError):
            body = None
        headers = {k: v for k, v in response.headers.items() if k in {'x-request-id', 'request-id', 'retry-after', 'content-type'}}
        self.event(attempt['role'], 'http_response', {'call_id': call_id, 'http_status': response.status_code,
                   'response': receipt, 'safe_headers': headers, 'http_seconds': elapsed})
        self._ledger(attempt, body, receipt, elapsed, response.status_code)
        del self.pending[call_id]

    def _ledger(self, attempt, body, response, seconds, status, error_type=None):
        from .generation import usage_cost
        from .jev import call_usage
        usage = body.get('usage') if isinstance(body, dict) else None
        try:
            if attempt['provider'] == 'jev':
                _, cost = call_usage([{'response': body}], self.prices.get('jev'))
            else:
                cost = usage_cost(usage, self.prices.get('openai'))
        except OverflowError:
            cost = None  # Malformed provider counts must not discard an already-received body.
        if cost is not None and not math.isfinite(cost):
            cost = None
        row = {**attempt['context'], 'call_id': attempt['call_id'], 'provider': attempt['provider'],
               'role': attempt['role'], 'model': attempt['model'], 'http_status': status, 'error_type': error_type,
               'started_utc': attempt['started_utc'], 'recorded_utc': datetime.now(timezone.utc).isoformat(),
               'returned_model': body.get('model') if isinstance(body, dict) else None,
               'http_seconds': seconds, 'api_usage': usage, 'api_cost_usd': cost,
               'request': attempt['request'], 'response': response}
        started = time.perf_counter()
        self._append(self.directory / 'api_calls.jsonl', self._redact(row))
        self.capture_seconds += time.perf_counter() - started

    def _finish_pending(self, error_type):
        for call_id, attempt in list(self.pending.items()):
            if attempt['context']['scope_id'] != self.current['scope_id']:
                continue
            seconds = time.perf_counter() - attempt['started']
            self.event(attempt['role'], 'http_failed', {'call_id': call_id, 'error_type': error_type,
                       'response_received': False, 'http_seconds': seconds})
            self._ledger(attempt, None, None, seconds, None, error_type)
            del self.pending[call_id]
