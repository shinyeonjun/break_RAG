"""Notebook setup must replace project definitions cached by an older kernel."""
import importlib
import inspect
import json
import os
from pathlib import Path
from queue import Empty
import sys
import tempfile
import time

import dotenv
from jupyter_client import KernelManager
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('notebook', ['05_measure_abc.ipynb', '05_measure_abcd.ipynb'])
def test_notebook_setup_refreshes_cached_retrieval_contracts(monkeypatch, notebook):
    models = importlib.import_module('rag_baselines.models')
    evaluation = importlib.import_module('rag_baselines.evaluation')

    class LegacyBaselines:
        def __init__(self, root, device='auto', retrieve_k=None, evidence_k=None):
            pass

    def legacy_quality(engine, questions, labels):
        return []

    # Simulate an already-running kernel that imported the pre-logging versions.
    monkeypatch.setattr(models, 'Baselines', LegacyBaselines)
    monkeypatch.setattr(evaluation, 'evaluate_comparative_quality', legacy_quality)
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: False)
    monkeypatch.chdir(ROOT)
    notebook_data = json.loads((ROOT/'notebook'/notebook).read_text(encoding='utf-8'))
    setup = next(cell['source'] for cell in notebook_data['cells']
                 if cell['cell_type'] == 'code' and 'from rag_baselines.models import Baselines' in ''.join(cell['source']))
    namespace = {}
    exec(compile(''.join(setup), notebook, 'exec'), namespace)
    # Exercise the exact rejected call signature without loading weights or calling APIs.
    inspect.signature(namespace['Baselines']).bind(ROOT, raw_log=None)
    inspect.signature(namespace['evaluate_comparative_quality']).bind(None, [], [], audit=None)
    assert Path(importlib.import_module('rag_baselines.models').__file__).resolve() == ROOT/'rag_baselines/models.py'


@pytest.mark.parametrize('notebook', ['05_measure_abc.ipynb', '05_measure_abcd.ipynb'])
def test_setup_survives_real_jupyter_kernel(notebook):
    """Catch Windows native DLL crashes missed by in-process exec tests."""
    data = json.loads((ROOT/'notebook'/notebook).read_text(encoding='utf-8'))
    setup = ''.join(next(cell['source'] for cell in data['cells'] if cell['cell_type'] == 'code'))
    environment = dict(os.environ, OPENAI_API_KEY='', TYPESAFE_API_KEY='',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-X', 'faulthandler', '-m',
                                'ipykernel_launcher', '-f', '{connection_file}']
    with tempfile.TemporaryDirectory(prefix='decision-llm-kernel-') as temporary:
        native_log_path = Path(temporary)/'native-stderr.log'
        environment['IPYTHONDIR'] = str(Path(temporary)/'ipython')
        with native_log_path.open('wb') as native_stderr:
            _run_setup_in_kernel(manager, environment, native_stderr, native_log_path, setup)


@pytest.mark.skipif(os.environ.get('DECISION_LLM_TEST_LAYA_RUNTIME') != '1',
                    reason='Opt in to the cached real Laya GPU smoke test.')
def test_laya_runtime_survives_real_jupyter_kernel():
    """Validate the installed loader, CUDA warmup and typed decisions without APIs."""
    data = json.loads((ROOT/'notebook/05_measure_abcd.ipynb').read_text(encoding='utf-8'))
    setup = ''.join(next(cell['source'] for cell in data['cells'] if cell['cell_type'] == 'code'))
    smoke = """
config = read_json(ROOT/'config/d_laya.json')
client = LayaDecisionClient(config, device='cuda')
assert client.actual_backend == config['backend'] == 'eager'
assert client.actual_device.type == 'cuda'
assert client.agent.revision == config['model']['revision']
assert client.dtype == 'torch.float16'
questions = {
    'branch': {'type': 'noul', 'instructions': 'Does the text state an annual leave entitlement?',
               'criteria': {'true': 'The entitlement is stated.', 'false': 'It is not stated.'}},
    'passage': {'type': 'score', 'instructions': 'How useful is the text for answering the query?',
                'criteria': ['Unrelated.', 'Related topic only.', 'Partial evidence.', 'Direct evidence.']},
}
calls = []
answers = client.decide({'query': 'How many annual leave days do employees receive?',
                        'text': 'Employees receive 20 days of annual leave per year.'},
                       questions, calls=calls, max_calls=1, stage='kernel_smoke')
assert set(answers) == {'branch', 'passage'}
assert calls[0]['status'] == 'completed'
assert calls[0]['response']['usage']['truncated'] is False
assert calls[0]['response']['usage']['state_tokens_dropped'] == 0
assert calls[0]['response']['usage']['truncated_questions'] == []
# Exercise the real tokenizer's truncation report using a small per-call budget.
client.max_len, client.head_max_len = 128, 64
truncated_calls = []
try:
    client.decide('annual leave ' * 300, {'branch': questions['branch']},
                  calls=truncated_calls, max_calls=1, stage='kernel_truncation_smoke')
except ValueError as error:
    assert 'truncated the decision input' in str(error)
else:
    raise AssertionError('A truncated real model input was accepted')
assert truncated_calls[0]['status'] == 'rejected_truncation'
assert truncated_calls[0]['response']['usage']['state_tokens_dropped'] > 0
"""
    environment = dict(os.environ, OPENAI_API_KEY='', TYPESAFE_API_KEY='',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-X', 'faulthandler', '-m',
                                'ipykernel_launcher', '-f', '{connection_file}']
    with tempfile.TemporaryDirectory(prefix='decision-llm-laya-kernel-') as temporary:
        native_log_path = Path(temporary)/'native-stderr.log'
        environment['IPYTHONDIR'] = str(Path(temporary)/'ipython')
        with native_log_path.open('wb') as native_stderr:
            _run_setup_in_kernel(manager, environment, native_stderr, native_log_path,
                                 setup, extra_smoke=smoke, execution_timeout=60)


def _run_setup_in_kernel(manager, environment, native_stderr, native_log_path, setup,
                         *, extra_smoke='', execution_timeout=20):
    client = None
    try:
        manager.start_kernel(cwd=str(ROOT/'notebook'), env=environment, stderr=native_stderr)
        client = manager.blocking_client()
        client.start_channels()
        client.wait_for_ready(timeout=15)
        # Repeat setup in one kernel to preserve the stale-definition reload contract.
        smoke = ("import pyarrow as pa; import torch\n"
                 "assert pa.table({'a': [1, 2]}).to_pydict() == {'a': [1, 2]}\n"
                 "assert pd.DataFrame({'a': [1, 2]}).shape == (2, 1)\n"
                 "assert torch.ones(2).sum().item() == 2\n")
        smoke += extra_smoke
        for source in (setup, setup, smoke):
            message_id = client.execute(source)
            deadline = time.monotonic() + execution_timeout
            idle = False
            while time.monotonic() < deadline:
                # Read through a separate handle; never move the kernel writer's file cursor.
                native_log = native_log_path.read_text(encoding='utf-8', errors='replace')
                assert 'Windows fatal exception' not in native_log, 'Native Jupyter kernel access violation'
                assert 'Fatal Python error' not in native_log, 'Fatal Jupyter kernel error'
                assert manager.is_alive(), 'Jupyter kernel process died'
                try:
                    message = client.get_iopub_msg(timeout=0.2)
                except Empty:
                    continue
                if message.get('parent_header', {}).get('msg_id') != message_id:
                    continue
                assert message['msg_type'] != 'error', '\n'.join(message['content'].get('traceback', []))
                if message['msg_type'] == 'status' and message['content']['execution_state'] == 'idle':
                    idle = True
                    break
            assert idle, f'Jupyter setup did not finish within {execution_timeout} seconds'
        native_log = native_log_path.read_text(encoding='utf-8', errors='replace')
        assert 'Windows fatal exception' not in native_log, 'Native Jupyter kernel access violation'
        assert 'Fatal Python error' not in native_log, 'Fatal Jupyter kernel error'
        assert manager.is_alive(), 'Jupyter kernel process died after setup'
    finally:
        if client is not None:
            client.stop_channels()
        manager.shutdown_kernel(now=True)
