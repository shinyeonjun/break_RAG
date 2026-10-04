import json
import copy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rag_baselines.laya_local import LayaDecisionClient


def test_pinned_laya_loader_reaches_checkpoint_acquisition_without_backend_module(monkeypatch):
    """Use the installed load + Agent constructor; stop before downloading weights."""
    config = json.loads((Path(__file__).resolve().parents[1] / "config/d_laya.json").read_text(encoding="utf-8"))
    requested = {}

    class CheckpointAcquisitionReached(Exception):
        pass

    def stop_download(repo, **kwargs):
        requested.update(repo=repo, **kwargs)
        raise CheckpointAcquisitionReached

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.snapshot_download", stop_download)
    with pytest.raises(CheckpointAcquisitionReached):
        LayaDecisionClient(config, device="cpu")
    assert requested["repo"] == config["model"]["repo"]
    assert requested["revision"] == config["model"]["revision"]
    assert requested["token"] is None


def make_laya_client(monkeypatch, body, *, backend="eager"):
    config = json.loads((Path(__file__).resolve().parents[1] / "config/d_laya.json").read_text(encoding="utf-8"))
    config["cuda_amp"] = None
    loaded = {}
    fake_agent = SimpleNamespace(
        device="cpu",
        backend=backend,
        revision=config["model"]["revision"],
        dtype=torch.float32,
        warmup=lambda: 0.01,
        tok=object(),
        cpu_fallback_count=0,
    )
    def predict_batch(states, questions, **kwargs):
        loaded["last_batch"] = {"states": states, "questions": questions, **kwargs}
        outputs = []
        for _ in states:
            result = copy.deepcopy(body)
            result["answers"] = {"decision": next(iter(result["answers"].values()))}
            outputs.append(result)
        return outputs
    fake_agent.predict_batch = predict_batch
    monkeypatch.setattr("rag_baselines.laya_local.audit_laya_input", lambda *a, **k: {
        "input_tokens": 10, "input_ids": [], "instruction_tokens_dropped": 0,
        "option_tokens_dropped": [0, 0], "state_tokens_dropped": 0})
    fake_laya = ModuleType("laya")

    def load(repo, *, device, revision, fast, compile):
        loaded.update(repo=repo, device=device, revision=revision, fast=fast, compile=compile)
        return fake_agent

    fake_laya.load = load
    monkeypatch.setitem(sys.modules, "laya", fake_laya)
    monkeypatch.setattr("rag_baselines.laya_local.version", lambda _: config["model"]["library_version"])
    monkeypatch.setattr("rag_baselines.laya_local.synchronize", lambda _: None)
    return LayaDecisionClient(config, device="cpu"), loaded


def test_local_decision_loads_pinned_model_and_records_valid_typed_response(monkeypatch):
    body = {
        "answers": {"branch": {"type": "noul", "noul": 0.8}},
        "usage": {"input_tokens": 10, "output_tokens": 1, "truncated": False,
                  "state_tokens_dropped": 0, "truncated_questions": []},
    }
    client, loaded = make_laya_client(monkeypatch, body)
    calls, records = [], []

    answers = client.decide(
        {"query": "Which branch?"},
        {"branch": {"type": "noul", "instructions": {"question": "Is branch relevant?", "branch": {"title": "Sales"}},
                    "criteria": {"true": "yes", "false": "no"}}},
        calls=calls, max_calls=1, stage="navigation", on_record=records.append,
    )

    assert answers == body["answers"]
    assert calls == records
    assert records[0]["status"] == "completed"
    assert records[0]["response"] == body
    assert records[0]["request"]["revision"] == client.revision
    assert {k: v for k, v in loaded.items() if k != "last_batch"} == {
        "repo": client.model, "device": "cpu", "revision": client.revision, "fast": False, "compile": False}
    batch = loaded["last_batch"]
    assert batch["states"][0]["decision_data"]["branch"]["title"] == "Sales"
    assert batch["questions"]["decision"]["instructions"] == "Is branch relevant?"
    assert records[0]["effective_inputs"][0]["question_id"] == "branch"
    assert client.runtime_info()["backend"] == "eager"


def test_local_decision_rejects_unexpected_runtime_backend(monkeypatch):
    with pytest.raises(RuntimeError, match="Laya selected compile; expected eager"):
        make_laya_client(monkeypatch, {}, backend="compile")


@pytest.mark.parametrize("bad_usage", [
    {"input_tokens": 10, "output_tokens": 0},
    {"input_tokens": True, "output_tokens": 0, "truncated": False, "state_tokens_dropped": 0, "truncated_questions": []},
    {"input_tokens": 10, "output_tokens": 0, "truncated": "false", "state_tokens_dropped": 0, "truncated_questions": []},
])
def test_missing_or_invalid_native_diagnostics_are_not_turned_into_no_truncation(monkeypatch, bad_usage):
    client, _ = make_laya_client(monkeypatch, {
        "answers": {"branch": {"type": "noul", "noul": .8}}, "usage": bad_usage})
    calls = []
    with pytest.raises(ValueError, match="per-row"):
        client.decide({"query": "Where?"}, {"branch": {"type": "noul", "instructions": "Relevant?",
                      "criteria": {"true": "yes", "false": "no"}}}, calls=calls, max_calls=1, stage="navigation")
    assert calls[0]["status"] == "failed"
    assert calls[0]["native_responses"][0]["response"]["usage"] == bad_usage


def test_a_temporary_cpu_fallback_is_rejected_even_when_device_is_restored(monkeypatch):
    body = {"answers": {"branch": {"type": "noul", "noul": .8}}, "usage": {
        "input_tokens": 10, "output_tokens": 0, "truncated": False,
        "state_tokens_dropped": 0, "truncated_questions": []}}
    client, _ = make_laya_client(monkeypatch, body)
    original = client.agent.predict_batch
    def fallback(*args, **kwargs):
        client.agent.cpu_fallback_count += 1
        return original(*args, **kwargs)
    client.agent.predict_batch = fallback
    calls = []
    with pytest.raises(RuntimeError, match="mixed-device"):
        client.decide({"query": "Where?"}, {"branch": {"type": "noul", "instructions": "Relevant?",
                      "criteria": {"true": "yes", "false": "no"}}}, calls=calls, max_calls=1, stage="navigation")
    assert calls[0]["cpu_fallbacks"] == 1 and calls[0]["response"]["answers"]


def test_local_decision_rejects_reported_input_truncation_and_keeps_raw_response(monkeypatch):
    body = {
        "answers": {"branch": {"type": "noul", "noul": 0.8}},
        "usage": {"input_tokens": 10, "output_tokens": 1, "truncated": True,
                  "state_tokens_dropped": 2, "truncated_questions": ["branch"]},
    }
    client, _ = make_laya_client(monkeypatch, body)
    calls, records = [], []

    with pytest.raises(ValueError, match="truncated the decision input"):
        client.decide(
            {"query": "Which branch?"},
            {"branch": {"type": "noul", "instructions": "Is branch relevant?", "criteria": {"true": "yes", "false": "no"}}},
            calls=calls, max_calls=1, stage="navigation", on_record=records.append,
        )

    assert calls == records
    assert records[0]["status"] == "rejected_truncation"
    assert records[0]["response"] == body
