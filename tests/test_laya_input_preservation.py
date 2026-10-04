"""Check the installed SDK's real packing without loading model weights."""
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from rag_baselines.laya_local import prepare_laya_inputs, audit_laya_input


@pytest.fixture(scope="module")
def tokenizer():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/d_laya.json").read_text(encoding="utf-8"))
    # Local, immutable tokenizer acquired together with the experiment checkpoint.
    snapshot = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual/snapshots"
    return AutoTokenizer.from_pretrained(snapshot / config["model"]["revision"] / "tokenizer", local_files_only=True)


def test_long_branch_data_survives_the_actual_laya_head_budget(tokenizer):
    branch = {"title": "Sales", "cards": ["monthly revenue " * 80, "TARGET_END September revenue 120"]}
    original = {"branch-id": {"type": "noul", "instructions": {
        "question": "Does branch contain evidence for any part of query?", "branch": branch},
        "criteria": {"true": "Needed evidence", "false": "Unrelated"}}}
    rows = prepare_laya_inputs({"query": "September revenue?"}, original)
    assert rows[0]["state"]["decision_data"]["branch"] == branch
    assert "TARGET_END" not in rows[0]["question"]["instructions"]
    audit = audit_laya_input(tokenizer, rows[0]["state"], rows[0]["question"], max_len=8192, head_max_len=256)
    assert audit["instruction_tokens_dropped"] == audit["state_tokens_dropped"] == 0
    assert all(n == 0 for n in audit["option_tokens_dropped"])
    assert "TARGET_END" in tokenizer.decode(audit["input_ids"])
    assert original["branch-id"]["instructions"]["branch"] == branch  # no caller mutation


@pytest.mark.parametrize("location", ["instructions", "criteria", "state"])
def test_rejects_silent_head_option_and_state_clipping(tokenizer, location):
    question = {"type": "noul", "instructions": "Is the evidence relevant?", "criteria": {"true": "yes", "false": "no"}}
    state = {"query": "What happened?"}
    if location == "instructions":
        question["instructions"] = "instruction " * 500
    elif location == "criteria":
        question["criteria"]["true"] = "positive option " * 100
    else:
        state["text"] = "document " * 9000
    with pytest.raises(ValueError, match="truncat"):
        audit_laya_input(tokenizer, state, question, max_len=8192, head_max_len=256)
