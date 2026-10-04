"""Pinned local Laya adapter with auditable typed decisions and truncation checks."""
from importlib.metadata import version
import copy
import hashlib
import json
import math
import os
import time

import torch

from .jev import JevBudgetExceeded, json_bytes
from .models import synchronize


INPUT_PROTOCOL = "per-target-state-v1"


def prepare_laya_inputs(state, questions):
    """Put target data in each state, leaving only the task in the SDK head.

    predict_batch evaluates one shared question over independent states. Each
    target therefore sees its own complete document, rather than all targets.
    """
    rows = []
    for qid, original in questions.items():
        question = copy.deepcopy(original)
        instructions = question["instructions"]
        effective_state = copy.deepcopy(state)
        if isinstance(instructions, dict):
            task = instructions.get("question")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("Structured Laya instructions require a question string.")
            effective_state = {"context": effective_state,
                               "decision_data": {k: v for k, v in instructions.items() if k != "question"}}
            question["instructions"] = task
        rows.append({"question_id": qid, "state": effective_state, "question": question})
    return rows


def audit_laya_input(tokenizer, state, question, *, max_len, head_max_len):
    """Compare full tokens with the installed SDK's actual head/state packing.

    Laya 0.3.26's usage.truncated checks state only. Instruction and option
    clipping are checked here before inference, including the 48-token cap.
    """
    from laya.agent import Agent
    from laya.common import build_head, build_sequence, encode_text, render_options, serialize_state
    internal = Agent._to_internal(question)
    encode = lambda text: encode_text(tokenizer, text, add_special_tokens=False)["input_ids"]
    instruction = f"{internal['t']} question: {internal['ins']}"
    options = render_options(internal)
    if any(tokenizer.mask_token in text for text in [instruction, serialize_state(state), *options]):
        raise ValueError("Laya would replace a literal mask token in the instructions or criteria.")
    order = internal.get("option_order", list(range(len(options))))
    if sorted(order) != list(range(len(options))):
        raise ValueError("Laya option_order must be a permutation of every option.")
    head, markers, _ = build_head(tokenizer, internal, head_max_len, option_order=order)
    sequence, actual_markers, _, state_stats = build_sequence(
        tokenizer, state, internal, max_len, head_max_len,
        option_order=order,
        return_stats=True, return_truncation_stats=True)
    instruction_dropped = len(encode(instruction)) - (markers[0] - 2)
    boundaries = markers[1:] + [len(head) - 1]
    option_dropped = [len(encode(" " + text)) - (end - begin - 1)
                      for text, begin, end in zip([options[i] for i in order], markers, boundaries)]
    audit = {"input_ids": sequence, "input_tokens": len(sequence),
             "instruction_tokens_dropped": instruction_dropped,
             "option_tokens_dropped": option_dropped, **state_stats,
             "input_ids_sha256": hashlib.sha256(json_bytes(sequence)).hexdigest()}
    if instruction_dropped or any(option_dropped) or state_stats["truncated"] or actual_markers != markers:
        raise ValueError(f"Laya would truncate input: instructions={instruction_dropped}, "
                         f"options={option_dropped}, state={state_stats['state_tokens_dropped']}.")
    return audit


class LayaDecisionClient:
    """Small local counterpart to JevClient's decide(state, questions) contract."""
    def __init__(self, config, *, device):
        self.config = config
        model_config = config["model"]
        self.model = model_config["repo"]
        self.revision = model_config["revision"]
        if len(self.revision) != 40 or any(char not in "0123456789abcdef" for char in self.revision):
            raise ValueError("Pin Laya to a 40-character Hugging Face commit SHA.")
        installed = version("laya")
        if installed != model_config["library_version"]:
            raise RuntimeError(f"Install laya=={model_config['library_version']} (found {installed}).")
        self.requested_device = torch.device(device)
        self.max_len = config["max_len"]
        self.head_max_len = config["head_max_len"]
        self.batch_size = config.get("batch_size", 8)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("Laya batch_size must be a positive integer.")
        if type(self.max_len) is not int or self.max_len < 1 or type(self.head_max_len) is not int or self.head_max_len < 1:
            raise ValueError("Laya token budgets must be positive integers.")
        if self.head_max_len >= self.max_len:
            raise ValueError("Laya head_max_len must leave room for the state.")
        if config["backend"] != "eager":
            raise ValueError("The first D run pins Laya's eager backend.")
        amp = config.get("cuda_amp")
        if amp not in (None, "fp16", "bf16"):
            raise ValueError("cuda_amp must be fp16, bf16, or null.")

        try:
            import laya
        except ImportError as error:
            raise RuntimeError("Install laya==0.3.26 in the notebook's Python environment.") from error

        previous_amp = os.environ.get("LAYA_CUDA_AMP")
        started = time.perf_counter()
        try:
            if self.requested_device.type == "cuda" and amp:
                os.environ["LAYA_CUDA_AMP"] = amp
            # The 0.3.26 wheel omits laya.backends. Its legacy flags still select eager.
            self.agent = laya.load(self.model, device=str(self.requested_device), revision=self.revision,
                                   fast=False, compile=False)
        finally:
            if previous_amp is None:
                os.environ.pop("LAYA_CUDA_AMP", None)
            else:
                os.environ["LAYA_CUDA_AMP"] = previous_amp
        synchronize(self.agent.device)
        self.load_seconds = time.perf_counter() - started
        self.actual_backend = self.agent.backend
        if self.actual_backend != config["backend"]:
            raise RuntimeError(f"Laya selected {self.actual_backend}; expected {config['backend']}.")
        self.actual_device = torch.device(self.agent.device)
        if self.actual_device != self.requested_device:
            raise RuntimeError(f"Laya fell back to {self.actual_device}; expected the A/B device {self.requested_device}.")
        if getattr(self.agent, "revision", None) != self.revision:
            raise RuntimeError("Laya did not load the configured immutable revision.")
        model_limit = model_config.get("max_context_tokens")
        if type(model_limit) is not int or self.max_len > model_limit:
            raise ValueError("Configured max_len exceeds the pinned checkpoint's declared context limit.")
        self.dtype = str(self.agent.dtype)
        self.warmup_seconds = self.agent.warmup()
        synchronize(self.actual_device)
        if getattr(self.agent, "cpu_fallback_count", 0):
            raise RuntimeError("Laya used a CPU fallback during warmup; the configured-device run is rejected.")

    def decide(self, state, questions, *, calls, max_calls, stage, on_record=None):
        if not questions:
            return {}
        if len(calls) + 1 > max_calls:
            raise JevBudgetExceeded("Laya local decision-call budget would be exceeded.")
        request = {"model": self.model, "revision": self.revision, "state": state, "questions": questions,
                   "max_len": self.max_len, "head_max_len": self.head_max_len,
                   "input_protocol": INPUT_PROTOCOL, "batch_size": self.batch_size}
        record = {"provider": "laya-local", "stage": stage, "request": request,
                  "request_bytes": len(json_bytes(request)), "status": "started", "response": None}
        calls.append(record)
        started = time.perf_counter()
        try:
            rows = prepare_laya_inputs(state, questions)
            groups = {}
            for row in rows:
                row["encoding"] = audit_laya_input(self.agent.tok, row["state"], row["question"],
                                                    max_len=self.max_len, head_max_len=self.head_max_len)
                groups.setdefault(json_bytes(row["question"]), []).append(row)
            record["effective_inputs"] = rows
            synchronize(self.actual_device)
            inference_started = time.perf_counter()
            fallback_before = getattr(self.agent, "cpu_fallback_count", 0)
            answers, native = {}, []
            for group in groups.values():
                outputs = self.agent.predict_batch([row["state"] for row in group],
                    {"decision": group[0]["question"]}, batch_size=self.batch_size,
                    max_len=self.max_len, head_max_len=self.head_max_len)
                if len(outputs) != len(group):
                    raise ValueError("Laya returned missing or unexpected batch rows.")
                for row, output in zip(group, outputs):
                    native.append({"question_id": row["question_id"], "response": output})
                    record["native_responses"] = native
                    usage = output.get("usage")
                    required = {"input_tokens", "output_tokens", "truncated", "state_tokens_dropped", "truncated_questions"}
                    if not isinstance(usage, dict) or not required <= usage.keys():
                        raise ValueError("Laya response lacks per-row token/truncation diagnostics.")
                    if any(type(usage[key]) is not int or usage[key] < 0
                           for key in ("input_tokens", "output_tokens", "state_tokens_dropped")):
                        raise ValueError("Laya response has invalid per-row token usage.")
                    if type(usage["truncated"]) is not bool or not isinstance(usage["truncated_questions"], list):
                        raise ValueError("Laya response has invalid per-row truncation diagnostics.")
                    if set(output.get("answers", {})) != {"decision"}:
                        raise ValueError("Laya returned missing or unexpected decision IDs.")
                    answers[row["question_id"]] = output["answers"]["decision"]
            body = {"answers": answers, "usage": {
                "input_tokens": sum(o["response"]["usage"]["input_tokens"] for o in native),
                "output_tokens": sum(o["response"]["usage"]["output_tokens"] for o in native),
                "truncated": any(o["response"]["usage"]["truncated"] for o in native),
                "state_tokens_dropped": sum(o["response"]["usage"]["state_tokens_dropped"] for o in native),
                "truncated_questions": [o["question_id"] for o in native
                                         if o["response"]["usage"]["truncated_questions"]],
            }}
            synchronize(self.actual_device)
            record["inference_seconds"] = time.perf_counter() - inference_started
            record["response"] = body
            record["cpu_fallbacks"] = getattr(self.agent, "cpu_fallback_count", 0) - fallback_before
            if record["cpu_fallbacks"] or torch.device(self.agent.device) != self.actual_device:
                raise RuntimeError("Laya used a different device during inference; mixed-device timing is rejected.")
            if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
                raise ValueError("Laya response has no answers object.")
            if set(body["answers"]) != set(questions):
                raise ValueError("Laya returned missing or unexpected decision IDs.")
            usage = body.get("usage")
            if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                                                  for key in ("input_tokens", "output_tokens")):
                raise ValueError("Laya response has invalid token usage.")
            truncation_fields = {"truncated", "state_tokens_dropped", "truncated_questions"}
            if not truncation_fields.intersection(usage):
                raise ValueError("This Laya version does not report enough input-truncation diagnostics.")
            dropped = usage.get("state_tokens_dropped", 0)
            truncated_questions = usage.get("truncated_questions", [])
            if type(dropped) is not int or dropped < 0 or not isinstance(truncated_questions, list):
                raise ValueError("Laya response has invalid truncation diagnostics.")
            if usage.get("truncated") or dropped or truncated_questions:
                record["status"] = "rejected_truncation"
                raise ValueError("Laya truncated the decision input; this run is not accepted as a valid observation.")
            if any(type(o["response"]["usage"].get(key)) is not int or o["response"]["usage"][key] < 0
                   for o in native for key in ("input_tokens", "output_tokens")):
                raise ValueError("Laya response has invalid per-row token usage.")
            if any(o["response"]["usage"]["input_tokens"] != row["encoding"]["input_tokens"]
                   for row, o in zip([row for group in groups.values() for row in group], native)):
                raise ValueError("Laya token usage disagrees with its audited effective input.")
            for key, question in questions.items():
                answer = body["answers"][key]
                if not isinstance(answer, dict) or answer.get("type") != question["type"]:
                    raise ValueError("Laya answer type does not match its question.")
                value_key = "noul" if question["type"] == "noul" else "score"
                value = answer.get(value_key)
                upper = 1 if value_key == "noul" else len(question["criteria"]) - 1
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
                    raise ValueError(f"Laya returned an invalid {value_key} value.")
            record["status"] = "completed"
            return body["answers"]
        except Exception as error:
            if record["status"] == "started":
                record["status"] = "failed"
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["seconds"] = time.perf_counter() - started
            if on_record:
                on_record(record)

    def runtime_info(self):
        allocated = torch.cuda.memory_allocated(self.actual_device) if self.actual_device.type == "cuda" else None
        peak = torch.cuda.max_memory_allocated(self.actual_device) if self.actual_device.type == "cuda" else None
        return {"model": self.model, "revision": self.revision, "library_version": version("laya"),
                "backend": self.actual_backend, "requested_device": str(self.requested_device),
                "actual_device": str(self.actual_device), "dtype": self.dtype,
                "max_len": self.max_len, "head_max_len": self.head_max_len,
                "input_protocol": INPUT_PROTOCOL, "batch_size": self.batch_size,
                "cpu_fallback_count": getattr(self.agent, "cpu_fallback_count", 0),
                "load_seconds": self.load_seconds, "warmup_seconds": self.warmup_seconds,
                "cuda_memory_allocated_bytes_after_warmup": allocated,
                "cuda_peak_memory_allocated_bytes": peak}
