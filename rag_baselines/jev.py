"""Pinned TypeSafe HTTP contract, bounded batches, and per-call audit records.

UTF-8 byte limits are conservative packing guards, not Jev token counts.
Authoritative input/output tokens come only from the provider's response.
"""
import json
import math
import os
import time

import httpx


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class JevBudgetExceeded(RuntimeError):
    pass


class JevClient:
    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, config, *, api_key=None, http_client=None):
        self.config = config
        self.model = config["model"]
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self._api_key:
            raise RuntimeError("Set TYPESAFE_API_KEY in the project-root .env.")
        if not self.model.startswith("jev-") or self.model in ("jev-latest", "jev-preview"):
            raise ValueError("Pin an explicit Jev version for reproducible experiments.")
        if not 1 <= config["questions_per_request"] <= 255:
            raise ValueError("Invalid question batch size.")
        for key in ("request_max_bytes", "state_question_max_bytes"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError("Jev byte guards must be positive integers.")
        call_usage([], config["prices"])
        self._owns_client = http_client is None
        self.http = http_client or httpx.Client(timeout=config["timeout_seconds"])

    def close(self):
        if self._owns_client:
            self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def batches(self, state, questions):
        if not questions:
            return []
        batches, current = [], {}
        for key, question in questions.items():
            if len(json_bytes({"state": state, "question": question})) > self.config["state_question_max_bytes"]:
                raise ValueError("Jev state plus one question exceeds the byte guard; no text was truncated.")
            trial = {**current, key: question}
            size = len(json_bytes({"model": self.model, "state": state, "questions": trial}))
            if current and (len(trial) > self.config["questions_per_request"] or size > self.config["request_max_bytes"]):
                batches.append(current)
                current = {key: question}
            else:
                current = trial
            if len(json_bytes({"model": self.model, "state": state, "questions": current})) > self.config["request_max_bytes"]:
                raise ValueError("One Jev question exceeds the request byte guard.")
        if current:
            batches.append(current)
        return batches

    def decide(self, state, questions, *, calls, max_calls, stage, on_record=None):
        batches = self.batches(state, questions)
        if len(calls) + len(batches) > max_calls:
            raise JevBudgetExceeded("Jev API call budget would be exceeded.")
        answers = {}
        for batch in batches:
            payload = {"model": self.model, "state": state, "questions": batch}
            record = {"stage": stage, "request": payload, "request_bytes": len(json_bytes(payload)),
                      "status": "started", "response": None}
            calls.append(record)
            started = time.perf_counter()
            try:
                response = self.http.post(self.endpoint, content=json_bytes(payload),
                                          headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
                record["http_status"] = response.status_code
                if response.status_code != 200:
                    # Provider error messages can echo credentials; record only safe status.
                    raise RuntimeError(f"Jev HTTP {response.status_code}; no automatic retry.")
                body = response.json()
                record["response"] = body
                self.validate_response(body, batch)
                record["status"] = "completed"
                answers.update(body["answers"])
            except Exception as error:
                record["status"] = "failed"
                record["error_type"] = type(error).__name__
                raise
            finally:
                record["seconds"] = time.perf_counter() - started
                if on_record:
                    on_record(record)
        return answers

    def validate_response(self, body, questions):
        if body.get("model") != self.model:
            raise ValueError("Jev returned a different model version.")
        if set(body.get("answers", {})) != set(questions):
            raise ValueError("Jev returned missing or unexpected answer IDs.")
        usage = body.get("usage", {})
        if any(type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
            raise ValueError("Jev response has invalid usage.")
        for key, question in questions.items():
            answer, kind = body["answers"][key], question["type"]
            if answer.get("type") != kind:
                raise ValueError("Jev answer type does not match the question.")
            if kind == "noul":
                self._number(answer.get("noul"), 0, 1)
                continue
            criteria = question["criteria"]
            options = set(criteria) if kind == "choice" else {str(i) for i in range(len(criteria))}
            probabilities = answer.get("probabilities", {})
            if set(probabilities) != options:
                raise ValueError("Jev probability keys do not match criteria.")
            for value in probabilities.values():
                self._number(value, 0, 1)
            # Live responses round each probability to two decimal places.
            # Keep those raw values; require a normalized distribution to fit
            # inside their rounding intervals instead of renormalizing them.
            rounding = 0.005
            lower = sum(max(0, value - rounding) for value in probabilities.values())
            upper = sum(min(1, value + rounding) for value in probabilities.values())
            if not lower - 1e-9 <= 1 <= upper + 1e-9:
                raise ValueError("Jev probabilities cannot sum to one within rounding precision.")
            self._number(answer.get("confidence"), 0, 1)
            if kind == "choice":
                if answer.get("choice") not in options:
                    raise ValueError("Jev returned an unknown choice.")
            elif kind == "score":
                self._number(answer.get("score"), 0, len(criteria) - 1)
                expected = sum(int(level) * value for level, value in probabilities.items())
                tolerance = rounding * (1 + sum(range(len(criteria))))
                if not math.isclose(answer["score"], expected, abs_tol=tolerance + 1e-9):
                    raise ValueError("Jev score disagrees with its probabilities.")
            else:
                raise ValueError("Unsupported Jev primitive.")

    @staticmethod
    def _number(value, lower, upper):
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("Jev returned an invalid numeric decision.")


def call_usage(calls, prices):
    responses = []
    for call in calls:
        body = call.get("response")
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict) and all(type(usage.get(key)) is int and usage[key] >= 0
                                           for key in ("input_tokens", "output_tokens")):
            responses.append(body)
    known = len(responses) == len(calls)
    usage = {key: sum(response["usage"][key] for response in responses) for key in ("input_tokens", "output_tokens")}
    usage.update({"requests": len(calls), "usage_complete": known})
    cost = None
    if known and prices is not None:
        if any(type(prices.get(key)) not in (int, float) or not math.isfinite(prices[key]) or prices[key] < 0
               for key in ("input_usd_per_million", "output_usd_per_million")):
            raise ValueError("Provide finite nonnegative Jev token prices.")
        cost = sum(usage[key] * prices[f"{key.removesuffix('_tokens')}_usd_per_million"]
                   for key in ("input_tokens", "output_tokens")) / 1_000_000
    return usage, cost
