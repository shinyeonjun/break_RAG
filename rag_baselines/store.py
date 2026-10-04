"""Common original-passage store and boundary checks for A/B/C/D."""
from pathlib import Path
from typing import Protocol

from .data import file_sha256, read_jsonl


class DocumentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = file_sha256(self.path)
        self.passages = read_jsonl(self.path)
        self.indices = {row["id"]: index for index, row in enumerate(self.passages)}
        if not self.passages or len(self.indices) != len(self.passages):
            raise ValueError("Corpus must have nonempty unique passage IDs.")

    def get(self, passage_id):
        return self.passages[self.indices[passage_id]]

    def validate_evidence(self, result):
        if result["corpus_sha256"] != self.sha256:
            raise ValueError("Returned evidence belongs to another corpus.")
        ids = [row["id"] for row in result["evidence"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence IDs.")
        for row in result["evidence"]:
            original = self.get(row["id"])
            if any(row.get(field) != original[field] for field in ("content", "source_spans", "title", "wikipedia_id")):
                raise ValueError("Evidence must return original content, source identity, and coordinates.")


class RetrievalAdapter(Protocol):
    """The shared original-evidence interface implemented by A/B and C."""
    store: DocumentStore

    def retrieve(self, question: str, method: str) -> dict: ...
