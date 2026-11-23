"""Test doubles shared across the suite. Nothing here touches the network or loads a model."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from app.schemas import Segment, Transcript

# A realistic, valid LLM answer for the incident transcript below.
VALID_ANALYSIS: dict[str, Any] = {
    "executive_summary": [
        "Checkout API returned 500s after the 14:05 deploy of payments-service v2.3.",
        "Rollback to v2.2 restored service at 14:40; root cause is a missing DB migration.",
        "Team agreed on a post-mortem and a migration check in CI.",
    ],
    "sentiment": "escalated",
    "action_items": [
        {
            "task": "Write the post-mortem for the checkout outage",
            "owner": "Priya",
            "due_date": "2026-11-27",
            "priority": "high",
        },
        {
            "task": "Add a migration check to the CI pipeline",
            "owner": "Marcus",
            "due_date": None,
            "priority": "medium",
        },
    ],
    "incident_ticket": {
        "title": "Checkout API 500s after payments-service v2.3 deploy",
        "severity": "critical",
        "category": "outage",
        "impacted_system": "payments-service / checkout API",
        "description": "Deploy of v2.3 shipped without the matching DB migration, causing 500s on checkout.",
        "suggested_next_steps": ["Roll back to v2.2", "Run the missing migration", "Write post-mortem"],
    },
}


def valid_analysis_json(**overrides: Any) -> str:
    return json.dumps({**VALID_ANALYSIS, **overrides})


def make_segments() -> list[Segment]:
    return [
        Segment(start_time=0.0, end_time=6.2, text="Okay everyone, the checkout API is returning 500 errors since the deploy."),
        Segment(start_time=6.2, end_time=14.0, text="Priya: it started right after payments-service version 2.3 went out at 2:05 PM."),
        Segment(start_time=14.0, end_time=22.5, text="Marcus: I think we forgot to run the database migration for the orders table."),
        Segment(start_time=22.5, end_time=30.0, text="Let's roll back to version 2.2 immediately and confirm the error rate drops."),
        Segment(start_time=30.0, end_time=41.0, text="Priya will write the post-mortem by Friday, and Marcus will add a migration check to CI."),
    ]


def make_transcript(source_file: str = "incident_call.wav") -> Transcript:
    return Transcript(
        source_file=source_file,
        provider="fake",
        language="en",
        duration=41.0,
        segments=make_segments(),
    )


class FakeLLM:
    """Scripted ``LLMClient``: pops queued responses (str, or an Exception to raise) and records every call."""

    def __init__(self, responses: list[str | Exception] | None = None, name: str = "fake") -> None:
        self.name = name
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def generate(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})
        if not self.responses:
            raise AssertionError("FakeLLM ran out of scripted responses")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Deterministic hashed bag-of-words embedder: shared words => higher cosine similarity.

    Lets us test real Chroma retrieval/ranking without downloading a model.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
