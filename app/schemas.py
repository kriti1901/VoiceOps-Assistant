"""Pydantic models shared by the STT engine, extractor, RAG store, API and UI."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

Sentiment = Literal["positive", "neutral", "escalated"]
Priority = Literal["low", "medium", "high"]
Severity = Literal["low", "medium", "high", "critical"]
Category = Literal[
    "outage",
    "performance",
    "security",
    "bug",
    "data",
    "infrastructure",
    "customer_support",
    "feature_request",
    "process",
    "other",
]

SENTIMENTS: tuple[str, ...] = ("positive", "neutral", "escalated")
PRIORITIES: tuple[str, ...] = ("low", "medium", "high")
SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
CATEGORIES: tuple[str, ...] = (
    "outage",
    "performance",
    "security",
    "bug",
    "data",
    "infrastructure",
    "customer_support",
    "feature_request",
    "process",
    "other",
)

_LEVEL_ALIASES = {
    "p0": "critical",
    "p1": "critical",
    "sev0": "critical",
    "sev1": "critical",
    "sev-1": "critical",
    "blocker": "critical",
    "urgent": "high",
    "p2": "high",
    "sev2": "high",
    "sev-2": "high",
    "normal": "medium",
    "moderate": "medium",
    "p3": "medium",
    "sev3": "medium",
    "sev-3": "medium",
    "minor": "low",
    "p4": "low",
    "sev4": "low",
    "sev-4": "low",
}

_ISO_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})(?:$|[T\s])")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s*")


def _normalize_level(value: Any) -> Any:
    """Lower-case / strip a categorical string and map common aliases (``P1`` -> ``critical``)."""
    if isinstance(value, str):
        cleaned = value.strip().lower().replace(" ", "_")
        return _LEVEL_ALIASES.get(cleaned, cleaned)
    return value


def format_timestamp(seconds: float) -> str:
    """``75.3`` -> ``01:15``; one hour or more -> ``1:02:03``."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_time_range(start: float, end: float) -> str:
    return f"{format_timestamp(start)}–{format_timestamp(end)}"


# --------------------------------------------------------------------------- #
# Transcripts
# --------------------------------------------------------------------------- #


class Segment(BaseModel):
    """One normalized, timestamped piece of speech (all providers return this shape)."""

    start_time: float = Field(ge=0, description="Start offset in seconds")
    end_time: float = Field(ge=0, description="End offset in seconds")
    text: NonEmptyStr
    speaker: str | None = None

    @model_validator(mode="after")
    def _end_not_before_start(self) -> Segment:
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self


class Transcript(BaseModel):
    source_file: NonEmptyStr
    provider: str = "unknown"
    language: str | None = None
    duration: float | None = Field(default=None, ge=0)
    segments: list[Segment] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_text(self) -> str:
        return " ".join(segment.text for segment in self.segments)

    def to_timestamped_text(self) -> str:
        """One line per segment, e.g. ``[01:15] Let's roll back the deploy.``"""
        lines = []
        for segment in self.segments:
            prefix = f"[{format_timestamp(segment.start_time)}]"
            if segment.speaker:
                prefix += f" {segment.speaker}:"
            lines.append(f"{prefix} {segment.text}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Structured extraction
# --------------------------------------------------------------------------- #


class ActionItem(BaseModel):
    task: NonEmptyStr
    owner: str = "Unassigned"
    due_date: str | None = Field(
        default=None, description="ISO 8601 date (YYYY-MM-DD) if one was stated, otherwise null"
    )
    priority: Priority = "medium"

    @field_validator("owner", mode="before")
    @classmethod
    def _default_owner(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "Unassigned"
        return value.strip() if isinstance(value, str) else value

    @field_validator("due_date", mode="before")
    @classmethod
    def _coerce_due_date(cls, value: Any) -> str | None:
        """Keep valid ISO dates; anything else ("next Friday", "ASAP", "") becomes ``None``."""
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if not isinstance(value, str):
            return None
        match = _ISO_DATE_RE.match(value.strip())
        if not match:
            return None
        try:
            return date.fromisoformat(match.group(1)).isoformat()
        except ValueError:
            return None

    @field_validator("priority", mode="before")
    @classmethod
    def _norm_priority(cls, value: Any) -> Any:
        return _normalize_level(value)


class IncidentTicket(BaseModel):
    """Jira-style ticket. Flat on purpose, so it maps 1:1 onto Jira / n8n / Google Sheets columns."""

    title: NonEmptyStr = Field(max_length=200)
    severity: Severity
    category: Category = "other"
    impacted_system: NonEmptyStr = "Unknown"
    description: NonEmptyStr
    suggested_next_steps: list[NonEmptyStr] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _truncate_title(cls, value: Any) -> Any:
        if isinstance(value, str) and len(value.strip()) > 200:
            return value.strip()[:197] + "..."
        return value

    @field_validator("severity", mode="before")
    @classmethod
    def _norm_severity(cls, value: Any) -> Any:
        return _normalize_level(value)

    @field_validator("category", mode="before")
    @classmethod
    def _norm_category(cls, value: Any) -> Any:
        """Unknown categories fall back to ``other`` instead of failing the whole extraction."""
        if not isinstance(value, str):
            return "other"
        cleaned = value.strip().lower().replace(" ", "_").replace("-", "_")
        return cleaned if cleaned in CATEGORIES else "other"

    @field_validator("impacted_system", mode="before")
    @classmethod
    def _default_system(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "Unknown"
        return value

    @field_validator("suggested_next_steps", mode="before")
    @classmethod
    def _steps_as_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [line for line in (_BULLET_PREFIX_RE.sub("", ln) for ln in value.splitlines()) if line.strip()]
        return value


class MeetingAnalysis(BaseModel):
    """The strict JSON contract the LLM must satisfy."""

    executive_summary: list[NonEmptyStr] = Field(
        min_length=1, max_length=3, description="Exactly three concise bullets"
    )
    sentiment: Sentiment
    action_items: list[ActionItem] = Field(default_factory=list)
    incident_ticket: IncidentTicket | None = Field(
        default=None,
        description="Populated only when the recording describes an incident, outage or support escalation",
    )

    @field_validator("executive_summary", mode="before")
    @classmethod
    def _bullets(cls, value: Any) -> Any:
        """Accept a bullet-formatted string, strip bullet glyphs, and keep at most 3 bullets."""
        if isinstance(value, str):
            value = value.splitlines()
        if isinstance(value, (list, tuple)):
            cleaned = [_BULLET_PREFIX_RE.sub("", item).strip() for item in value if isinstance(item, str)]
            return [item for item in cleaned if item][:3]
        return value

    @field_validator("sentiment", mode="before")
    @classmethod
    def _norm_sentiment(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("action_items", mode="before")
    @classmethod
    def _null_items(cls, value: Any) -> Any:
        return [] if value is None else value


# --------------------------------------------------------------------------- #
# API payloads
# --------------------------------------------------------------------------- #


class ProcessAudioResponse(BaseModel):
    """Result of ``POST /api/process-audio``; also the webhook-ready export format."""

    event: Literal["voiceops.meeting.processed"] = "voiceops.meeting.processed"
    schema_version: str = "1.0"
    source_file: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    transcript: Transcript
    analysis: MeetingAnalysis | None = None
    indexed_chunks: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    def webhook_payload(self, include_transcript: bool = False) -> dict[str, Any]:
        """JSON-safe dict for Jira / n8n / Google Sheets. The (large) transcript is opt-in."""
        exclude = None if include_transcript else {"transcript"}
        return self.model_dump(mode="json", exclude=exclude)


class AskRequest(BaseModel):
    question: NonEmptyStr = Field(max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    source_file: str | None = Field(default=None, description="Restrict the search to one recording")


class Citation(BaseModel):
    source_file: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    snippet: str
    score: float | None = Field(default=None, description="Cosine similarity, higher = more relevant")

    @property
    def label(self) -> str:
        return f"{self.source_file} @ {format_time_range(self.start_time, self.end_time)}"


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    answer_source: Literal["llm", "extractive", "no_data"] = Field(
        default="llm",
        description="llm = grounded LLM answer; extractive = LLM unavailable, raw excerpts; "
        "no_data = nothing indexed / nothing relevant",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    stt_provider: str
    llm_provider: str | None = Field(default=None, description="Resolved LLM provider order, None if unconfigured")
    llm_configured: bool = False
    indexed_chunks: int | None = None
