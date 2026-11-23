"""Schema validation, coercion and serialization contracts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.schemas import (
    ActionItem,
    AskRequest,
    Citation,
    IncidentTicket,
    MeetingAnalysis,
    ProcessAudioResponse,
    Segment,
    Transcript,
    format_time_range,
    format_timestamp,
)
from tests.helpers import VALID_ANALYSIS, make_transcript


# --------------------------------------------------------------------------- timestamps
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "00:00"), (5.9, "00:05"), (75.3, "01:15"), (3599, "59:59"), (3600, "1:00:00"), (3723, "1:02:03"), (-4, "00:00")],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


def test_format_time_range() -> None:
    assert format_time_range(65, 130) == "01:05–02:10"


# --------------------------------------------------------------------------- segments / transcript
def test_segment_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="end_time"):
        Segment(start_time=5, end_time=4, text="hello")


def test_segment_rejects_negative_time_and_blank_text() -> None:
    with pytest.raises(ValidationError):
        Segment(start_time=-1, end_time=2, text="hello")
    with pytest.raises(ValidationError):
        Segment(start_time=0, end_time=2, text="   ")


def test_segment_strips_text() -> None:
    assert Segment(start_time=0, end_time=1, text="  hi  ").text == "hi"


def test_transcript_full_text_and_timestamped_text() -> None:
    transcript = Transcript(
        source_file="a.wav",
        segments=[
            Segment(start_time=0, end_time=2, text="Hello"),
            Segment(start_time=65, end_time=70, text="world", speaker="A"),
        ],
    )
    assert transcript.full_text == "Hello world"
    assert transcript.to_timestamped_text().splitlines() == ["[00:00] Hello", "[01:05] A: world"]


def test_transcript_dump_includes_full_text_and_roundtrips() -> None:
    transcript = make_transcript()
    dumped = transcript.model_dump(mode="json")
    assert dumped["full_text"] == transcript.full_text
    assert Transcript.model_validate(dumped) == transcript  # computed field is ignored on input


# --------------------------------------------------------------------------- action items
def test_action_item_defaults() -> None:
    item = ActionItem(task="Do the thing", owner=None, priority="HIGH")  # type: ignore[arg-type]
    assert item.owner == "Unassigned"
    assert item.priority == "high"
    assert item.due_date is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-11-27", "2026-11-27"),
        (" 2026-11-27T09:00:00Z ", "2026-11-27"),
        ("next Friday", None),
        ("ASAP", None),
        ("", None),
        ("2026-13-45", None),
        (None, None),
        (20261127, None),
    ],
)
def test_action_item_due_date_coercion(raw: object, expected: str | None) -> None:
    assert ActionItem(task="x", due_date=raw).due_date == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(("raw", "expected"), [("Urgent", "high"), ("normal", "medium"), ("Low", "low")])
def test_priority_aliases(raw: str, expected: str) -> None:
    assert ActionItem(task="x", priority=raw).priority == expected  # type: ignore[arg-type]


def test_invalid_priority_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ActionItem(task="x", priority="whenever")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- incident ticket
def test_ticket_normalizes_severity_and_category() -> None:
    ticket = IncidentTicket(
        title="t",
        severity="P1",  # type: ignore[arg-type]
        category="Customer Support",  # type: ignore[arg-type]
        description="d",
    )
    assert ticket.severity == "critical"
    assert ticket.category == "customer_support"
    assert ticket.impacted_system == "Unknown"
    assert ticket.suggested_next_steps == []


def test_ticket_unknown_category_falls_back_to_other() -> None:
    ticket = IncidentTicket(title="t", severity="low", category="weather", description="d")  # type: ignore[arg-type]
    assert ticket.category == "other"


def test_ticket_unknown_severity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        IncidentTicket(title="t", severity="apocalyptic", description="d")  # type: ignore[arg-type]


def test_ticket_next_steps_accepts_bulleted_string() -> None:
    ticket = IncidentTicket(
        title="t", severity="low", description="d", suggested_next_steps="- roll back\n* page oncall\n3. write RCA"  # type: ignore[arg-type]
    )
    assert ticket.suggested_next_steps == ["roll back", "page oncall", "write RCA"]


def test_ticket_long_title_is_truncated() -> None:
    ticket = IncidentTicket(title="x" * 500, severity="low", description="d")
    assert len(ticket.title) == 200
    assert ticket.title.endswith("...")


# --------------------------------------------------------------------------- analysis
def test_analysis_accepts_valid_payload() -> None:
    analysis = MeetingAnalysis.model_validate(VALID_ANALYSIS)
    assert len(analysis.executive_summary) == 3
    assert analysis.sentiment == "escalated"
    assert analysis.incident_ticket is not None
    assert analysis.action_items[0].owner == "Priya"


def test_analysis_trims_summary_to_three_and_strips_bullets() -> None:
    payload = {**VALID_ANALYSIS, "executive_summary": ["- one", "• two", "3. three", "four", "five"]}
    analysis = MeetingAnalysis.model_validate(payload)
    assert analysis.executive_summary == ["one", "two", "three"]


def test_analysis_summary_as_single_bulleted_string() -> None:
    payload = {**VALID_ANALYSIS, "executive_summary": "- first\n- second\n\n- third"}
    assert MeetingAnalysis.model_validate(payload).executive_summary == ["first", "second", "third"]


def test_analysis_rejects_empty_summary() -> None:
    with pytest.raises(ValidationError):
        MeetingAnalysis.model_validate({**VALID_ANALYSIS, "executive_summary": []})


def test_analysis_rejects_bad_sentiment_but_normalizes_case() -> None:
    with pytest.raises(ValidationError):
        MeetingAnalysis.model_validate({**VALID_ANALYSIS, "sentiment": "angry"})
    assert MeetingAnalysis.model_validate({**VALID_ANALYSIS, "sentiment": " Escalated "}).sentiment == "escalated"


def test_analysis_ticket_is_optional_and_null_items_ok() -> None:
    analysis = MeetingAnalysis.model_validate(
        {"executive_summary": ["a"], "sentiment": "neutral", "action_items": None, "incident_ticket": None}
    )
    assert analysis.incident_ticket is None
    assert analysis.action_items == []


def test_analysis_json_roundtrip() -> None:
    analysis = MeetingAnalysis.model_validate(VALID_ANALYSIS)
    assert MeetingAnalysis.model_validate_json(analysis.model_dump_json()) == analysis


def test_analysis_json_schema_lists_the_contract_fields() -> None:
    schema = MeetingAnalysis.model_json_schema()
    assert set(schema["properties"]) == {"executive_summary", "sentiment", "action_items", "incident_ticket"}


# --------------------------------------------------------------------------- API payloads
def test_process_response_webhook_payload_excludes_transcript_by_default() -> None:
    response = ProcessAudioResponse(
        source_file="incident_call.wav",
        transcript=make_transcript(),
        analysis=MeetingAnalysis.model_validate(VALID_ANALYSIS),
        indexed_chunks=3,
    )
    payload = response.webhook_payload()
    assert "transcript" not in payload
    assert payload["event"] == "voiceops.meeting.processed"
    assert payload["analysis"]["incident_ticket"]["severity"] == "critical"
    json.dumps(payload)  # JSON-safe (datetime serialized)
    assert "transcript" in response.webhook_payload(include_transcript=True)


def test_ask_request_validation() -> None:
    assert AskRequest(question="  what happened? ").question == "what happened?"
    with pytest.raises(ValidationError):
        AskRequest(question="   ")
    with pytest.raises(ValidationError):
        AskRequest(question="q", top_k=0)
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 2001)


def test_citation_label() -> None:
    citation = Citation(source_file="a.wav", start_time=61, end_time=95, snippet="s")
    assert citation.label == "a.wav @ 01:01–01:35"
