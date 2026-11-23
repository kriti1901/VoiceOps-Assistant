"""Streamlit UI smoke tests via ``AppTest``: the real script runs against a fake-backed pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.config import Settings
from app.pipeline import VoiceOpsService
from app.rag_store import RagStore
from tests.helpers import FakeEmbedder, FakeLLM, valid_analysis_json
from tests.test_api import FakeSTT

UI_SCRIPT = str(Path(__file__).resolve().parents[1] / "app" / "ui.py")
AUDIO_BYTES = b"RIFF....WAVEfmt fake-audio-bytes"


@pytest.fixture
def service(make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch) -> Iterator[VoiceOpsService]:
    """Patch the service class the UI script imports, so it builds our fake-backed instance."""
    settings = make_settings(chunk_max_chars=100, chunk_overlap_chars=0)
    llm = FakeLLM([valid_analysis_json(), "They forgot the migration (incident_call.wav @ 00:14–00:22)."])
    svc = VoiceOpsService(settings, stt=FakeSTT(), llm=llm, store=RagStore(settings, embedder=FakeEmbedder()))
    monkeypatch.setattr("app.pipeline.VoiceOpsService", lambda *_a, **_k: svc)
    st.cache_resource.clear()
    yield svc
    st.cache_resource.clear()


def run_app() -> AppTest:
    at = AppTest.from_file(UI_SCRIPT, default_timeout=30)
    at.run()
    return at


def test_empty_state_renders_three_tabs_without_errors(service: VoiceOpsService) -> None:
    at = run_app()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Upload & Transcribe", "Structured Tickets & Actions", "Ask Your Meetings (RAG)"]
    assert at.title[0].value == "VoiceOps Assistant"
    assert any("Upload a recording" in i.value for i in at.info)
    assert any("Process a recording" in i.value for i in at.info)


def test_sidebar_shows_health(service: VoiceOpsService) -> None:
    at = run_app()
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "fake-stt" in sidebar
    assert "In-process pipeline" in sidebar
    assert "Indexed chunks" in sidebar


def test_upload_process_and_view_results(service: VoiceOpsService) -> None:
    at = run_app()

    at.file_uploader(key="audio_upload").set_value(("incident_call.wav", AUDIO_BYTES, "audio/wav"))
    at.run()
    assert not at.exception
    process = next(b for b in at.button if b.label == "Transcribe & analyze")
    process.click()
    at.run()

    assert not at.exception, at.exception
    assert any("incident_call.wav" in s.value for s in at.success)
    # tab 1: full, timestamped transcript (wrapping text view, so nothing is truncated)
    transcript_blocks = [c for c in at.code if c.language == "plaintext"]
    assert transcript_blocks, "timestamped transcript should be rendered"
    assert "[00:00\u201300:06] Okay everyone, the checkout API is returning 500 errors since the deploy." in transcript_blocks[0].value
    assert "[00:30\u201300:41] Priya will write the post-mortem by Friday, and Marcus will add a migration check to CI." in transcript_blocks[0].value
    # tab 2: ticket + action cards + JSON
    markdown = " ".join(m.value for m in at.markdown)
    assert "Checkout API 500s after payments-service v2.3 deploy" in markdown
    assert "Write the post-mortem" in markdown
    assert "Priya" in " ".join(c.value for c in at.caption)
    code_blocks = [c for c in at.code if c.language == "json"]
    assert code_blocks and '"event": "voiceops.meeting.processed"' in code_blocks[0].value
    assert '"severity": "critical"' in code_blocks[0].value
    assert "transcript" not in code_blocks[0].value  # off by default; the transcript is opt-in


def test_chat_answers_with_timestamp_citations(service: VoiceOpsService) -> None:
    at = run_app()
    at.file_uploader(key="audio_upload").set_value(("incident_call.wav", AUDIO_BYTES, "audio/wav"))
    at.run()
    next(b for b in at.button if b.label == "Transcribe & analyze").click()
    at.run()

    at.chat_input[0].set_value("What did they forget to run?").run()

    assert not at.exception, at.exception
    messages = [(m.name, " ".join(x.value for x in m.markdown)) for m in at.chat_message]
    assert messages[0][0] == "user" and "forget to run" in messages[0][1]
    assert messages[1][0] == "assistant" and "migration" in messages[1][1]
    assert any("Sources (" in e.label for e in at.expander)
    assert any("incident_call.wav @ 00:14–00:22" in m.value for m in at.markdown)


def test_chat_degrades_cleanly_when_no_llm(service: VoiceOpsService) -> None:
    service._llm = None  # no key configured -> extractive excerpts
    at = run_app()
    at.session_state["results"] = {"incident_call.wav": service.process_audio("x.wav", "incident_call.wav")}
    at.run()
    at.chat_input[0].set_value("database migration").run()
    assert not at.exception
    assert any("raw excerpts" in c.value for c in at.caption)


def test_oversized_file_is_blocked(make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch, service: VoiceOpsService) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    at = run_app()
    at.file_uploader(key="audio_upload").set_value(("big.wav", b"0" * (1024 * 1024 + 1), "audio/wav"))
    at.run()
    assert any("larger than the 1 MB limit" in e.value for e in at.error)
    assert next(b for b in at.button if b.label == "Transcribe & analyze").disabled


def test_failed_processing_is_shown_not_raised(service: VoiceOpsService) -> None:
    from app.stt_engine import TranscriptionError

    service._stt = FakeSTT(error=TranscriptionError("cannot decode audio"))
    at = run_app()
    at.file_uploader(key="audio_upload").set_value(("broken.wav", AUDIO_BYTES, "audio/wav"))
    at.run()
    next(b for b in at.button if b.label == "Transcribe & analyze").click()
    at.run()
    assert not at.exception
    assert any("cannot decode audio" in e.value for e in at.error)


def test_without_llm_the_ticket_tab_explains_why(service: VoiceOpsService) -> None:
    service._llm = None
    at = run_app()
    at.session_state["results"] = {"incident_call.wav": service.process_audio("x.wav", "incident_call.wav")}
    at.run()
    assert not at.exception
    assert any("No structured analysis" in w.value for w in at.warning)
    assert any("GOOGLE_API_KEY" in w.value for w in at.sidebar.warning)


def test_remote_backend_is_used_when_api_url_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """With VOICEOPS_API_URL the UI must be a pure HTTP client (no in-process pipeline)."""
    import httpx

    from app.config import get_settings

    calls: list[tuple[str, str]] = []

    def fake_request(self: httpx.Client, method: str, url: str, **kwargs: Any) -> httpx.Response:
        calls.append((method, url))
        request = httpx.Request(method, url)
        if url.endswith("/api/health"):
            body: Any = {"status": "ok", "version": "1", "stt_provider": "remote-stt", "llm_provider": "gemini", "llm_configured": True, "indexed_chunks": 7}
        elif url.endswith("/api/sources"):
            body = ["incident_call.wav"]
        else:
            return httpx.Response(500, json={"detail": "boom"}, request=request)
        return httpx.Response(200, json=body, request=request)

    monkeypatch.setattr(httpx.Client, "request", fake_request)
    monkeypatch.setenv("VOICEOPS_API_URL", "http://api.test:8000/")
    get_settings.cache_clear()
    st.cache_resource.clear()
    try:
        at = run_app()
        assert not at.exception
        sidebar = " ".join(m.value for m in at.sidebar.markdown)
        assert "API at http://api.test:8000" in sidebar
        assert "remote-stt" in sidebar and "Indexed chunks:** 7" in sidebar
        assert ("GET", "http://api.test:8000/api/health") in calls
    finally:
        st.cache_resource.clear()
