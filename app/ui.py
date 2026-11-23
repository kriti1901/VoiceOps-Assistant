"""Streamlit frontend.

Run with:  streamlit run app/ui.py

By default the UI calls the pipeline in-process (one command, no server). If ``VOICEOPS_API_URL`` is set
(e.g. ``http://localhost:8000``) it becomes a thin client of the FastAPI backend instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# `streamlit run app/ui.py` puts app/ (the script's folder) on sys.path, not the repository root,
# so `from app...` would fail without this.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402

from app import __version__  # noqa: E402
from app.client import Backend, BackendError, LocalBackend, RemoteBackend  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.pipeline import VoiceOpsService  # noqa: E402
from app.schemas import Citation, ProcessAudioResponse, format_timestamp  # noqa: E402
from app.stt_engine import SUPPORTED_EXTENSIONS  # noqa: E402

SENTIMENT_COLORS = {"positive": "green", "neutral": "gray", "escalated": "red"}
SEVERITY_COLORS = {"critical": "red", "high": "orange", "medium": "blue", "low": "gray"}
PRIORITY_COLORS = {"high": "red", "medium": "orange", "low": "gray"}
ALL_RECORDINGS = "All recordings"


@st.cache_resource(show_spinner=False)
def get_backend(api_url: str | None) -> Backend:
    if api_url:
        return RemoteBackend(api_url)
    return LocalBackend(VoiceOpsService(get_settings()))


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #


def results_state() -> dict[str, ProcessAudioResponse]:
    return st.session_state.setdefault("results", {})


def chat_state() -> list[dict[str, Any]]:
    return st.session_state.setdefault("chat", [])


def recording_picker(label: str, key: str) -> ProcessAudioResponse | None:
    results = results_state()
    if not results:
        return None
    names = list(results)
    choice = st.selectbox(label, names, index=len(names) - 1, key=key) if len(names) > 1 else names[0]
    return results[choice]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_sidebar(backend: Backend) -> None:
    with st.sidebar:
        st.title("VoiceOps")
        st.caption(f"Assistant v{__version__}")
        st.markdown(f"**Backend:** {backend.label}")
        try:
            health = backend.health()
        except BackendError as exc:
            st.error(str(exc))
            return
        st.markdown(f"**Speech-to-text:** `{health.stt_provider}`")
        if health.llm_configured:
            st.markdown(f"**LLM:** `{health.llm_provider}`")
        else:
            st.warning(
                "No LLM configured. Transcription and search excerpts still work, but summaries, "
                "tickets and chat answers need `GOOGLE_API_KEY` or `GROQ_API_KEY` in `.env` (both free)."
            )
        if health.indexed_chunks is not None:
            st.markdown(f"**Indexed chunks:** {health.indexed_chunks}")
        if health.status != "ok":
            st.warning("The knowledge base is degraded; check the server logs.")


def render_upload_tab(backend: Backend) -> None:
    max_mb = get_settings().max_upload_mb
    st.subheader("Upload a recording")
    st.caption(
        f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)} (up to {max_mb} MB). With the default local "
        "engine, audio never leaves this machine; the first run downloads the Whisper model."
    )
    uploaded = st.file_uploader(
        "Meeting or incident-call audio",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        key="audio_upload",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        st.audio(data, format=uploaded.type or "audio/wav")
        too_big = len(data) > max_mb * 1024 * 1024
        if too_big:
            st.error(f"This file is larger than the {max_mb} MB limit (MAX_UPLOAD_MB).")
        if st.button("Transcribe & analyze", type="primary", disabled=too_big):
            with st.spinner("Transcribing, extracting action items and indexing... (long recordings take a while on CPU)"):
                try:
                    result = backend.process_audio(uploaded.name, data)
                except BackendError as exc:
                    st.error(str(exc))
                else:
                    results_state()[result.source_file] = result
                    st.success(f"Processed **{result.source_file}** - {len(result.transcript.segments)} segments.")

    result = recording_picker("Recording", key="transcript_picker")
    if result is None:
        st.info("Upload a recording and press **Transcribe & analyze** to get started.")
        return

    transcript = result.transcript
    for warning in result.warnings:
        st.warning(warning)

    st.caption(f"Engine: **{transcript.provider}**  \u00b7  Detected language: **{transcript.language or 'n/a'}**")
    duration_col, segments_col, _spacer = st.columns([1, 1, 3])
    duration_col.metric("Duration", format_timestamp(transcript.duration) if transcript.duration is not None else "n/a")
    segments_col.metric("Segments", len(transcript.segments))

    # A wrapping, scrollable text view (not a dataframe: dataframe cells truncate long segments).
    lines = [
        f"[{format_timestamp(s.start_time)}\u2013{format_timestamp(s.end_time)}] "
        + (f"{s.speaker}: " if s.speaker else "")
        + s.text
        for s in transcript.segments
    ]
    wrapped_rows = sum(len(line) // 100 + 1 for line in lines)
    st.code("\n".join(lines), language=None, wrap_lines=True, height=max(110, min(480, 24 * wrapped_rows + 28)))
    st.download_button(
        "Download transcript (.txt)",
        transcript.to_timestamped_text(),
        file_name=f"{Path(result.source_file).stem}_transcript.txt",
        mime="text/plain",
    )


def render_tickets_tab() -> None:
    result = recording_picker("Recording", key="ticket_picker")
    if result is None:
        st.info("Process a recording in the first tab to see its summary, action items and ticket here.")
        return

    analysis = result.analysis
    if analysis is None:
        st.warning("No structured analysis is available for this recording.")
        for warning in result.warnings:
            st.write(f"- {warning}")
        return

    left, right = st.columns([3, 1])
    left.subheader("Executive summary")
    with right:
        st.badge(analysis.sentiment.capitalize(), color=SENTIMENT_COLORS[analysis.sentiment])
    for bullet in analysis.executive_summary:
        st.markdown(f"- {bullet}")

    st.subheader(f"Action items ({len(analysis.action_items)})")
    if not analysis.action_items:
        st.caption("No action items were identified.")
    for item in analysis.action_items:
        with st.container(border=True):
            top, badge_col = st.columns([5, 1])
            top.markdown(f"**{item.task}**")
            with badge_col:
                st.badge(item.priority.capitalize(), color=PRIORITY_COLORS[item.priority])
            st.caption(f"Owner: **{item.owner}**  ·  Due: **{item.due_date or 'not specified'}**")

    st.subheader("Incident ticket")
    ticket = analysis.incident_ticket
    if ticket is None:
        st.caption("This recording does not describe an incident, so no ticket was created.")
    else:
        with st.container(border=True):
            top, badge_col = st.columns([5, 1])
            top.markdown(f"#### {ticket.title}")
            with badge_col:
                st.badge(ticket.severity.capitalize(), color=SEVERITY_COLORS[ticket.severity])
            st.caption(f"Category: **{ticket.category.replace('_', ' ')}**  ·  Impacted system: **{ticket.impacted_system}**")
            st.write(ticket.description)
            if ticket.suggested_next_steps:
                st.markdown("**Suggested next steps**")
                for number, step in enumerate(ticket.suggested_next_steps, start=1):
                    st.markdown(f"{number}. {step}")

    st.subheader("Webhook-ready JSON")
    st.caption("Use the copy icon at the top-right of the block, or download the file. Paste into Jira, n8n or Google Sheets.")
    scope = st.radio(
        "Payload",
        ["Full payload", "Incident ticket only", "Action items only"],
        horizontal=True,
        key="json_scope",
    )
    if scope == "Incident ticket only":
        payload: Any = ticket.model_dump(mode="json") if ticket else None
    elif scope == "Action items only":
        payload = [item.model_dump(mode="json") for item in analysis.action_items]
    else:
        include_transcript = st.toggle("Include the full transcript", value=False, key="json_transcript")
        payload = result.webhook_payload(include_transcript=include_transcript)

    if payload is None:
        st.info("There is no incident ticket for this recording.")
        return
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    st.code(json_text, language="json")
    slug = {"Full payload": "payload", "Incident ticket only": "ticket", "Action items only": "action_items"}[scope]
    st.download_button(
        "Download JSON",
        json_text,
        file_name=f"{Path(result.source_file).stem}_{slug}.json",
        mime="application/json",
    )


def _render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations: list[Citation] = message.get("citations") or []
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for citation in citations:
                    score = f" · relevance {citation.score:.2f}" if citation.score is not None else ""
                    st.markdown(f"**{citation.label}**{score}")
                    st.caption(citation.snippet)
        if message.get("answer_source") == "extractive":
            st.caption("The LLM was unavailable, so these are raw excerpts instead of a written answer.")


def render_chat_tab(backend: Backend) -> None:
    try:
        sources = backend.sources()
    except BackendError:
        sources = []
    sources = sorted(set(sources) | set(results_state()))

    top_left, top_mid, top_right = st.columns([3, 2, 1])
    scope = top_left.selectbox("Search in", [ALL_RECORDINGS, *sources], key="chat_scope")
    top_k = top_mid.slider("Excerpts to retrieve", min_value=1, max_value=10, value=get_settings().rag_top_k, key="chat_top_k")
    if top_right.button("Clear chat", width="stretch"):
        chat_state().clear()

    history_box = st.container()
    prompt = st.chat_input("Ask about your meetings, e.g. \"Who owns the post-mortem and when is it due?\"")
    if prompt:
        chat_state().append({"role": "user", "content": prompt})
        with st.spinner("Searching your meetings..."):
            try:
                answer = backend.ask(prompt, top_k=top_k, source_file=None if scope == ALL_RECORDINGS else scope)
            except BackendError as exc:
                chat_state().append({"role": "assistant", "content": f"Sorry, that failed: {exc}"})
            else:
                chat_state().append(
                    {
                        "role": "assistant",
                        "content": answer.answer,
                        "citations": answer.citations,
                        "answer_source": answer.answer_source,
                    }
                )

    with history_box:
        if not chat_state():
            st.caption("Answers are grounded in your transcripts and cite the recording and timestamp they came from.")
        for message in chat_state():
            _render_message(message)


def main() -> None:
    st.set_page_config(page_title="VoiceOps Assistant", page_icon="\U0001F399️", layout="wide")
    st.title("VoiceOps Assistant")
    st.caption("Turn meeting and incident-call recordings into action items, tickets and a searchable knowledge base.")

    backend = get_backend(get_settings().voiceops_api_url)

    tab_upload, tab_tickets, tab_chat = st.tabs(
        ["Upload & Transcribe", "Structured Tickets & Actions", "Ask Your Meetings (RAG)"]
    )
    with tab_upload:
        render_upload_tab(backend)
    with tab_tickets:
        render_tickets_tab()
    with tab_chat:
        render_chat_tab(backend)

    # Rendered last so the health panel (e.g. "Indexed chunks") reflects what this run just processed.
    render_sidebar(backend)


if __name__ == "__main__":
    main()
