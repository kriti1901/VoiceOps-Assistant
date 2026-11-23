"""Orchestration shared by the FastAPI backend and the Streamlit UI (in-process mode).

``process_audio`` = transcribe -> extract -> index. Extraction and indexing failures never throw away an
expensive transcription: the transcript is still returned, with human-readable ``warnings``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app import __version__
from app.config import Settings, get_settings
from app.extractor import (
    ExtractionError,
    LLMClient,
    LLMError,
    LLMNotConfiguredError,
    configured_llm_providers,
    extract_analysis,
    get_llm_client,
)
from app.rag_store import RagStore, get_default_store
from app.schemas import (
    AskRequest,
    HealthResponse,
    MeetingAnalysis,
    ProcessAudioResponse,
    RAGAnswer,
)
from app.stt_engine import STTProvider, get_stt_provider

logger = logging.getLogger(__name__)

_MAX_WARNING_CHARS = 300


class NoSpeechDetectedError(ValueError):
    """The audio was decoded but contained no recognizable speech."""


def safe_filename(raw: str | None) -> str:
    """Strip any directory part (``../../etc/passwd``, ``C:\\x\\y.wav``) and control characters."""
    name = re.split(r"[\\/]", raw or "")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip().lstrip(".")
    return name[:120] or "audio"


def _short(message: str) -> str:
    message = " ".join(message.split())
    return message if len(message) <= _MAX_WARNING_CHARS else message[: _MAX_WARNING_CHARS - 1] + "…"


class VoiceOpsService:
    """Facade over STT, LLM extraction and the RAG store. All dependencies are injectable."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        stt: STTProvider | None = None,
        llm: LLMClient | None = None,
        store: RagStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._stt = stt
        self._llm = llm
        self._store = store

    # ----- lazily resolved collaborators ------------------------------------
    @property
    def stt(self) -> STTProvider:
        if self._stt is None:
            self._stt = get_stt_provider(self.settings)
        return self._stt

    @property
    def store(self) -> RagStore:
        if self._store is None:
            self._store = get_default_store()
        return self._store

    def _get_llm(self) -> LLMClient:
        """Return the configured LLM or raise ``LLMNotConfiguredError``."""
        if self._llm is None:
            self._llm = get_llm_client(self.settings)
        return self._llm

    # ----- use cases ---------------------------------------------------------
    def process_audio(self, audio_path: str | Path, source_file: str | None = None) -> ProcessAudioResponse:
        transcript = self.stt.transcribe(audio_path, source_file=source_file)
        if not transcript.segments:
            raise NoSpeechDetectedError("No speech was detected in this recording.")

        warnings: list[str] = []

        analysis: MeetingAnalysis | None = None
        try:
            analysis = extract_analysis(transcript, self._get_llm())
        except LLMNotConfiguredError:
            warnings.append(
                "No LLM configured - summary, action items and ticket were skipped. "
                "Set GOOGLE_API_KEY or GROQ_API_KEY in .env (both have free tiers)."
            )
        except (ExtractionError, LLMError) as exc:
            logger.warning("Extraction failed for %s: %s", transcript.source_file, exc)
            warnings.append(_short(f"Structured extraction failed: {exc}"))

        indexed = 0
        try:
            indexed = self.store.index_transcript(transcript)
        except Exception as exc:  # noqa: BLE001 - indexing problems must not lose the transcript
            logger.exception("Indexing failed for %s", transcript.source_file)
            warnings.append(_short(f"Indexing into the knowledge base failed: {type(exc).__name__}: {exc}"))

        return ProcessAudioResponse(
            source_file=transcript.source_file,
            transcript=transcript,
            analysis=analysis,
            indexed_chunks=indexed,
            warnings=warnings,
        )

    def ask(self, request: AskRequest) -> RAGAnswer:
        try:
            llm: LLMClient | None = self._get_llm()
        except LLMNotConfiguredError:
            llm = None  # the store falls back to showing the most relevant excerpts
        return self.store.query(request.question, top_k=request.top_k, source_file=request.source_file, llm=llm)

    def list_sources(self) -> list[str]:
        return self.store.list_sources()

    def health(self) -> HealthResponse:
        providers = configured_llm_providers(self.settings)
        indexed: int | None = None
        status = "ok"
        try:
            indexed = self.store.count()
        except Exception:  # noqa: BLE001 - health must never raise
            logger.exception("Health check could not read the vector store")
            status = "degraded"
        return HealthResponse(
            status=status,  # type: ignore[arg-type]
            version=__version__,
            stt_provider=self.stt.name,
            llm_provider="→".join(providers) if providers else None,
            llm_configured=bool(providers),
            indexed_chunks=indexed,
        )
