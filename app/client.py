"""Backends the Streamlit UI talks to: the in-process pipeline, or a remote FastAPI server.

These live in a normal module (not in ``ui.py``) on purpose: Streamlit re-executes ``ui.py`` on every
interaction, which would redefine ``BackendError`` as a *new* class each time. A cached backend object would
then keep raising the previous run's class and the UI's ``except BackendError`` would no longer match it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import httpx

from app.pipeline import NoSpeechDetectedError, VoiceOpsService, safe_filename
from app.schemas import AskRequest, HealthResponse, ProcessAudioResponse, RAGAnswer
from app.stt_engine import TranscriptionError


class BackendError(RuntimeError):
    """A user-presentable failure from either backend."""


class Backend(Protocol):
    label: str

    def process_audio(self, filename: str, data: bytes) -> ProcessAudioResponse: ...
    def ask(self, question: str, top_k: int | None = None, source_file: str | None = None) -> RAGAnswer: ...
    def health(self) -> HealthResponse: ...
    def sources(self) -> list[str]: ...


class LocalBackend:
    """Calls the pipeline directly in this Streamlit process."""

    label = "In-process pipeline"

    def __init__(self, service: VoiceOpsService) -> None:
        self.service = service

    def process_audio(self, filename: str, data: bytes) -> ProcessAudioResponse:
        name = safe_filename(filename)
        with tempfile.TemporaryDirectory(prefix="voiceops_") as tmp_dir:
            path = Path(tmp_dir) / f"{uuid4().hex}{Path(name).suffix.lower()}"
            path.write_bytes(data)
            try:
                return self.service.process_audio(path, source_file=name)
            except (NoSpeechDetectedError, TranscriptionError) as exc:
                raise BackendError(str(exc).replace(str(path), name)) from exc

    def ask(self, question: str, top_k: int | None = None, source_file: str | None = None) -> RAGAnswer:
        try:
            return self.service.ask(AskRequest(question=question, top_k=top_k, source_file=source_file))
        except Exception as exc:  # noqa: BLE001 - e.g. embedding model could not be loaded
            raise BackendError(f"Knowledge base unavailable: {type(exc).__name__}: {exc}") from exc

    def health(self) -> HealthResponse:
        return self.service.health()

    def sources(self) -> list[str]:
        return self.service.list_sources()


class RemoteBackend:
    """Talks to a running FastAPI backend over HTTP."""

    def __init__(self, base_url: str, timeout_seconds: float = 900.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.label = f"API at {self.base_url}"
        # Transcribing a long recording on CPU can take minutes, so the read timeout is generous.
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=10.0))

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise BackendError(f"Cannot reach the VoiceOps API at {self.base_url} ({type(exc).__name__})") from exc
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise BackendError(f"API error {response.status_code}: {detail}")
        return response.json()

    def process_audio(self, filename: str, data: bytes) -> ProcessAudioResponse:
        payload = self._request("POST", "/api/process-audio", files={"file": (filename, data)})
        return ProcessAudioResponse.model_validate(payload)

    def ask(self, question: str, top_k: int | None = None, source_file: str | None = None) -> RAGAnswer:
        body = AskRequest(question=question, top_k=top_k, source_file=source_file).model_dump(exclude_none=True)
        return RAGAnswer.model_validate(self._request("POST", "/api/ask", json=body))

    def health(self) -> HealthResponse:
        return HealthResponse.model_validate(self._request("GET", "/api/health"))

    def sources(self) -> list[str]:
        return list(self._request("GET", "/api/sources"))
