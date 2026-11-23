"""FastAPI backend.

Run with:  uvicorn app.api:app --reload

* ``GET  /api/health``         - liveness + configuration summary (never loads a model, never leaks keys)
* ``POST /api/process-audio``  - upload audio -> transcribe + extract + index; returns a webhook-ready JSON document
* ``POST /api/ask``            - RAG question answering over every processed recording
* ``GET  /api/sources``        - recordings currently in the knowledge base

NOTE: there is no authentication. Put it behind a reverse proxy / API gateway before exposing it.
"""

from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from app import __version__
from app.config import get_settings
from app.pipeline import NoSpeechDetectedError, VoiceOpsService, safe_filename
from app.schemas import AskRequest, HealthResponse, ProcessAudioResponse, RAGAnswer
from app.stt_engine import SUPPORTED_EXTENSIONS, TranscriptionError

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceOps Assistant API",
    version=__version__,
    description="Turn recorded meetings and incident calls into action items, tickets and a searchable knowledge base.",
)

_CHUNK_BYTES = 1024 * 1024
_MULTIPART_OVERHEAD_BYTES = 1024 * 1024


@lru_cache
def get_service() -> VoiceOpsService:
    """One service instance per process (models and the Chroma client are expensive to create)."""
    return VoiceOpsService(get_settings())


ServiceDep = Annotated[VoiceOpsService, Depends(get_service)]


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health(service: ServiceDep) -> HealthResponse:
    return service.health()


@app.get("/api/sources", response_model=list[str], tags=["knowledge-base"])
def sources(service: ServiceDep) -> list[str]:
    """Names of the recordings that are currently searchable."""
    try:
        return service.list_sources()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not list sources")
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {type(exc).__name__}") from exc


@app.post("/api/process-audio", response_model=ProcessAudioResponse, tags=["pipeline"])
def process_audio(
    request: Request,
    service: ServiceDep,
    file: Annotated[UploadFile, File(description="Audio file: .wav, .mp3, .m4a or .ogg")],
) -> ProcessAudioResponse:
    """Transcribe the recording, extract the summary / action items / ticket, and index it for RAG.

    If no LLM is configured (or it fails) the transcript is still returned and indexed, with an
    explanation in ``warnings`` and ``analysis`` set to ``null``.
    """
    max_bytes = service.settings.max_upload_bytes
    too_large = HTTPException(status_code=413, detail=f"File exceeds the {service.settings.max_upload_mb} MB limit")

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + _MULTIPART_OVERHEAD_BYTES:
        raise too_large

    filename = safe_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix or 'none'}'. Allowed: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # A private temp dir per request; removed on every exit path, including errors.
    with tempfile.TemporaryDirectory(prefix="voiceops_") as tmp_dir:
        destination = Path(tmp_dir) / f"{uuid4().hex}{suffix}"
        size = 0
        with destination.open("wb") as out:
            while chunk := file.file.read(_CHUNK_BYTES):
                size += len(chunk)
                if size > max_bytes:
                    raise too_large
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty")

        try:
            return service.process_audio(destination, source_file=filename)
        except NoSpeechDetectedError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except TranscriptionError as exc:
            logger.warning("Transcription failed for %s: %s", filename, exc)
            detail = str(exc).replace(str(destination), filename)  # don't expose server temp paths
            raise HTTPException(status_code=422, detail=detail) from exc


@app.post("/api/ask", response_model=RAGAnswer, tags=["knowledge-base"])
def ask(body: AskRequest, service: ServiceDep) -> RAGAnswer:
    """Answer a question from the indexed recordings, with timestamped citations."""
    try:
        return service.ask(body)
    except Exception as exc:  # noqa: BLE001 - e.g. embedding model could not be loaded
        logger.exception("RAG query failed")
        raise HTTPException(status_code=503, detail=f"Knowledge base unavailable: {type(exc).__name__}: {exc}") from exc
