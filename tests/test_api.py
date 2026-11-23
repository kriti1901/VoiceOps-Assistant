"""FastAPI endpoints + the pipeline service, with fake STT / LLM / embedder (no models, no network)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import app, get_service
from app.config import Settings
from app.pipeline import NoSpeechDetectedError, VoiceOpsService
from app.rag_store import RagStore
from app.schemas import AskRequest, Transcript
from app.stt_engine import TranscriptionError
from tests.helpers import FakeEmbedder, FakeLLM, make_transcript, valid_analysis_json

AUDIO_BYTES = b"RIFF....WAVEfmt fake-audio-bytes"


class FakeSTT:
    """Records what it was given, and checks the uploaded bytes really were on disk at call time."""

    name = "fake-stt"

    def __init__(self, transcript: Transcript | None = None, error: Exception | None = None) -> None:
        self.transcript = transcript
        self.error = error
        self.seen_path: Path | None = None
        self.seen_bytes: bytes | None = None
        self.seen_source: str | None = None

    def transcribe(self, audio_path: str | Path, source_file: str | None = None) -> Transcript:
        self.seen_path = Path(audio_path)
        # HTTP tests upload a real file (so it must exist now); service-level tests pass dummy paths.
        self.seen_bytes = self.seen_path.read_bytes() if self.seen_path.exists() else None
        self.seen_source = source_file
        if self.error:
            raise self.error
        base = self.transcript or make_transcript()
        return base.model_copy(update={"source_file": source_file or base.source_file})


@dataclass
class Env:
    client: TestClient
    service: VoiceOpsService
    stt: FakeSTT
    llm: FakeLLM | None
    store: RagStore
    settings: Settings
    extras: dict[str, Any] = field(default_factory=dict)


class ExplodingEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding model unavailable")


@pytest.fixture
def make_env(make_settings: Callable[..., Settings]) -> Iterator[Callable[..., Env]]:
    clients: list[TestClient] = []

    def _make(
        *,
        llm_responses: list[str | Exception] | None = None,
        with_llm: bool = True,
        stt: FakeSTT | None = None,
        embedder: Any = None,
        **settings_overrides: Any,
    ) -> Env:
        settings = make_settings(chunk_max_chars=100, chunk_overlap_chars=0, **settings_overrides)
        llm = FakeLLM(llm_responses if llm_responses is not None else [valid_analysis_json()]) if with_llm else None
        stt = stt or FakeSTT()
        store = RagStore(settings, embedder=embedder or FakeEmbedder())
        service = VoiceOpsService(settings, stt=stt, llm=llm, store=store)
        app.dependency_overrides[get_service] = lambda: service
        client = TestClient(app)
        clients.append(client)
        return Env(client, service, stt, llm, store, settings)

    yield _make
    app.dependency_overrides.clear()


def upload(env: Env, name: str = "incident_call.wav", data: bytes = AUDIO_BYTES) -> Any:
    return env.client.post("/api/process-audio", files={"file": (name, data, "application/octet-stream")})


# =========================================================================== health
def test_health_reports_configuration_without_leaking_secrets(make_env: Callable[..., Env]) -> None:
    env = make_env(google_api_key="SUPER-SECRET-VALUE", groq_api_key="ANOTHER-SECRET")
    response = env.client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["stt_provider"] == "fake-stt"
    assert body["llm_configured"] is True
    assert body["llm_provider"] == "gemini\u2192groq"
    assert body["indexed_chunks"] == 0
    assert "SUPER-SECRET-VALUE" not in response.text and "ANOTHER-SECRET" not in response.text


def test_health_without_llm_key(make_env: Callable[..., Env]) -> None:
    body = make_env(with_llm=False).client.get("/api/health").json()
    assert body["llm_configured"] is False and body["llm_provider"] is None


def test_health_never_raises_even_if_the_store_is_broken(make_env: Callable[..., Env], monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env()
    monkeypatch.setattr(RagStore, "count", lambda self: (_ for _ in ()).throw(RuntimeError("chroma down")))
    response = env.client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["indexed_chunks"] is None


# =========================================================================== process-audio: happy path
def test_process_audio_end_to_end(make_env: Callable[..., Env]) -> None:
    env = make_env()

    response = upload(env, name="Incident Call.WAV")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["event"] == "voiceops.meeting.processed"
    assert body["source_file"] == "Incident Call.WAV"
    assert body["analysis"]["sentiment"] == "escalated"
    assert body["analysis"]["incident_ticket"]["severity"] == "critical"
    assert body["analysis"]["action_items"][0]["owner"] == "Priya"
    assert body["transcript"]["segments"][0] == {
        "start_time": 0.0,
        "end_time": 6.2,
        "text": "Okay everyone, the checkout API is returning 500 errors since the deploy.",
        "speaker": None,
    }
    assert body["indexed_chunks"] == 5 and body["warnings"] == []

    # the upload really reached the STT engine, under a server-generated name with the right suffix
    assert env.stt.seen_bytes == AUDIO_BYTES
    assert env.stt.seen_path is not None and env.stt.seen_path.suffix == ".wav"
    assert env.stt.seen_path.name != "Incident Call.WAV"
    assert env.store.count() == 5


def test_temp_upload_is_always_deleted(make_env: Callable[..., Env]) -> None:
    env = make_env()
    assert upload(env).status_code == 200
    assert env.stt.seen_path is not None
    assert not env.stt.seen_path.exists()
    assert not env.stt.seen_path.parent.exists()  # the per-request temp dir is gone too

    failing = make_env(stt=FakeSTT(error=TranscriptionError("boom")))
    assert upload(failing).status_code == 422
    assert failing.stt.seen_path is not None and not failing.stt.seen_path.exists()


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".m4a", ".ogg"])
def test_all_supported_extensions_are_accepted(make_env: Callable[..., Env], ext: str) -> None:
    assert upload(make_env(), name=f"call{ext}").status_code == 200


def test_process_then_ask_roundtrip(make_env: Callable[..., Env]) -> None:
    env = make_env(llm_responses=[valid_analysis_json(), "They forgot the migration (incident_call.wav @ 00:14\u201300:22)."])
    assert upload(env).status_code == 200

    response = env.client.post("/api/ask", json={"question": "What did they forget to run?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer_source"] == "llm"
    assert "migration" in body["answer"]
    assert len(body["citations"]) == 2
    assert body["citations"][0]["source_file"] == "incident_call.wav"
    assert body["citations"][0]["start_time"] == 14.0
    assert env.client.get("/api/sources").json() == ["incident_call.wav"]


# =========================================================================== process-audio: validation
@pytest.mark.parametrize("name", ["notes.txt", "malware.exe", "noextension", "audio.wav.sh", "x.flac"])
def test_unsupported_file_types_are_rejected(make_env: Callable[..., Env], name: str) -> None:
    env = make_env()
    response = upload(env, name=name)
    assert response.status_code == 415
    assert env.stt.seen_path is None  # never reached the engine


def test_oversized_upload_is_rejected(make_env: Callable[..., Env]) -> None:
    env = make_env(max_upload_mb=1)
    response = upload(env, data=b"0" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert env.stt.seen_path is None
    assert upload(env, data=b"0" * 1024).status_code == 200  # under the limit is fine


def test_empty_upload_is_rejected(make_env: Callable[..., Env]) -> None:
    assert upload(make_env(), data=b"").status_code == 400


def test_missing_file_field_is_a_validation_error(make_env: Callable[..., Env]) -> None:
    assert make_env().client.post("/api/process-audio").status_code == 422


@pytest.mark.parametrize(
    ("hostile", "expected"),
    [("../../etc/passwd.wav", "passwd.wav"), ("C:\\evil\\dir\\x.wav", "x.wav"), ("..\\..\\boot.mp3", "boot.mp3"), ("/abs/path/c.ogg", "c.ogg")],
)
def test_filenames_are_sanitized_against_path_traversal(make_env: Callable[..., Env], hostile: str, expected: str) -> None:
    env = make_env()
    response = upload(env, name=hostile)
    assert response.status_code == 200
    assert response.json()["source_file"] == expected
    assert env.stt.seen_source == expected


# =========================================================================== process-audio: failure handling
def test_no_speech_detected_is_422(make_env: Callable[..., Env]) -> None:
    env = make_env(stt=FakeSTT(transcript=Transcript(source_file="silence.wav", segments=[])))
    response = upload(env, name="silence.wav")
    assert response.status_code == 422
    assert "No speech" in response.json()["detail"]
    assert env.store.count() == 0


def test_transcription_error_is_422_and_hides_server_paths(make_env: Callable[..., Env]) -> None:
    stt = FakeSTT()

    def transcribe_and_fail(audio_path: str | Path, source_file: str | None = None) -> Transcript:
        raise TranscriptionError(f"Local transcription failed: cannot decode '{audio_path}'")

    stt.transcribe = transcribe_and_fail  # type: ignore[method-assign]
    detail = upload(make_env(stt=stt), name="broken.mp3")
    assert detail.status_code == 422
    assert "cannot decode 'broken.mp3'" in detail.json()["detail"]
    assert "voiceops_" not in detail.json()["detail"]


def test_without_an_llm_key_we_still_return_and_index_the_transcript(make_env: Callable[..., Env]) -> None:
    env = make_env(with_llm=False)
    response = upload(env)
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"] is None
    assert any("GOOGLE_API_KEY" in w for w in body["warnings"])
    assert len(body["transcript"]["segments"]) == 5
    assert body["indexed_chunks"] == 5


def test_extraction_failure_degrades_gracefully(make_env: Callable[..., Env]) -> None:
    env = make_env(llm_responses=["not json", "still not json"])
    body = upload(env).json()
    assert body["analysis"] is None
    assert any("Structured extraction failed" in w for w in body["warnings"])
    assert body["indexed_chunks"] == 5  # the expensive transcript is not thrown away


def test_indexing_failure_degrades_gracefully(make_env: Callable[..., Env]) -> None:
    env = make_env(embedder=ExplodingEmbedder())
    response = upload(env)
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["sentiment"] == "escalated"
    assert body["indexed_chunks"] == 0
    assert any("Indexing" in w and "embedding model unavailable" in w for w in body["warnings"])


# =========================================================================== ask
def test_ask_on_empty_knowledge_base(make_env: Callable[..., Env]) -> None:
    response = make_env().client.post("/api/ask", json={"question": "anything?"})
    assert response.status_code == 200
    assert response.json()["answer_source"] == "no_data"


def test_ask_without_llm_returns_excerpts(make_env: Callable[..., Env]) -> None:
    env = make_env(with_llm=False)
    upload(env)
    body = env.client.post("/api/ask", json={"question": "database migration"}).json()
    assert body["answer_source"] == "extractive"
    assert body["citations"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "q", "top_k": 0}, {"question": "q", "top_k": 99}])
def test_ask_validates_input(make_env: Callable[..., Env], payload: dict[str, Any]) -> None:
    assert make_env().client.post("/api/ask", json=payload).status_code == 422


def test_ask_reports_503_if_the_knowledge_base_is_unavailable(make_env: Callable[..., Env]) -> None:
    env = make_env(embedder=ExplodingEmbedder())
    env.store.collection.upsert(  # something must be indexed so the query reaches the embedder
        ids=["x"], embeddings=[[0.1] * 256], documents=["d"], metadatas=[{"source_file": "a.wav", "start_time": 0.0, "end_time": 1.0, "chunk_index": 0}]
    )
    response = env.client.post("/api/ask", json={"question": "hello"})
    assert response.status_code == 503
    assert "Knowledge base unavailable" in response.json()["detail"]


# =========================================================================== service-level + schema
def test_service_raises_no_speech_error(make_settings: Callable[..., Settings]) -> None:
    service = VoiceOpsService(
        make_settings(),
        stt=FakeSTT(transcript=Transcript(source_file="s.wav", segments=[])),
        llm=FakeLLM([]),
        store=RagStore(make_settings(), embedder=FakeEmbedder()),
    )
    with pytest.raises(NoSpeechDetectedError):
        service.process_audio("whatever.wav")


def test_service_ask_passes_filters_through(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(chunk_max_chars=100, chunk_overlap_chars=0)
    llm = FakeLLM([valid_analysis_json(), "ok"])
    service = VoiceOpsService(settings, stt=FakeSTT(), llm=llm, store=RagStore(settings, embedder=FakeEmbedder()))
    assert service.process_audio("a.wav", "a.wav").indexed_chunks == 5
    answer = service.ask(AskRequest(question="migration", top_k=1, source_file="a.wav"))
    assert len(answer.citations) == 1 and answer.citations[0].source_file == "a.wav"


def test_openapi_exposes_the_required_endpoints(make_env: Callable[..., Env]) -> None:
    paths = make_env().client.get("/openapi.json").json()["paths"]
    assert "get" in paths["/api/health"]
    assert "post" in paths["/api/process-audio"]
    assert "post" in paths["/api/ask"]
