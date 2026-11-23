"""STT providers, exercised with fake ``faster_whisper`` / ``assemblyai`` modules (no weights, no network)."""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import Settings
from app.stt_engine import (
    SUPPORTED_EXTENSIONS,
    AssemblyAIProvider,
    FasterWhisperProvider,
    TranscriptionError,
    clear_model_cache,
    get_stt_provider,
)


@pytest.fixture(autouse=True)
def _fresh_model_cache() -> None:
    clear_model_cache()
    yield
    clear_model_cache()


# --------------------------------------------------------------------------- fake faster_whisper
class FakeWhisperModel:
    instances: list[FakeWhisperModel] = []

    def __init__(self, size: str, device: str = "auto", compute_type: str = "default") -> None:
        self.init_args = (size, device, compute_type)
        self.transcribe_calls: list[dict[str, Any]] = []
        self.raw_segments: list[Any] = []
        self.error: Exception | None = None
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio: str, **kwargs: Any) -> tuple[Any, Any]:
        self.transcribe_calls.append({"audio": audio, **kwargs})
        if self.error:
            raise self.error
        return iter(self.raw_segments), SimpleNamespace(language="en", duration=41.234)


def _seg(start: float, end: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(start=start, end=end, text=text)


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch) -> type[FakeWhisperModel]:
    FakeWhisperModel.instances = []
    module = types.ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeWhisperModel


def test_supported_extensions() -> None:
    assert set(SUPPORTED_EXTENSIONS) == {".wav", ".mp3", ".m4a", ".ogg"}


def test_local_provider_defaults_to_base_cpu_int8_with_vad(fake_whisper: type[FakeWhisperModel]) -> None:
    provider = get_stt_provider(Settings(_env_file=None))
    assert isinstance(provider, FasterWhisperProvider)

    provider.transcribe("/tmp/meeting.wav")

    model = fake_whisper.instances[0]
    assert model.init_args == ("base", "cpu", "int8")
    call = model.transcribe_calls[0]
    assert call["vad_filter"] is True  # trims silence + prevents repetition hallucinations
    assert call["condition_on_previous_text"] is False
    assert call["audio"] == "/tmp/meeting.wav"


def test_local_provider_normalizes_segments(fake_whisper: type[FakeWhisperModel]) -> None:
    provider = FasterWhisperProvider()
    provider._get_model()  # create the fake model so we can script its output
    fake_whisper.instances[0].raw_segments = [
        _seg(0.0, 3.14159, "  Hello team  "),
        _seg(3.2, 3.9, "   "),  # blank -> dropped
        _seg(4.0, 3.5, "End before start"),  # clamped
        _seg(-1.0, 2.0, "Negative start"),  # clamped
    ]

    transcript = provider.transcribe("a.mp3", source_file="standup.mp3")

    assert transcript.source_file == "standup.mp3"
    assert transcript.provider == "faster-whisper"
    assert transcript.language == "en"
    assert transcript.duration == 41.23
    assert [(s.start_time, s.end_time, s.text) for s in transcript.segments] == [
        (0.0, 3.14, "Hello team"),
        (4.0, 4.0, "End before start"),
        (0.0, 2.0, "Negative start"),
    ]


def test_local_provider_source_file_defaults_to_filename(fake_whisper: type[FakeWhisperModel]) -> None:
    transcript = FasterWhisperProvider().transcribe("/some/dir/call.ogg")
    assert transcript.source_file == "call.ogg"


def test_model_is_loaded_once_and_reused(fake_whisper: type[FakeWhisperModel]) -> None:
    FasterWhisperProvider().transcribe("a.wav")
    FasterWhisperProvider().transcribe("b.wav")
    assert len(fake_whisper.instances) == 1
    assert len(fake_whisper.instances[0].transcribe_calls) == 2


def test_language_setting_is_forwarded(fake_whisper: type[FakeWhisperModel]) -> None:
    FasterWhisperProvider(language="de").transcribe("a.wav")
    assert fake_whisper.instances[0].transcribe_calls[0]["language"] == "de"


def test_local_failures_become_transcription_error(fake_whisper: type[FakeWhisperModel]) -> None:
    provider = FasterWhisperProvider()
    provider._get_model()
    fake_whisper.instances[0].error = RuntimeError("corrupt audio")
    with pytest.raises(TranscriptionError, match="corrupt audio"):
        provider.transcribe("bad.wav")


def test_model_load_failure_becomes_transcription_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("faster_whisper")

    def boom(*_: Any, **__: Any) -> None:
        raise OSError("cannot download model")

    module.WhisperModel = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    with pytest.raises(TranscriptionError) as info:
        FasterWhisperProvider().transcribe("a.wav")
    message = str(info.value)
    assert "cannot download model" in message  # original cause is preserved
    assert "huggingface.co" in message and "first use" in message  # ...plus an actionable hint


# --------------------------------------------------------------------------- provider selection
def test_assemblyai_selected_when_key_present(make_settings: Callable[..., Settings]) -> None:
    provider = get_stt_provider(make_settings(stt_provider="assemblyai", assemblyai_api_key="aai-secret"))
    assert isinstance(provider, AssemblyAIProvider)
    assert provider.api_key == "aai-secret"


def test_assemblyai_without_key_falls_back_to_local(
    make_settings: Callable[..., Settings], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        provider = get_stt_provider(make_settings(stt_provider="assemblyai"))
    assert isinstance(provider, FasterWhisperProvider)
    assert "ASSEMBLYAI_API_KEY" in caplog.text


def test_blank_key_in_env_counts_as_unset(make_settings: Callable[..., Settings]) -> None:
    provider = get_stt_provider(make_settings(stt_provider="assemblyai", assemblyai_api_key="  "))
    assert isinstance(provider, FasterWhisperProvider)


def test_local_provider_honours_model_settings(make_settings: Callable[..., Settings]) -> None:
    provider = get_stt_provider(make_settings(whisper_model_size="small", stt_language="fr"))
    assert isinstance(provider, FasterWhisperProvider)
    assert (provider.model_size, provider.device, provider.compute_type, provider.language) == (
        "small",
        "cpu",
        "int8",
        "fr",
    )


def test_secret_never_appears_in_settings_repr(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(assemblyai_api_key="aai-secret", google_api_key="g-secret", groq_api_key="q-secret")
    rendered = repr(settings) + str(settings.model_dump())
    assert "aai-secret" not in rendered and "g-secret" not in rendered and "q-secret" not in rendered


# --------------------------------------------------------------------------- fake assemblyai
class FakeTranscriber:
    last: FakeTranscriber | None = None
    result: Any = None

    def __init__(self, api_key: str | None = None, config: Any = None) -> None:
        self.api_key, self.config = api_key, config
        self.audio: str | None = None
        FakeTranscriber.last = self

    def transcribe(self, audio: str) -> Any:
        self.audio = audio
        return FakeTranscriber.result


@pytest.fixture
def fake_aai(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("assemblyai")
    module.Transcriber = FakeTranscriber  # type: ignore[attr-defined]
    module.TranscriptionConfig = lambda **kwargs: SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    module.TranscriptStatus = SimpleNamespace(completed="completed", error="error")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "assemblyai", module)
    FakeTranscriber.last = None
    return module


def _aai_result(**kwargs: Any) -> SimpleNamespace:
    defaults: dict[str, Any] = {
        "status": "completed",
        "error": None,
        "utterances": [],
        "text": "",
        "audio_duration": 12.0,
        "json_response": {"language_code": "en_us"},
        "get_sentences": lambda: [],
    }
    return SimpleNamespace(**{**defaults, **kwargs})


def test_assemblyai_maps_utterances_to_normalized_segments(fake_aai: types.ModuleType) -> None:
    FakeTranscriber.result = _aai_result(
        utterances=[
            SimpleNamespace(start=0, end=4500, text="Is the API down?", speaker="A"),
            SimpleNamespace(start=4500, end=9120, text="  Yes, since the deploy.  ", speaker="B"),
            SimpleNamespace(start=9120, end=9200, text="  ", speaker="A"),
        ]
    )

    transcript = AssemblyAIProvider("key-123").transcribe("/tmp/call.mp3", source_file="call.mp3")

    assert FakeTranscriber.last is not None
    assert FakeTranscriber.last.api_key == "key-123"
    assert FakeTranscriber.last.config.speaker_labels is True
    assert FakeTranscriber.last.config.language_detection is True
    assert FakeTranscriber.last.audio == "/tmp/call.mp3"
    assert transcript.provider == "assemblyai"
    assert transcript.language == "en_us"
    assert transcript.duration == 12.0
    assert [(s.start_time, s.end_time, s.text, s.speaker) for s in transcript.segments] == [
        (0.0, 4.5, "Is the API down?", "Speaker A"),
        (4.5, 9.12, "Yes, since the deploy.", "Speaker B"),
    ]


def test_assemblyai_explicit_language_disables_detection(fake_aai: types.ModuleType) -> None:
    FakeTranscriber.result = _aai_result(utterances=[SimpleNamespace(start=0, end=10, text="hola", speaker="A")])
    AssemblyAIProvider("k", language="es").transcribe("a.wav")
    assert FakeTranscriber.last is not None
    assert FakeTranscriber.last.config.language_code == "es"
    assert not hasattr(FakeTranscriber.last.config, "language_detection")


def test_assemblyai_error_status_raises(fake_aai: types.ModuleType) -> None:
    FakeTranscriber.result = _aai_result(status="error", error="Audio file is too short")
    with pytest.raises(TranscriptionError, match="too short"):
        AssemblyAIProvider("k").transcribe("a.wav")


def test_assemblyai_falls_back_to_sentences_then_whole_text(fake_aai: types.ModuleType) -> None:
    FakeTranscriber.result = _aai_result(
        get_sentences=lambda: [SimpleNamespace(start=1000, end=2500, text="One sentence.")]
    )
    transcript = AssemblyAIProvider("k").transcribe("a.wav")
    assert [(s.start_time, s.end_time, s.text) for s in transcript.segments] == [(1.0, 2.5, "One sentence.")]

    FakeTranscriber.result = _aai_result(text="Whole file text", audio_duration=30.0)
    transcript = AssemblyAIProvider("k").transcribe("a.wav")
    assert [(s.start_time, s.end_time, s.text) for s in transcript.segments] == [(0.0, 30.0, "Whole file text")]


def test_assemblyai_sdk_exception_is_wrapped(fake_aai: types.ModuleType) -> None:
    class Exploding(FakeTranscriber):
        def transcribe(self, audio: str) -> Any:
            raise ConnectionError("network down")

    fake_aai.Transcriber = Exploding  # type: ignore[attr-defined]
    with pytest.raises(TranscriptionError, match="network down"):
        AssemblyAIProvider("k").transcribe("a.wav")


def test_assemblyai_requires_key() -> None:
    with pytest.raises(ValueError):
        AssemblyAIProvider("")
