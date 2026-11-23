"""Speech-to-text providers that all return the same normalized ``Transcript``.

* ``FasterWhisperProvider`` (default, zero cost): local ``faster-whisper`` on CPU with int8
  quantization and VAD filtering. Audio never leaves the machine.
* ``AssemblyAIProvider`` (optional, free cloud tier): enabled with ``STT_PROVIDER=assemblyai``
  plus ``ASSEMBLYAI_API_KEY``. Adds speaker labels.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings, get_settings
from app.schemas import Segment, Transcript

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".ogg")


class TranscriptionError(RuntimeError):
    """Transcription failed (bad audio, model could not be loaded, provider error...)."""


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio_path: str | Path, source_file: str | None = None) -> Transcript: ...


# --------------------------------------------------------------------------- #
# Local: faster-whisper
# --------------------------------------------------------------------------- #

_model_cache: dict[tuple[str, str, str], Any] = {}
_model_lock = threading.Lock()  # guards model loading
_inference_lock = threading.Lock()  # one decode at a time: avoids CPU oversubscription / OOM


def clear_model_cache() -> None:
    """Drop loaded Whisper models (used by tests and for freeing memory)."""
    with _model_lock:
        _model_cache.clear()


class FasterWhisperProvider:
    """Local transcription. The model is downloaded on first use, then cached on disk and in memory."""

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

    def _get_model(self) -> Any:
        key = (self.model_size, self.device, self.compute_type)
        with _model_lock:
            if key not in _model_cache:
                from faster_whisper import (
                    WhisperModel,  # lazy: heavy import, needs weights on first use
                )

                logger.info("Loading faster-whisper model %s (%s, %s)", *key)
                _model_cache[key] = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            return _model_cache[key]

    def transcribe(self, audio_path: str | Path, source_file: str | None = None) -> Transcript:
        path = Path(audio_path)
        try:
            model = self._get_model()
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(
                f"Could not load the faster-whisper '{self.model_size}' model. It is downloaded from huggingface.co "
                f"on first use, so the first run needs network access (or a pre-populated Hugging Face cache): {exc}"
            ) from exc
        try:
            with _inference_lock:
                raw_segments, info = model.transcribe(
                    str(path),
                    language=self.language,
                    beam_size=5,
                    # Trim silence before decoding: faster, and prevents Whisper's classic
                    # "repeat the same sentence forever" hallucination on quiet stretches.
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=False,  # stops one bad window from poisoning the next
                )
                segments = _normalize_whisper_segments(raw_segments)  # generator: decoding happens here
        except Exception as exc:  # noqa: BLE001 - surface a single, readable error type
            raise TranscriptionError(f"Local transcription failed: {exc}") from exc

        return Transcript(
            source_file=source_file or path.name,
            provider=self.name,
            language=getattr(info, "language", None),
            duration=_round(getattr(info, "duration", None)),
            segments=segments,
        )


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _normalize_whisper_segments(raw_segments: Any) -> list[Segment]:
    segments: list[Segment] = []
    for raw in raw_segments:
        text = (raw.text or "").strip()
        if not text:
            continue
        start = max(0.0, float(raw.start))
        end = max(start, float(raw.end))
        segments.append(Segment(start_time=round(start, 2), end_time=round(end, 2), text=text))
    return segments


# --------------------------------------------------------------------------- #
# Cloud: AssemblyAI (free tier)
# --------------------------------------------------------------------------- #


class AssemblyAIProvider:
    """AssemblyAI cloud transcription (utterance-level segments with speaker labels)."""

    name = "assemblyai"

    def __init__(self, api_key: str, language: str | None = None) -> None:
        if not api_key:
            raise ValueError("AssemblyAI API key is required")
        self.api_key = api_key
        self.language = language

    def transcribe(self, audio_path: str | Path, source_file: str | None = None) -> Transcript:
        path = Path(audio_path)
        try:
            import assemblyai as aai  # lazy import

            config_kwargs: dict[str, Any] = {"speaker_labels": True}
            if self.language:
                config_kwargs["language_code"] = self.language
            else:
                config_kwargs["language_detection"] = True

            transcriber = aai.Transcriber(api_key=self.api_key, config=aai.TranscriptionConfig(**config_kwargs))
            result = transcriber.transcribe(str(path))
            if result.status == aai.TranscriptStatus.error:
                raise TranscriptionError(f"AssemblyAI error: {result.error}")
            segments = _segments_from_assemblyai(result)
            language = (result.json_response or {}).get("language_code")
            duration = result.audio_duration
        except TranscriptionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"AssemblyAI transcription failed: {exc}") from exc

        return Transcript(
            source_file=source_file or path.name,
            provider=self.name,
            language=language,
            duration=_round(duration),
            segments=segments,
        )


def _segments_from_assemblyai(result: Any) -> list[Segment]:
    """Utterances (ms) -> Segments (s). Falls back to sentences, then to one whole-file segment."""
    segments: list[Segment] = []
    for utterance in result.utterances or []:
        text = (utterance.text or "").strip()
        if text:
            segments.append(
                Segment(
                    start_time=round(utterance.start / 1000, 2),
                    end_time=round(max(utterance.start, utterance.end) / 1000, 2),
                    text=text,
                    speaker=f"Speaker {utterance.speaker}" if utterance.speaker else None,
                )
            )
    if segments:
        return segments

    try:
        for sentence in result.get_sentences() or []:
            text = (sentence.text or "").strip()
            if text:
                segments.append(
                    Segment(
                        start_time=round(sentence.start / 1000, 2),
                        end_time=round(max(sentence.start, sentence.end) / 1000, 2),
                        text=text,
                    )
                )
    except Exception:  # noqa: BLE001 - sentences are a nicety; fall through to the whole-file segment
        logger.debug("AssemblyAI sentence lookup failed", exc_info=True)
    if segments:
        return segments

    text = (result.text or "").strip()
    if text:
        return [Segment(start_time=0.0, end_time=round(float(result.audio_duration or 0.0), 2), text=text)]
    return []


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_stt_provider(settings: Settings | None = None) -> STTProvider:
    """Pick the provider from settings.

    ``STT_PROVIDER=assemblyai`` without ``ASSEMBLYAI_API_KEY`` falls back to local transcription
    (with a warning) rather than failing, and rather than silently uploading audio anywhere.
    """
    settings = settings or get_settings()
    if settings.stt_provider == "assemblyai":
        if settings.assemblyai_api_key is not None:
            return AssemblyAIProvider(settings.assemblyai_api_key.get_secret_value(), language=settings.stt_language)
        logger.warning("STT_PROVIDER=assemblyai but ASSEMBLYAI_API_KEY is not set; using local faster-whisper")
    return FasterWhisperProvider(
        model_size=settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.stt_language,
    )


def transcribe_audio(
    audio_path: str | Path, source_file: str | None = None, settings: Settings | None = None
) -> Transcript:
    """Convenience wrapper: transcribe with whichever provider the settings select."""
    return get_stt_provider(settings).transcribe(audio_path, source_file=source_file)
