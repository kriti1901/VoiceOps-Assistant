"""Real-library compatibility checks for the local STT stack (no model weights, no network).

Everything else in the suite mocks ``faster_whisper``. These tests deliberately do NOT: they exercise the real
audio decoder and VAD so a bad dependency combination is caught here instead of on a user's first upload.

Background: faster-whisper 1.2.x calls ``av.open(..., metadata_errors=...)``, which PyAV 19 removed. An unbounded
``av`` requirement therefore installs a set in which every transcription fails while decoding audio, and mocked
tests cannot see that. requirements.txt caps ``av<19``; these tests keep that honest.
"""

from __future__ import annotations

import inspect
import wave
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("faster_whisper")

from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402
from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: E402

RATE = 16_000
SECONDS = 2


def _tone() -> np.ndarray:
    return (np.sin(2 * np.pi * 440 * np.arange(RATE * SECONDS) / RATE) * 0.3 * 32767).astype(np.int16)


def _write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(_tone().tobytes())


def _write_with_pyav(path: Path, codecs: tuple[str, ...]) -> None:
    """Encode the tone with the first available PyAV encoder (skips if the build has none)."""
    last_error: Exception | None = None
    for codec in codecs:
        try:
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream(codec, rate=RATE)
                stream.layout = "mono"
                resampler = av.AudioResampler(format=stream.codec_context.format.name, layout="mono", rate=RATE)
                frame = av.AudioFrame.from_ndarray(_tone().reshape(1, -1), format="s16", layout="mono")
                frame.sample_rate = RATE
                for out in resampler.resample(frame) + resampler.resample(None):
                    for packet in stream.encode(out):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            return
        except Exception as exc:  # noqa: BLE001 - encoder not compiled into this PyAV build
            last_error = exc
            path.unlink(missing_ok=True)
    pytest.skip(f"no PyAV encoder available for {path.suffix}: {last_error}")


@pytest.mark.parametrize(
    ("suffix", "codecs"),
    [
        (".wav", ()),
        (".mp3", ("libmp3lame",)),
        (".m4a", ("aac",)),
        (".ogg", ("libopus", "libvorbis")),
    ],
)
def test_faster_whisper_decodes_every_supported_format(tmp_path: Path, suffix: str, codecs: tuple[str, ...]) -> None:
    path = tmp_path / f"tone{suffix}"
    _write_wav(path) if suffix == ".wav" else _write_with_pyav(path, codecs)

    audio = decode_audio(str(path), sampling_rate=RATE)

    assert audio.dtype == np.float32
    assert abs(len(audio) / RATE - SECONDS) < 0.25  # lossy codecs add a little padding
    assert float(np.abs(audio).max()) > 0.1  # real signal came out, not silence


def test_silero_vad_runs_with_the_options_we_pass() -> None:
    """The bundled VAD model + onnxruntime load, and our ``vad_parameters`` dict is a valid ``VadOptions``."""
    options = VadOptions(**{"min_silence_duration_ms": 500})
    silence = np.zeros(RATE * 2, dtype=np.float32)
    assert get_speech_timestamps(silence, options) == []  # nothing to transcribe in pure silence


def test_whisper_transcribe_still_accepts_the_arguments_we_use() -> None:
    parameters = inspect.signature(WhisperModel.transcribe).parameters
    for name in ("language", "beam_size", "vad_filter", "vad_parameters", "condition_on_previous_text"):
        assert name in parameters, f"faster-whisper's transcribe() dropped '{name}'"
    init = inspect.signature(WhisperModel.__init__).parameters
    assert {"device", "compute_type"} <= set(init)
