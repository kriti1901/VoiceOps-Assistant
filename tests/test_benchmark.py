"""The speech-to-text benchmark script: word-error-rate maths, timing and reporting (fake model, fake clock)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from app.schemas import Transcript
from app.stt_engine import TranscriptionError, clear_model_cache
from scripts import benchmark_stt as bench
from tests.helpers import make_transcript


# =========================================================================== normalising words
def test_normalize_ignores_case_and_punctuation() -> None:
    assert bench.normalize_words("Hello, World! It's 2:05 PM.") == ["hello", "world", "it's", "2", "05", "pm"]


def test_normalize_splits_hyphens_and_handles_curly_apostrophes_and_quotes() -> None:
    assert bench.normalize_words("post-mortem for ‘Priya’ and don’t") == [
        "post",
        "mortem",
        "for",
        "priya",
        "and",
        "don't",
    ]


def test_normalize_empty_and_punctuation_only() -> None:
    assert bench.normalize_words("") == []
    assert bench.normalize_words(" ... --- ?! ") == []


# =========================================================================== word error rate (hand-checkable)
def test_identical_text_has_zero_errors_even_if_case_and_punctuation_differ() -> None:
    result = bench.word_error_rate("The cat sat.", "the CAT, sat")
    assert result.wer == 0.0 and result.errors == 0 and result.reference_words == 3


def test_one_wrong_word_in_ten_is_ten_percent() -> None:
    reference = "one two three four five six seven eight nine ten"
    hypothesis = "one two three four five six seven eight nine eleven"
    result = bench.word_error_rate(reference, hypothesis)
    assert result.wer == pytest.approx(0.10)
    assert (result.substitutions, result.deletions, result.insertions) == (1, 0, 0)


def test_textbook_example_one_changed_and_one_missed_word() -> None:
    # "sat" heard as "sit" (changed) and the second "the" missed (deleted): 2 errors in 6 words = 33.3%
    result = bench.word_error_rate("the cat sat on the mat", "the cat sit on mat")
    assert result.wer == pytest.approx(2 / 6)
    assert (result.substitutions, result.deletions, result.insertions) == (1, 1, 0)


def test_extra_words_count_as_insertions() -> None:
    result = bench.word_error_rate("good morning team", "good morning to the team")
    assert (result.substitutions, result.deletions, result.insertions) == (0, 0, 2)
    assert result.wer == pytest.approx(2 / 3)


def test_nothing_heard_is_one_hundred_percent_deletions() -> None:
    result = bench.word_error_rate("a b c d", "")
    assert result.wer == 1.0 and result.deletions == 4 and result.substitutions == 0 and result.insertions == 0


def test_completely_different_text_of_same_length_is_all_substitutions() -> None:
    result = bench.word_error_rate("a b c", "x y z")
    assert result.wer == 1.0 and result.substitutions == 3


def test_error_rate_can_exceed_one_hundred_percent_when_many_words_are_invented() -> None:
    result = bench.word_error_rate("hi", "hi there how are you today")
    assert result.wer == pytest.approx(5.0)  # 5 extra words against a 1-word reference


def test_error_counts_always_add_up_to_the_edit_distance() -> None:
    reference = "please roll back the checkout service to version two point two now"
    hypothesis = "please rollback the check out service to version 2.2 right now"
    result = bench.word_error_rate(reference, hypothesis)
    assert result.errors == round(result.wer * result.reference_words)


def test_empty_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="no words"):
        bench.word_error_rate("   ...  ", "anything")


# =========================================================================== timing and speed (fake clock)
class FakeProvider:
    model_size, device, compute_type = "base", "cpu", "int8"

    def __init__(self, transcript: Transcript | None = None, error: Exception | None = None) -> None:
        self.transcript = transcript or make_transcript()  # 41.0 seconds of audio, 5 segments
        self.error = error
        self.loaded = 0
        self.transcribed: list[Path] = []

    def load(self) -> None:
        self.loaded += 1

    def transcribe(self, audio_path: Path, source_file: str | None = None) -> Transcript:
        if self.error:
            raise self.error
        self.transcribed.append(audio_path)
        return self.transcript


def fake_clock(*values: float) -> Any:
    ticks = iter(values)
    return lambda: next(ticks)


def test_speed_and_real_time_factor_are_computed_from_the_clock(tmp_path: Path) -> None:
    provider = FakeProvider()
    # clock: model load takes 2 s (0 -> 2); the single transcription takes 10 s (2 -> 12)
    result = bench.benchmark(provider, tmp_path / "call.wav", timer=fake_clock(0, 2, 2, 12))

    assert provider.loaded == 1 and provider.transcribed == [tmp_path / "call.wav"]
    assert (result.audio_seconds, result.load_seconds, result.transcribe_seconds) == (41.0, 2.0, 10.0)
    assert result.speed_multiple == 4.1  # 41 s of audio in 10 s
    assert result.real_time_factor == 0.244
    assert result.words == len(bench.normalize_words(make_transcript().full_text))
    assert (result.segments, result.language, result.wer) == (5, "en", None)


def test_repeat_averages_the_runs(tmp_path: Path) -> None:
    provider = FakeProvider()
    # load: 0 -> 1; run 1: 1 -> 11 (10 s); run 2: 11 -> 17 (6 s); mean 8 s
    result = bench.benchmark(provider, tmp_path / "a.wav", repeat=2, timer=fake_clock(0, 1, 1, 11, 11, 17))
    assert result.run_seconds == [10.0, 6.0] and result.transcribe_seconds == 8.0
    assert len(provider.transcribed) == 2


def test_slower_than_real_time_is_described_honestly(tmp_path: Path) -> None:
    result = bench.benchmark(FakeProvider(), tmp_path / "a.wav", timer=fake_clock(0, 0, 0, 82))  # 82 s for 41 s of audio
    assert result.speed_multiple == 0.5
    assert "slower than the audio's length" in bench.speed_phrase(result)
    assert "faster than real time" not in bench.speed_phrase(result)


def test_accuracy_is_measured_when_a_reference_is_given(tmp_path: Path) -> None:
    reference = make_transcript().full_text.replace("Okay everyone", "Okay team")  # exactly one word changed
    result = bench.benchmark(FakeProvider(), tmp_path / "a.wav", reference_text=reference, timer=fake_clock(0, 0, 0, 5))
    assert result.wer is not None
    assert result.wer["substitutions"] == 1 and result.wer["deletions"] == 0 and result.wer["insertions"] == 0
    assert result.wer["wer"] == pytest.approx(1 / result.wer["reference_words"])


def test_missing_duration_falls_back_to_the_last_segment_and_silence_does_not_crash(tmp_path: Path) -> None:
    no_duration = make_transcript().model_copy(update={"duration": None})
    assert bench.benchmark(FakeProvider(no_duration), tmp_path / "a.wav", timer=fake_clock(0, 0, 0, 10)).audio_seconds == 41.0

    silent = Transcript(source_file="s.wav", segments=[], duration=None)
    result = bench.benchmark(FakeProvider(silent), tmp_path / "s.wav", timer=fake_clock(0, 0, 0, 3))
    assert result.speed_multiple is None and result.words == 0
    assert "no words were recognised" in bench.format_report(result)


# =========================================================================== report text
def _result(tmp_path: Path, **kwargs: Any) -> bench.BenchmarkResult:
    return bench.benchmark(FakeProvider(), tmp_path / "incident_call.wav", timer=fake_clock(0, 2, 2, 12), **kwargs)


def test_format_duration() -> None:
    assert bench.format_duration(9.4) == "9 s"
    assert bench.format_duration(252.3) == "4 min 12 s"


def test_report_shows_every_number_the_user_needs(tmp_path: Path) -> None:
    reference = make_transcript().full_text.replace("Okay everyone", "Okay team")
    report = bench.format_report(_result(tmp_path, reference_text=reference))
    assert "incident_call.wav" in report and "(41 s)" in report
    assert "faster-whisper 'base' | cpu | int8" in report
    assert "4.1x faster than real time" in report and "real-time factor 0.244" in report
    assert "accuracy (WER):" in report and "1 changed, 0 missed, 0 extra" in report


def test_report_explains_how_to_get_accuracy_when_no_reference(tmp_path: Path) -> None:
    assert "not measured (add --reference" in bench.format_report(_result(tmp_path))


def test_resume_line_uses_only_measured_values(tmp_path: Path) -> None:
    reference = make_transcript().full_text.replace("Okay everyone", "Okay team")
    line = bench.resume_line(_result(tmp_path, reference_text=reference))
    assert "faster-whisper (base, int8, CPU)" in line
    assert "transcribed 41 s of audio in 10 s (4.1x faster than real time)" in line
    assert "word error rate on a " in line and "-word reference" in line
    assert line.endswith(".")
    assert "word error rate" not in bench.resume_line(_result(tmp_path))  # no reference -> no accuracy claim


# =========================================================================== command line
@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "call.wav"
    path.write_bytes(b"not real audio; the provider is faked")
    return path


def use_fake_provider(monkeypatch: pytest.MonkeyPatch, provider: FakeProvider) -> list[tuple[Any, ...]]:
    seen: list[tuple[Any, ...]] = []

    def build(model: str, device: str, compute_type: str, language: str | None) -> FakeProvider:
        seen.append((model, device, compute_type, language))
        return provider

    monkeypatch.setattr(bench, "build_provider", build)
    return seen


def test_cli_prints_report_and_resume_line(monkeypatch: pytest.MonkeyPatch, audio_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    seen = use_fake_provider(monkeypatch, FakeProvider())
    assert bench.main([str(audio_file)]) == 0
    out = capsys.readouterr().out
    assert "VoiceOps speech-to-text benchmark" in out and "Resume line" in out and "Benchmarked local faster-whisper" in out
    assert seen == [("base", "cpu", "int8", None)]  # defaults come from the app's own settings


def test_cli_flags_reach_the_provider_and_json_is_saved(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path, tmp_path: Path
) -> None:
    seen = use_fake_provider(monkeypatch, FakeProvider())
    reference = tmp_path / "ref.txt"
    reference.write_text(make_transcript().full_text, encoding="utf-8")  # a perfect reference
    out_json = tmp_path / "result.json"

    code = bench.main([str(audio_file), "-r", str(reference), "--model", "small", "--language", "en", "--json", str(out_json)])

    assert code == 0 and seen == [("small", "cpu", "int8", "en")]
    saved = json.loads(out_json.read_text())
    assert saved["wer"]["wer"] == 0.0 and saved["audio_seconds"] == 41.0


def test_cli_reports_missing_files_with_exit_code_2(audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert bench.main([str(tmp_path / "nope.wav")]) == 2
    assert "Audio file not found" in capsys.readouterr().err
    assert bench.main([str(audio_file), "--reference", str(tmp_path / "nope.txt")]) == 2
    assert "Reference file not found" in capsys.readouterr().err


def test_cli_rejects_an_empty_reference_file(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_provider(monkeypatch, FakeProvider())
    reference = tmp_path / "empty.txt"
    reference.write_text("  ...  ")
    assert bench.main([str(audio_file), "-r", str(reference)]) == 2
    assert "no words" in capsys.readouterr().err


def test_cli_turns_model_errors_into_a_friendly_message(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_provider(monkeypatch, FakeProvider(error=TranscriptionError("Could not load the model (huggingface.co)")))
    assert bench.main([str(audio_file)]) == 1
    err = capsys.readouterr().err
    assert "Could not run the benchmark" in err and "huggingface.co" in err
    assert "Traceback" not in err



def test_cli_with_the_real_provider_class_shows_a_friendly_message_when_the_model_cannot_load(
    monkeypatch: pytest.MonkeyPatch, audio_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: the model-load failure used to escape as a raw traceback because only the fake was tested."""
    module = types.ModuleType("faster_whisper")

    def cannot_download(*_: Any, **__: Any) -> None:
        raise OSError("403 Forbidden")

    module.WhisperModel = cannot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    clear_model_cache()
    try:
        assert bench.main([str(audio_file)]) == 1  # real build_provider -> real FasterWhisperProvider
    finally:
        clear_model_cache()
    err = capsys.readouterr().err
    assert "Could not run the benchmark" in err and "huggingface.co" in err and "403 Forbidden" in err
    assert "Traceback" not in err
