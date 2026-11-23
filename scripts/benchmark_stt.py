#!/usr/bin/env python3
"""Measure how fast and how accurate VoiceOps' local speech-to-text is on YOUR recording and YOUR computer.

Usage (from the project folder, with the virtual environment active):

    python scripts/benchmark_stt.py my_call.wav --reference my_call.txt

* ``--reference`` is a plain-text file with what is *actually said* (no timestamps). It is optional; with it you
  also get the word error rate (WER). Without it you only get speed.
* The first run downloads the Whisper model, so it needs internet once.

The numbers describe this one recording on this one machine; say so when you quote them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Make `from app...` work when this file is run as a script from any folder.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import get_settings  # noqa: E402
from app.stt_engine import FasterWhisperProvider, TranscriptionError  # noqa: E402

# --------------------------------------------------------------------------- #
# Word error rate
# --------------------------------------------------------------------------- #

_QUOTE_VARIANTS = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})
_NOT_A_WORD_CHAR = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_words(text: str) -> list[str]:
    """Lower-case and strip punctuation so ``Hello, World!`` and ``hello world`` count as the same words.

    Hyphens split words ("post-mortem" -> "post mortem"); apostrophes stay ("don't").
    Numbers are *not* converted: "2:05" and "two oh five" are different, so write the reference the way
    Whisper writes numbers (digits), or expect those spots to count as errors.
    """
    cleaned = _NOT_A_WORD_CHAR.sub(" ", text.translate(_QUOTE_VARIANTS).lower().replace("-", " "))
    words = (token.strip("'") for token in cleaned.split())
    return [word for word in words if word]


@dataclass(frozen=True)
class WerResult:
    wer: float  # 0.073 means 7.3% of the reference words were wrong
    reference_words: int
    substitutions: int  # a different word was heard
    deletions: int  # a word was missed
    insertions: int  # an extra word was added

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def word_error_rate(reference: str, hypothesis: str) -> WerResult:
    """Word-level edit distance: WER = (substitutions + deletions + insertions) / reference words."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        raise ValueError("The reference text contains no words")

    # One row of (cost, substitutions, deletions, insertions) per reference word keeps memory small.
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, start=1):
            cost, subs, dels, inss = previous[j - 1]
            substitute = (cost, subs, dels, inss) if ref_word == hyp_word else (cost + 1, subs + 1, dels, inss)
            cost, subs, dels, inss = previous[j]
            delete = (cost + 1, subs, dels + 1, inss)
            cost, subs, dels, inss = current[j - 1]
            insert = (cost + 1, subs, dels, inss + 1)
            current.append(min(substitute, delete, insert, key=lambda cell: cell[0]))
        previous = current

    cost, subs, dels, inss = previous[-1]
    return WerResult(wer=cost / len(ref), reference_words=len(ref), substitutions=subs, deletions=dels, insertions=inss)


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkResult:
    audio_file: str
    audio_seconds: float
    load_seconds: float
    run_seconds: list[float]
    transcribe_seconds: float  # mean of the runs
    speed_multiple: float | None  # 4.0 = four times faster than real time
    real_time_factor: float | None  # 0.25 = takes a quarter of the audio's length
    words: int
    segments: int
    language: str | None
    model_size: str
    device: str
    compute_type: str
    machine: str
    wer: dict[str, Any] | None = field(default=None)


def machine_description() -> str:
    return f"{os.cpu_count() or '?'}-core {platform.system() or 'unknown'} machine"


def benchmark(
    provider: Any,
    audio_path: Path,
    reference_text: str | None = None,
    repeat: int = 1,
    timer: Callable[[], float] = time.perf_counter,
) -> BenchmarkResult:
    """Load the model (timed separately, since it is a one-time cost), then time ``repeat`` transcriptions."""
    started = timer()
    if callable(getattr(provider, "load", None)):
        provider.load()
    load_seconds = timer() - started

    run_seconds: list[float] = []
    transcript = None
    for _ in range(max(1, repeat)):
        started = timer()
        transcript = provider.transcribe(audio_path)
        run_seconds.append(timer() - started)
    assert transcript is not None

    audio_seconds = float(transcript.duration or (transcript.segments[-1].end_time if transcript.segments else 0.0))
    mean_seconds = sum(run_seconds) / len(run_seconds)
    has_speed = audio_seconds > 0 and mean_seconds > 0

    wer = None
    if reference_text is not None:
        wer = asdict(word_error_rate(reference_text, transcript.full_text))

    return BenchmarkResult(
        audio_file=audio_path.name,
        audio_seconds=round(audio_seconds, 2),
        load_seconds=round(load_seconds, 2),
        run_seconds=[round(s, 2) for s in run_seconds],
        transcribe_seconds=round(mean_seconds, 2),
        speed_multiple=round(audio_seconds / mean_seconds, 2) if has_speed else None,
        real_time_factor=round(mean_seconds / audio_seconds, 3) if has_speed else None,
        words=len(normalize_words(transcript.full_text)),
        segments=len(transcript.segments),
        language=transcript.language,
        model_size=getattr(provider, "model_size", "?"),
        device=getattr(provider, "device", "?"),
        compute_type=getattr(provider, "compute_type", "?"),
        machine=machine_description(),
        wer=wer,
    )


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_duration(seconds: float) -> str:
    total = round(seconds)
    minutes, secs = divmod(total, 60)
    return f"{minutes} min {secs} s" if minutes else f"{secs} s"


def speed_phrase(result: BenchmarkResult) -> str:
    if result.speed_multiple is None:
        return "n/a"
    if result.speed_multiple >= 1:
        return f"{result.speed_multiple:.1f}x faster than real time"
    return f"{result.speed_multiple:.2f}x real time (slower than the audio's length)"


def format_report(result: BenchmarkResult) -> str:
    lines = [
        "VoiceOps speech-to-text benchmark",
        f"  audio:          {result.audio_file}  ({format_duration(result.audio_seconds)})",
        f"  setup:          faster-whisper '{result.model_size}' | {result.device} | {result.compute_type}",
        f"  machine:        {result.machine}",
        f"  model load:     {result.load_seconds:.1f} s  (one-time; the very first run also downloads the model)",
        f"  transcription:  {result.transcribe_seconds:.1f} s"
        + (f"  (mean of {len(result.run_seconds)} runs: {', '.join(f'{s:.1f}' for s in result.run_seconds)})" if len(result.run_seconds) > 1 else ""),
        f"  speed:          {speed_phrase(result)}"
        + (f"  (real-time factor {result.real_time_factor})" if result.real_time_factor is not None else ""),
        f"  output:         {result.words} words in {result.segments} segments, language: {result.language or 'unknown'}",
    ]
    if result.wer:
        w = result.wer
        errors = w["substitutions"] + w["deletions"] + w["insertions"]
        lines.append(
            f"  accuracy (WER): {w['wer'] * 100:.1f}%  ({errors} wrong in {w['reference_words']} reference words: "
            f"{w['substitutions']} changed, {w['deletions']} missed, {w['insertions']} extra)"
        )
    else:
        lines.append("  accuracy (WER): not measured (add --reference your_text.txt)")
    if result.words == 0:
        lines.append("  WARNING: no words were recognised. Is the file silent or the wrong format?")
    return "\n".join(lines)


def resume_line(result: BenchmarkResult) -> str:
    setup = f"local faster-whisper ({result.model_size}, {result.compute_type}, {result.device.upper()})"
    line = (
        f"Benchmarked {setup} on a {result.machine}: transcribed {format_duration(result.audio_seconds)} of audio "
        f"in {format_duration(result.transcribe_seconds)}"
    )
    if result.speed_multiple is not None:
        line += f" ({speed_phrase(result)})"
    if result.wer:
        line += f", with a {result.wer['wer'] * 100:.1f}% word error rate on a {result.wer['reference_words']}-word reference"
    return line + "."


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def build_provider(model: str, device: str, compute_type: str, language: str | None) -> FasterWhisperProvider:
    return FasterWhisperProvider(model_size=model, device=device, compute_type=compute_type, language=language)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Measure VoiceOps' speech-to-text speed and accuracy.")
    parser.add_argument("audio", type=Path, help="recording to transcribe (.wav, .mp3, .m4a or .ogg)")
    parser.add_argument("--reference", "-r", type=Path, help="text file with the correct words, to measure accuracy")
    parser.add_argument("--model", default=settings.whisper_model_size, help="Whisper size (default: %(default)s)")
    parser.add_argument("--device", default=settings.whisper_device, help="cpu or cuda (default: %(default)s)")
    parser.add_argument("--compute-type", default=settings.whisper_compute_type, help="default: %(default)s")
    parser.add_argument("--language", default=settings.stt_language, help="language hint such as en (default: auto)")
    parser.add_argument("--repeat", type=int, default=1, help="transcribe N times and average (default: 1)")
    parser.add_argument("--json", type=Path, dest="json_path", help="also save the results to this JSON file")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.audio.is_file():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 2
    reference_text = None
    if args.reference is not None:
        if not args.reference.is_file():
            print(f"Reference file not found: {args.reference}", file=sys.stderr)
            return 2
        reference_text = args.reference.read_text(encoding="utf-8")

    provider = build_provider(args.model, args.device, args.compute_type, args.language)
    print(f"Transcribing {args.audio.name} with '{args.model}' (the first run downloads the model)...", file=sys.stderr)
    try:
        result = benchmark(provider, args.audio, reference_text, repeat=args.repeat)
    except TranscriptionError as exc:
        print(f"Could not run the benchmark: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # e.g. an empty reference file
        print(f"Could not run the benchmark: {exc}", file=sys.stderr)
        return 2

    print()
    print(format_report(result))
    print()
    print("Resume line (check it, and mention it is one recording on one machine):")
    print(f"  {resume_line(result)}")
    if args.json_path:
        args.json_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        print(f"\nSaved the numbers to {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
