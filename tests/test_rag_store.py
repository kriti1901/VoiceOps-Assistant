"""Chunking, indexing, retrieval and grounded answering (real Chroma in tmp_path, fake local embedder)."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings, get_settings
from app.extractor import LLMError
from app.rag_store import (
    NO_DATA_ANSWER,
    Chunk,
    RagStore,
    SentenceTransformerEmbedder,
    chunk_segments,
    get_default_store,
    index_transcript,
    query_knowledge_base,
)
from app.schemas import Segment, Transcript, format_time_range
from tests.helpers import FakeEmbedder, FakeLLM, make_segments, make_transcript


def seg(start: float, text: str, dur: float = 5.0, speaker: str | None = None) -> Segment:
    return Segment(start_time=start, end_time=start + dur, text=text, speaker=speaker)


def marketing_transcript() -> Transcript:
    return Transcript(
        source_file="marketing_sync.mp3",
        provider="fake",
        segments=[
            seg(0, "Welcome to the quarterly marketing sync, today we review the campaign budget."),
            seg(5, "The agency contract renewal is due and the launch campaign needs more social spend."),
            seg(10, "We agreed to increase the social media budget by ten percent next quarter."),
        ],
    )


@pytest.fixture
def store(make_settings: Callable[..., Settings]) -> RagStore:
    # chunk_max_chars=100 / overlap=0 -> one transcript segment per chunk: easy to assert on
    return RagStore(make_settings(chunk_max_chars=100, chunk_overlap_chars=0), embedder=FakeEmbedder())


# =========================================================================== chunking
def test_chunking_empty_input() -> None:
    assert chunk_segments([], "a.wav") == []


def test_single_chunk_preserves_metadata() -> None:
    chunks = chunk_segments([seg(3, "hello there"), seg(8, "general kenobi")], "call.wav", max_chars=800)
    assert chunks == [Chunk("hello there general kenobi", "call.wav", 3, 13, 0)]


def test_chunks_respect_max_chars_and_cover_every_segment() -> None:
    segments = [seg(i * 5, f"segment number {i:02d} " + "word " * 12) for i in range(30)]  # ~75 chars each
    chunks = chunk_segments(segments, "a.wav", max_chars=300, overlap_chars=80)

    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    joined = " ".join(c.text for c in chunks)
    for s in segments:
        assert s.text in joined  # nothing dropped


def test_consecutive_chunks_overlap_by_whole_segments() -> None:
    segments = [seg(i * 5, f"unique-sentence-{i} " + "pad " * 15) for i in range(10)]  # ~80 chars each
    chunks = chunk_segments(segments, "a.wav", max_chars=260, overlap_chars=100)

    assert len(chunks) > 2
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.start_time < previous.end_time  # time ranges overlap
    # The final segment of each chunk is repeated at the start of the next one.
    for i in range(len(chunks) - 1):
        prev_last_idx = max(j for j in range(10) if f"unique-sentence-{j} " in chunks[i].text)
        assert f"unique-sentence-{prev_last_idx} " in chunks[i + 1].text


def test_zero_overlap_repeats_nothing() -> None:
    segments = [seg(i * 5, f"unique-sentence-{i} " + "pad " * 15) for i in range(10)]
    chunks = chunk_segments(segments, "a.wav", max_chars=260, overlap_chars=0)
    for i in range(10):
        assert sum(f"unique-sentence-{i} " in c.text for c in chunks) == 1


def test_oversized_segments_each_become_their_own_chunk_and_terminate() -> None:
    segments = [seg(i * 10, "x" * 500) for i in range(4)]
    chunks = chunk_segments(segments, "a.wav", max_chars=200, overlap_chars=150)
    assert len(chunks) == 4
    assert [c.start_time for c in chunks] == [0, 10, 20, 30]


def test_no_trailing_chunk_consisting_only_of_overlap() -> None:
    segments = [seg(i * 5, "w" * 90) for i in range(4)]  # exactly two per 200-char chunk
    chunks = chunk_segments(segments, "a.wav", max_chars=200, overlap_chars=100)
    # every chunk must contain at least one segment the previous chunk did not
    seen_end = -1.0
    for chunk in chunks:
        assert chunk.end_time > seen_end
        seen_end = chunk.end_time
    assert chunks[-1].end_time == segments[-1].end_time


def test_chunk_times_are_ordered_and_valid() -> None:
    chunks = chunk_segments(make_segments(), "a.wav", max_chars=160, overlap_chars=60)
    assert all(c.start_time <= c.end_time for c in chunks)
    assert [c.start_time for c in chunks] == sorted(c.start_time for c in chunks)


def test_speaker_labels_are_kept_in_chunk_text() -> None:
    chunks = chunk_segments([seg(0, "we are down", speaker="Speaker A")], "a.wav")
    assert chunks[0].text == "Speaker A: we are down"


# =========================================================================== indexing
def test_index_and_count_and_sources(store: RagStore) -> None:
    assert store.count() == 0
    assert store.index_transcript(make_transcript()) == 5
    assert store.index_transcript(marketing_transcript()) == 3
    assert store.count() == 8
    assert store.list_sources() == ["incident_call.wav", "marketing_sync.mp3"]


def test_empty_transcript_indexes_nothing(store: RagStore) -> None:
    assert store.index_transcript(Transcript(source_file="silence.wav", segments=[])) == 0
    assert store.count() == 0


def test_reindexing_same_source_is_idempotent_and_replaces_old_chunks(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    store.index_transcript(make_transcript())
    assert store.count() == 5

    shorter = Transcript(source_file="incident_call.wav", segments=make_segments()[:2])
    store.index_transcript(shorter)
    assert store.count() == 2  # stale chunks from the earlier version are gone
    assert "database migration" not in " ".join(c.snippet for c in store.retrieve("migration", top_k=5))


def test_index_persists_to_disk_across_instances(make_settings: Callable[..., Settings]) -> None:
    from chromadb.api.shared_system_client import SharedSystemClient

    settings = make_settings(chunk_max_chars=100, chunk_overlap_chars=0)
    RagStore(settings, embedder=FakeEmbedder()).index_transcript(make_transcript())

    SharedSystemClient.clear_system_cache()  # force a cold start: nothing cached in-process
    reopened = RagStore(settings, embedder=FakeEmbedder())
    assert reopened.count() == 5
    assert reopened.list_sources() == ["incident_call.wav"]
    assert (Path(settings.chroma_dir)).exists()


# =========================================================================== retrieval
def test_retrieval_returns_the_relevant_chunk_with_timestamps(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    store.index_transcript(marketing_transcript())

    hits = store.retrieve("which database migration did we forget to run?", top_k=3)

    top = hits[0]
    assert top.source_file == "incident_call.wav"
    assert (top.start_time, top.end_time) == (14.0, 22.5)
    assert "database migration" in top.snippet
    assert top.label == f"incident_call.wav @ {format_time_range(14.0, 22.5)}"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)  # best first
    assert all(-1.0 <= h.score <= 1.0 for h in hits if h.score is not None)


def test_retrieval_routes_to_the_right_recording(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    store.index_transcript(marketing_transcript())
    top = store.retrieve("how much did we raise the social media budget?", top_k=1)[0]
    assert top.source_file == "marketing_sync.mp3"
    assert "ten percent" in top.snippet


def test_source_filter_restricts_results(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    store.index_transcript(marketing_transcript())
    hits = store.retrieve("budget", top_k=5, source_file="incident_call.wav")
    assert hits and {h.source_file for h in hits} == {"incident_call.wav"}
    assert store.retrieve("budget", source_file="does-not-exist.wav") == []


def test_top_k_is_respected_and_clipped_to_collection_size(store: RagStore) -> None:
    store.index_transcript(marketing_transcript())  # 3 chunks
    assert len(store.retrieve("budget", top_k=2)) == 2
    assert len(store.retrieve("budget", top_k=20)) == 3


def test_retrieve_on_empty_store_returns_nothing(store: RagStore) -> None:
    assert store.retrieve("anything") == []


# =========================================================================== answering
def test_empty_store_answers_without_calling_the_llm(store: RagStore) -> None:
    llm = FakeLLM([])
    answer = store.query("what happened?", llm=llm)
    assert answer.answer == NO_DATA_ANSWER
    assert answer.answer_source == "no_data"
    assert answer.citations == []
    assert llm.calls == []


def test_unmatched_source_filter_is_reported_as_no_data(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    llm = FakeLLM([])
    answer = store.query("what happened?", source_file="nope.wav", llm=llm)
    assert answer.answer_source == "no_data"
    assert llm.calls == []


def test_grounded_answer_prompt_carries_excerpts_and_timestamps(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    llm = FakeLLM(["  They forgot the DB migration (incident_call.wav @ 00:14–00:22).  "])

    answer = store.query("what did we forget to run?", top_k=2, llm=llm)

    assert answer.answer_source == "llm"
    assert answer.answer == "They forgot the DB migration (incident_call.wav @ 00:14–00:22)."  # stripped
    assert len(answer.citations) == 2
    assert answer.citations[0].source_file == "incident_call.wav"
    assert (answer.citations[0].start_time, answer.citations[0].end_time) == (14.0, 22.5)

    call = llm.calls[0]
    assert call["json_mode"] is False
    assert "ONLY" in call["system"] and "untrusted" in call["system"]
    assert "Question: what did we forget to run?" in call["user"]
    assert "[1] incident_call.wav @ 00:14–00:22" in call["user"]  # numbered, timestamp-labelled excerpt
    assert "forgot to run the database migration" in call["user"]


def test_without_llm_configured_we_return_extractive_excerpts(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    answer = store.query("database migration")  # no keys in the hermetic env, no llm injected
    assert answer.answer_source == "extractive"
    assert "No LLM API key is configured" in answer.answer
    assert "incident_call.wav @ 00:14–00:22" in answer.answer
    assert answer.citations


def test_llm_failure_degrades_to_extractive_excerpts(store: RagStore) -> None:
    store.index_transcript(make_transcript())
    answer = store.query("database migration", llm=FakeLLM([LLMError("quota exceeded")]))
    assert answer.answer_source == "extractive"
    assert "LLM request failed" in answer.answer
    assert answer.citations


def test_injected_llm_on_the_store_is_used(make_settings: Callable[..., Settings]) -> None:
    llm = FakeLLM(["from the store llm"])
    store = RagStore(make_settings(chunk_max_chars=100, chunk_overlap_chars=0), embedder=FakeEmbedder(), llm=llm)
    store.index_transcript(make_transcript())
    assert store.query("migration").answer == "from the store llm"


# =========================================================================== module-level API
def test_query_knowledge_base_function(store: RagStore) -> None:
    index_transcript(make_transcript(), store=store)
    llm = FakeLLM(["answer text"])
    result = query_knowledge_base("migration?", top_k=1, store=store, llm=llm)
    assert result.answer == "answer text"
    assert len(result.citations) == 1


def test_default_store_is_built_lazily_from_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "from_env"))
    get_settings.cache_clear()
    get_default_store.cache_clear()
    try:
        default = get_default_store()
        assert default.settings.chroma_dir == tmp_path / "from_env"
        assert isinstance(default.embedder, SentenceTransformerEmbedder)
        assert default.embedder._model is None  # constructing the store never loads the model
        assert get_default_store() is default
    finally:
        get_default_store.cache_clear()


# =========================================================================== local embedder
def test_sentence_transformer_embedder_is_lazy_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[str] = []
    encode_kwargs: dict[str, Any] = {}

    class FakeST:
        def __init__(self, name: str) -> None:
            loads.append(name)

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            encode_kwargs.update(kwargs)
            return [[1.0, 0.0] for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    assert loads == []  # nothing loaded at construction
    assert embedder.embed([]) == []
    assert loads == []

    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    embedder.embed(["c"])
    assert loads == ["all-MiniLM-L6-v2"]  # loaded exactly once
    assert encode_kwargs["normalize_embeddings"] is True


def test_embedder_load_failure_explains_the_first_run_download(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom:
        def __init__(self, name: str) -> None:
            raise OSError("connection refused")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = Boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    with pytest.raises(RuntimeError, match="huggingface.co") as info:
        SentenceTransformerEmbedder("all-MiniLM-L6-v2").embed(["hello"])
    assert "connection refused" in str(info.value)
