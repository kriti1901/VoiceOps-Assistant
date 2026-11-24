"""Settings defaults, .env parsing and the shipped .env.example."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, get_settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"


def test_defaults_match_the_spec() -> None:
    s = Settings(_env_file=None)
    assert s.stt_provider == "local"
    assert (s.whisper_model_size, s.whisper_device, s.whisper_compute_type) == ("base", "cpu", "int8")
    assert s.gemini_model == "gemini-3.8-flash"
    assert s.embedding_model == "all-MiniLM-L6-v2"
    assert s.chroma_dir == Path("./chroma_db")
    assert s.llm_provider == "auto"
    assert not (s.has_gemini or s.has_groq or s.has_assemblyai)


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STT_PROVIDER", "assemblyai")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    s = Settings(_env_file=None)
    assert s.stt_provider == "assemblyai" and s.has_assemblyai and s.has_gemini
    assert s.max_upload_bytes == 5 * 1024 * 1024


def test_dotenv_file_is_read_and_blank_values_mean_unset(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=abc\nGROQ_API_KEY=\nSTT_LANGUAGE=\nVOICEOPS_API_URL=\n")
    s = Settings(_env_file=env_file)
    assert s.google_api_key is not None and s.google_api_key.get_secret_value() == "abc"
    assert s.groq_api_key is None
    assert s.stt_language is None and s.voiceops_api_url is None


def test_invalid_values_fail_fast() -> None:
    with pytest.raises(ValueError):
        Settings(_env_file=None, stt_provider="whisperx")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Settings(_env_file=None, rag_top_k=0)


def test_get_settings_is_cached_until_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("RAG_TOP_K", "9")
    get_settings.cache_clear()
    assert get_settings().rag_top_k == 9


def test_env_example_parses_and_matches_code_defaults() -> None:
    """Guards against documentation drift: the shipped example must load and agree with the defaults."""
    shipped = Settings(_env_file=ENV_EXAMPLE)
    defaults = Settings(_env_file=None)
    assert shipped.model_dump() == defaults.model_dump()  # blank keys == unset, inline comments stripped
