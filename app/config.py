"""Application settings, loaded from environment variables / a local ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# Groq retires models regularly; override with GROQ_MODEL. (llama-3.3-70b-versatile is no longer served.)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class Settings(BaseSettings):
    """All runtime configuration. Every field can be set via an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Speech-to-text -------------------------------------------------
    stt_provider: Literal["local", "assemblyai"] = "local"
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    stt_language: str | None = None  # None -> auto-detect
    assemblyai_api_key: SecretStr | None = None

    # --- LLM (free tiers) ------------------------------------------------
    # "auto": Gemini first (if GOOGLE_API_KEY), Groq as fallback (if GROQ_API_KEY).
    llm_provider: Literal["auto", "gemini", "groq"] = "auto"
    google_api_key: SecretStr | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    groq_api_key: SecretStr | None = None
    groq_model: str = DEFAULT_GROQ_MODEL
    llm_timeout_seconds: float = 120.0

    # --- RAG -------------------------------------------------------------
    chroma_dir: Path = Path("./chroma_db")
    chroma_collection: str = "voiceops_transcripts"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_max_chars: int = Field(default=800, ge=100)
    chunk_overlap_chars: int = Field(default=150, ge=0)
    rag_top_k: int = Field(default=4, ge=1, le=20)

    # --- App -------------------------------------------------------------
    max_upload_mb: int = Field(default=100, ge=1)
    # When set, the Streamlit UI talks to a running FastAPI backend over HTTP
    # instead of calling the pipeline in-process.
    voiceops_api_url: str | None = None

    @field_validator("voiceops_api_url", "stt_language", mode="before")
    @classmethod
    def _blank_to_none(cls, value: object) -> object:
        """Treat ``KEY=`` (empty) in a .env file as unset."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("assemblyai_api_key", "google_api_key", "groq_api_key", mode="before")
    @classmethod
    def _blank_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_gemini(self) -> bool:
        return self.google_api_key is not None

    @property
    def has_groq(self) -> bool:
        return self.groq_api_key is not None

    @property
    def has_assemblyai(self) -> bool:
        return self.assemblyai_api_key is not None


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()
