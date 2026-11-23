"""Suite-wide fixtures: hermetic env, offline guard, settings factory."""

from __future__ import annotations

import socket
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import Settings, get_settings

_SECRET_ENV_VARS = [
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "ASSEMBLYAI_API_KEY",
    "STT_PROVIDER",
    "LLM_PROVIDER",
    "VOICEOPS_API_URL",
    "WHISPER_MODEL_SIZE",
    "STT_LANGUAGE",
    "CHROMA_DIR",
    "GEMINI_MODEL",
    "GROQ_MODEL",
    "MAX_UPLOAD_MB",
]

_LOOPBACK = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No real keys, no developer ``.env``, and any relative path (./chroma_db) lands in tmp."""
    for name in _SECRET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # Settings reads ".env" relative to cwd -> none here
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to reach a non-loopback host: the suite must run fully offline."""
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> None:
        if isinstance(address, tuple) and address and address[0] not in _LOOPBACK:
            raise AssertionError(f"Test attempted a real network connection to {address!r}")
        return real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Build ``Settings`` ignoring any .env file; the Chroma dir defaults to a tmp path."""

    def _make(**overrides: object) -> Settings:
        overrides.setdefault("chroma_dir", tmp_path / "chroma_db")
        return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]

    return _make
