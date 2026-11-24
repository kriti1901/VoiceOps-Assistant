"""scripts/check_llm.py: finds out why the AI step fails and says what to do (real SDK objects, no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.extractor import GeminiClient, GroqClient, LLMError
from scripts import check_llm

GEMINI_MODELS = [
    genai_types.Model(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
    genai_types.Model(name="models/gemini-3.8-flash-image", supported_actions=["generateContent"]),
    genai_types.Model(name="models/gemini-3.8-pro", supported_actions=["generateContent"]),
    genai_types.Model(name="models/gemini-embedding-001", supported_actions=["embedContent"]),
]


def gemini(monkeypatch: pytest.MonkeyPatch, model: str = "gemini-3.8-flash", reply: Any = "OK") -> GeminiClient:
    client = GeminiClient("fake-key", model=model)
    monkeypatch.setattr(client._client.models, "list", lambda **_: iter(GEMINI_MODELS))
    if isinstance(reply, Exception):

        def reject(**_: Any) -> Any:
            raise reply

        monkeypatch.setattr(client._client.models, "generate_content", reject)
    else:
        monkeypatch.setattr(client._client.models, "generate_content", lambda **_: SimpleNamespace(text=reply))
    return client


# =========================================================================== listing models
def test_gemini_lists_only_models_that_can_generate_text(monkeypatch: pytest.MonkeyPatch) -> None:
    assert gemini(monkeypatch).available_models() == ["gemini-3.8-flash", "gemini-3.8-flash-image", "gemini-3.8-pro"]


def test_groq_lists_model_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GroqClient("fake-key")
    listing = SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b"), SimpleNamespace(id="whisper-large-v3")])
    monkeypatch.setattr(client._client.models, "list", lambda: listing)
    assert client.available_models() == ["openai/gpt-oss-120b", "whisper-large-v3"]


def test_listing_failures_become_llm_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient("fake-key")
    bad_key = genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})

    def reject(**_: Any) -> Any:
        raise bad_key

    monkeypatch.setattr(client._client.models, "list", reject)
    with pytest.raises(LLMError, match="API key not valid"):
        client.available_models()


# =========================================================================== suggestions
def test_suggestions_skip_picture_speech_and_embedding_models() -> None:
    models = ["gemini-3.8-flash", "gemini-3.8-flash-image", "gemini-3.8-pro", "gemini-live-3.8-flash", "gemini-tts-3", "text-embedding-9"]
    assert check_llm.suggest_models(models) == ["gemini-3.8-flash"]  # "flash" text models are preferred


def test_suggestions_fall_back_to_any_text_model_and_are_capped() -> None:
    assert check_llm.suggest_models(["model-b", "model-a"]) == ["model-b", "model-a"]
    assert len(check_llm.suggest_models([f"flash-{i}" for i in range(20)])) == 6


# =========================================================================== the check itself
def test_everything_works(monkeypatch: pytest.MonkeyPatch) -> None:
    report = check_llm.check_client(gemini(monkeypatch))
    text = "\n".join(report.lines)
    assert report.ok and "available to your key (good)" in text and "test request: worked" in text


def test_a_retired_model_is_diagnosed_and_the_fix_is_spelled_out(monkeypatch: pytest.MonkeyPatch) -> None:
    gone = genai_errors.ClientError(404, {"error": {"code": 404, "message": "This model models/gemini-2.5-flash is no longer available to new users.", "status": "NOT_FOUND"}})
    report = check_llm.check_client(gemini(monkeypatch, model="gemini-2.5-flash", reply=gone))
    text = "\n".join(report.lines)

    assert not report.ok
    assert "NOT available to your key" in text and "probably retired" in text
    assert "set  GEMINI_MODEL=gemini-3.8-flash  and restart" in text  # a concrete, copyable fix
    assert "gemini-3.8-flash-image" not in text.split("fix:")[0].split("instead:")[1]  # picture model not suggested
    assert "test request: FAILED" in text and "GEMINI_MODEL" in text


def test_a_bad_key_is_reported_for_both_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GeminiClient("wrong-key")
    bad = genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}})

    def reject(**_: Any) -> Any:
        raise bad

    monkeypatch.setattr(client._client.models, "list", reject)
    monkeypatch.setattr(client._client.models, "generate_content", reject)
    report = check_llm.check_client(client)
    text = "\n".join(report.lines)
    assert not report.ok and "could not list the models" in text and "test request: FAILED" in text
    assert "API key not valid" in text


def test_groq_is_checked_the_same_way(monkeypatch: pytest.MonkeyPatch) -> None:
    client = GroqClient("fake-key")
    monkeypatch.setattr(client._client.models, "list", lambda: SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")]))
    chat = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **_: chat)
    assert check_llm.check_client(client).ok


# =========================================================================== command line
def test_without_any_key_it_explains_what_to_do(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_llm.main() == 1
    out = capsys.readouterr().out
    assert "NOT FOUND" in out and "project folder" in out  # no .env in the (temporary) current folder
    assert "GOOGLE_API_KEY: not set" in out and "No AI key is configured" in out


def test_main_reports_success_and_never_prints_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Any
) -> None:
    (tmp_path / ".env").write_text("GOOGLE_API_KEY=SUPER-SECRET-KEY\n")
    monkeypatch.setenv("GOOGLE_API_KEY", "SUPER-SECRET-KEY")
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(check_llm, "build_llm_clients", lambda settings: [gemini(monkeypatch)])

    assert check_llm.main() == 0
    out = capsys.readouterr().out
    assert "(found)" in out and "GOOGLE_API_KEY: set" in out and "All good" in out
    assert "SUPER-SECRET-KEY" not in out


def test_main_exits_with_1_and_a_next_step_when_the_model_is_retired(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    gone = genai_errors.ClientError(404, {"error": {"code": 404, "message": "no longer available", "status": "NOT_FOUND"}})
    monkeypatch.setattr(check_llm, "build_llm_clients", lambda settings: [gemini(monkeypatch, model="gemini-2.5-flash", reply=gone)])

    assert check_llm.main() == 1
    out = capsys.readouterr().out
    assert "GEMINI_MODEL=gemini-3.8-flash" in out and "need fixing" in out


def test_one_working_service_is_enough(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    from app.config import get_settings

    get_settings.cache_clear()
    bad = genai_errors.ClientError(404, {"error": {"code": 404, "message": "gone", "status": "NOT_FOUND"}})
    broken = gemini(monkeypatch, model="gemini-2.5-flash", reply=bad)
    working = GroqClient("fake-key")
    monkeypatch.setattr(working._client.models, "list", lambda: SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")]))
    chat = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    monkeypatch.setattr(working._client.chat.completions, "create", lambda **_: chat)
    monkeypatch.setattr(check_llm, "build_llm_clients", lambda settings: [broken, working])

    assert check_llm.main() == 0
    assert "At least one AI service works" in capsys.readouterr().out


def test_the_apps_own_default_is_suggested_first_when_available() -> None:
    class Gone:
        name, model = "groq", "llama-3.3-70b-versatile"

        def available_models(self) -> list[str]:
            return ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # alphabetical order would put the weaker 20b first

        def generate(self, *_a: Any, **_k: Any) -> str:
            raise LLMError("model_not_found")

    report = check_llm.check_client(Gone())
    assert "set  GROQ_MODEL=openai/gpt-oss-120b  and restart" in "\n".join(report.lines)
