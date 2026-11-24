#!/usr/bin/env python3
"""Check that your AI key and model work, and say in plain words what to fix if they don't.

Usage (from the project folder, with the virtual environment active):

    python scripts/check_llm.py

It reads your `.env`, asks each configured AI service which models YOUR key can use, and sends one tiny test
request. It never prints your key. AI companies retire models regularly, so if the model in your settings is
gone, this tells you which ones to use instead.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make `from app...` work when this file is run as a script from any folder.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL, get_settings  # noqa: E402
from app.extractor import LLMError, build_llm_clients  # noqa: E402

_ENV_VARS = {"gemini": "GEMINI_MODEL", "groq": "GROQ_MODEL"}
_APP_DEFAULTS = {"gemini": DEFAULT_GEMINI_MODEL, "groq": DEFAULT_GROQ_MODEL}
# Models that can generate text but are not what this app needs (pictures, speech, live audio, embeddings...).
_NOT_FOR_TEXT = ("image", "tts", "live", "audio", "embedding", "robotics", "computer-use", "guard", "whisper", "orpheus")
_MAX_SUGGESTIONS = 6


@dataclass
class ProviderReport:
    provider: str
    model: str
    ok: bool
    lines: list[str]


def suggest_models(models: list[str]) -> list[str]:
    """Models that look suitable for this app: text models, newest-looking names first."""
    text_models = [m for m in models if not any(word in m.lower() for word in _NOT_FOR_TEXT)]
    preferred = [m for m in text_models if "flash" in m.lower() or "gpt-oss" in m.lower()] or text_models
    return sorted(preferred, reverse=True)[:_MAX_SUGGESTIONS]


def check_client(client: Any) -> ProviderReport:
    """Two checks: is the configured model offered to this key, and does a tiny request work?"""
    provider, model = client.name, client.model
    env_var = _ENV_VARS.get(provider, "the model setting")
    lines: list[str] = []
    ok = True

    try:
        models = client.available_models()
    except LLMError as exc:
        ok = False
        lines.append(f"  could not list the models for your key: {exc}")
    else:
        if model in models:
            lines.append(f"  model '{model}': available to your key (good)")
        else:
            ok = False
            lines.append(f"  model '{model}': NOT available to your key. It was probably retired.")
            suggestions = suggest_models(models)
            default = _APP_DEFAULTS.get(provider)
            if default in suggestions:  # the app's own tested default comes first when your key can use it
                suggestions.remove(default)
                suggestions.insert(0, default)
            if suggestions:
                lines.append(f"  models that should work instead: {', '.join(suggestions)}")
                lines.append(f"  fix: in your .env file set  {env_var}={suggestions[0]}  and restart the app")
            else:
                lines.append("  your key can't see any text models; check the key at the provider's website")

    try:
        reply = client.generate("This is a connection test.", "Reply with the single word OK.")
    except LLMError as exc:
        ok = False
        lines.append(f"  test request: FAILED. {exc}")
    else:
        lines.append(f"  test request: worked (the AI replied {reply.strip()[:40]!r})")
    return ProviderReport(provider=provider, model=model, ok=ok, lines=lines)


def main() -> int:
    settings = get_settings()
    env_file = Path(".env").resolve()
    print(f"Looking for settings in: {env_file}  ({'found' if env_file.is_file() else 'NOT FOUND'})")
    if not env_file.is_file():
        print("  Run this command from the project folder, the one that contains your .env file.")
    print(f"  GOOGLE_API_KEY: {'set' if settings.has_gemini else 'not set'}")
    print(f"  GROQ_API_KEY:   {'set' if settings.has_groq else 'not set'}")

    clients = build_llm_clients(settings)
    if not clients:
        print("\nNo AI key is configured, so there is nothing to check.")
        print("Add a free key to .env: GOOGLE_API_KEY (https://aistudio.google.com/apikey)")
        print("or GROQ_API_KEY (https://console.groq.com/keys), then run this again.")
        return 1

    reports = []
    for client in clients:
        print(f"\n{client.name.capitalize()}:")
        report = check_client(client)
        print("\n".join(report.lines))
        reports.append(report)

    print()
    if all(report.ok for report in reports):
        print("All good: your AI settings work. Start the app with:  streamlit run app/ui.py")
        return 0
    if any(report.ok for report in reports):
        print("At least one AI service works, so the app will use it. Fix the other one above when you can.")
        return 0
    print("The AI settings need fixing (see above). Then run this check again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
