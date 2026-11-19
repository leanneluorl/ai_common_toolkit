"""Shared test setup: make the toolkit importable and keep every test offline and side-effect free."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Modules that did `from dotenv import load_dotenv` hold their own reference to it.
_MODULES_WITH_DOTENV = (
    "ai_client", "groq_client", "agnes_client", "agnes_vision_client",
    "aliyun_vision_client", "qwen_client", "cogvideox_client", "trend_search",
)


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Strip real credentials, block .env loading and redirect runtime files to a temp dir."""
    for name in list(os.environ):
        if name.startswith(("GEMINI_", "GOOGLE_API_KEY", "GROQ_", "AGNES_", "QWEN_", "DASHSCOPE_",
                            "COGVIDEOX_", "AI_PROVIDER")):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_USAGE_LOG_PATH", str(tmp_path / "usage.json"))
    monkeypatch.setenv("GEMINI_USAGE_CSV_PATH", "")
    monkeypatch.setenv("GEMINI_MODEL_REGISTRY_PATH", str(tmp_path / "models.json"))

    # Never let a developer's local .env leak into a test run (and never reach a real API).
    no_dotenv = lambda *a, **k: False  # noqa: E731
    monkeypatch.setattr("dotenv.load_dotenv", no_dotenv)
    for name in _MODULES_WITH_DOTENV:
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(module, "load_dotenv"):
            monkeypatch.setattr(module, "load_dotenv", no_dotenv)
    yield
