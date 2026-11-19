"""ai_router: provider resolution and exception unification (providers are mocked)."""
import pytest

import ai_client
import ai_router
import groq_client


@pytest.fixture(autouse=True)
def reset_runtime_choice():
    ai_router.set_runtime_choice(None, None)
    yield
    ai_router.set_runtime_choice(None, None)


def test_default_provider_is_gemini():
    assert ai_router.get_provider() == "gemini"


def test_env_selects_provider(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "Groq")
    assert ai_router.get_provider() == "groq"


def test_argument_beats_env(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "groq")
    assert ai_router.get_provider("gemini") == "gemini"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError):
        ai_router.get_provider("openai")


def test_runtime_choice_applies_process_wide(monkeypatch):
    ai_router.set_runtime_choice("groq", "my-model")
    assert ai_router.get_provider() == "groq"
    assert ai_router._resolve_model("groq", None) == "my-model"
    assert ai_router._resolve_model("gemini", None) is None
    assert ai_router._resolve_model("groq", "explicit") == "explicit"


def test_groq_route_returns_text_and_used_model(monkeypatch):
    monkeypatch.setattr(groq_client, "generate", lambda *a, **k: '{"ok": true}')
    monkeypatch.setattr(groq_client, "last_used_model", lambda: "fallback-model")
    text, used = ai_router.generate_json_text("hi", provider="groq")
    assert text == '{"ok": true}' and used == "fallback-model"


@pytest.mark.parametrize("raised, expected", [
    (groq_client.GroqAPIError("quota"), ai_router.AIQuotaExhaustedError),
    (groq_client.GroqRequestTooLargeError("big"), ai_router.AIRequestTooLargeError),
    (groq_client.GroqRequestFailedError("bad"), ai_router.AIRequestFailedError),
])
def test_groq_errors_are_unified(monkeypatch, raised, expected):
    def boom(*a, **k):
        raise raised

    monkeypatch.setattr(groq_client, "generate", boom)
    with pytest.raises(expected):
        ai_router.generate_json_text("hi", provider="groq")
    with pytest.raises(expected):
        ai_router.generate_text("hi", provider="groq")


def test_gemini_quota_errors_are_unified(monkeypatch):
    def boom(*a, **k):
        raise ai_client.AllApiKeysExhaustedError("today is used up")

    monkeypatch.setattr(ai_client, "generate", boom)
    monkeypatch.setattr(ai_client, "usage_snapshot", lambda: {"owners": []})
    with pytest.raises(ai_router.AIQuotaExhaustedError):
        ai_router.generate_json_text("hi", provider="gemini")


def test_gemini_route_prepends_system_and_requests_json(monkeypatch):
    seen = {}

    def fake_generate(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["cfg"] = kwargs["generation_config"]
        return "{}"

    monkeypatch.setattr(ai_client, "generate", fake_generate)
    monkeypatch.setattr(ai_client, "usage_snapshot", lambda: {"owners": []})
    monkeypatch.setattr(ai_client, "get_default_model_name", lambda: "m")
    text, used = ai_router.generate_json_text("body", provider="gemini", system="be terse", temperature=0.2)
    assert seen["prompt"].startswith("be terse") and seen["prompt"].endswith("body")
    assert seen["cfg"]["response_mime_type"] == "application/json"
    assert seen["cfg"]["temperature"] == 0.2
    assert used == "m"
