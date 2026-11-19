"""groq_client: retry / fallback behaviour against a mocked HTTP layer (no network)."""
import pytest
import requests

import groq_client


class FakeResponse:
    def __init__(self, status=200, body=None, text="", headers=None):
        self.status_code = status
        self._body = body
        self.text = text or (str(body) if body is not None else "")
        self.headers = headers or {}

    def json(self):
        return self._body


def ok_body(text="hello"):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


@pytest.fixture(autouse=True)
def groq_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(groq_client, "_CONFIGURED", False)
    monkeypatch.setattr(groq_client.time, "sleep", lambda s: None)


def install(monkeypatch, handler):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return handler(json, len(calls))

    monkeypatch.setattr(groq_client.requests, "post", fake_post)
    return calls


def test_success_returns_text_and_records_model(monkeypatch):
    calls = install(monkeypatch, lambda payload, n: FakeResponse(200, ok_body("hi there")))
    assert groq_client.generate("say hi") == "hi there"
    assert calls[0]["model"] == groq_client.DEFAULT_MODEL_NAME
    assert groq_client.last_used_model() == groq_client.DEFAULT_MODEL_NAME


def test_empty_prompt_is_rejected_before_any_request(monkeypatch):
    calls = install(monkeypatch, lambda payload, n: FakeResponse(200, ok_body()))
    with pytest.raises(ValueError):
        groq_client.generate("   ")
    assert calls == []


def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        groq_client.generate("hello")


def test_long_retry_after_switches_to_fallback_model(monkeypatch):
    def handler(payload, n):
        if payload["model"] == groq_client.DEFAULT_MODEL_NAME:
            return FakeResponse(429, text="daily limit", headers={"retry-after": "5000"})
        return FakeResponse(200, ok_body("from fallback"))

    install(monkeypatch, handler)
    assert groq_client.generate("hello") == "from fallback"
    assert groq_client.last_used_model() == groq_client.DEFAULT_FALLBACK_MODELS[0]


def test_short_429_is_retried_on_same_model(monkeypatch):
    def handler(payload, n):
        if n == 1:
            return FakeResponse(429, text="slow down", headers={"retry-after": "1"})
        return FakeResponse(200, ok_body("second try"))

    calls = install(monkeypatch, handler)
    assert groq_client.generate("hello") == "second try"
    assert {c["model"] for c in calls} == {groq_client.DEFAULT_MODEL_NAME}


def test_all_models_exhausted_raises_api_error(monkeypatch):
    install(monkeypatch, lambda p, n: FakeResponse(429, text="quota", headers={"retry-after": "9999"}))
    with pytest.raises(groq_client.GroqAPIError):
        groq_client.generate("hello")


def test_request_too_large_without_fallback_raises_specific_error(monkeypatch):
    calls = install(monkeypatch, lambda p, n: FakeResponse(413, text="Request too large"))
    with pytest.raises(groq_client.GroqRequestTooLargeError):
        groq_client.generate("hello", fallback_on_too_large=False)
    assert len(calls) == 1


def test_request_too_large_everywhere_raises_too_large(monkeypatch):
    install(monkeypatch, lambda p, n: FakeResponse(413, text="Request too large"))
    with pytest.raises(groq_client.GroqRequestTooLargeError):
        groq_client.generate("hello")


def test_json_mode_drops_unsupported_response_format(monkeypatch):
    def handler(payload, n):
        if "response_format" in payload:
            return FakeResponse(400, text="response_format is not supported")
        return FakeResponse(200, ok_body('{"ok": true}'))

    calls = install(monkeypatch, handler)
    assert groq_client.generate("give me JSON", json_mode=True) == '{"ok": true}'
    assert len(calls) == 2 and "response_format" not in calls[1]


def test_json_mode_adds_json_hint_when_prompt_lacks_it(monkeypatch):
    calls = install(monkeypatch, lambda p, n: FakeResponse(200, ok_body("{}")))
    groq_client.generate("list three colours", json_mode=True)
    assert any("json" in m["content"].lower() for m in calls[0]["messages"] if m["role"] == "system")


def test_connection_error_is_retried(monkeypatch):
    state = {"n": 0}

    def fake_post(*a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise requests.ConnectionError("boom")
        return FakeResponse(200, ok_body("recovered"))

    monkeypatch.setattr(groq_client.requests, "post", fake_post)
    assert groq_client.generate("hello") == "recovered"


def test_extract_text_handles_block_content():
    data = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
    assert groq_client._extract_text(data) == "ab"
    with pytest.raises(groq_client.GroqAPIError):
        groq_client._extract_text({})


def test_fallback_list_excludes_primary(monkeypatch):
    monkeypatch.setenv("GROQ_FALLBACK_MODELS", "a, b ,c")
    assert groq_client.get_fallback_model_names("b") == ["a", "c"]
