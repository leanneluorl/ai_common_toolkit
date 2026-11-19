"""agnes_client: the free-model allow-list guard and env-driven configuration."""
import pytest

import agnes_client


def test_paid_model_is_blocked():
    with pytest.raises(agnes_client.AgnesPaidModelBlockedError):
        agnes_client._assert_free_model("agnes-2.5-pro")


def test_free_models_pass_the_guard():
    for name in agnes_client.FREE_MODELS:
        agnes_client._assert_free_model(name)


def test_generate_refuses_paid_model_without_touching_network(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "test-key")
    monkeypatch.setattr(agnes_client, "_CONFIGURED", False)

    def no_network(*a, **k):
        raise AssertionError("network must not be reached")

    monkeypatch.setattr(agnes_client.requests, "post", no_network)
    with pytest.raises(agnes_client.AgnesPaidModelBlockedError):
        agnes_client.generate("hello", model="agnes-2.5-pro")


def test_fallbacks_exclude_the_primary(monkeypatch):
    monkeypatch.setenv("AGNES_FALLBACK_MODELS", "a,b,c")
    assert agnes_client.get_fallback_model_names("b") == ["a", "c"]


def test_timeout_parsing(monkeypatch):
    assert agnes_client.get_timeout() == agnes_client.DEFAULT_TIMEOUT
    monkeypatch.setenv("AGNES_TIMEOUT_SECONDS", "12.5")
    assert agnes_client.get_timeout() == 12.5
    monkeypatch.setenv("AGNES_TIMEOUT_SECONDS", "not-a-number")
    assert agnes_client.get_timeout() == agnes_client.DEFAULT_TIMEOUT
    assert agnes_client.get_timeout(7) == 7
