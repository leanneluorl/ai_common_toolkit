# -*- coding: utf-8 -*-
"""
agnes_client.py - Client for the Agnes AI chat API (free-tier models only)
==========================================================================
Calls Agnes' OpenAI-compatible Chat Completions API
(https://apihub.agnes-ai.com/v1/chat/completions).

Cost guard: the client is locked to models that are currently free of charge:
agnes-3.0-flash (primary) and agnes-2.5-flash (fallback). agnes-2.5-pro is a
metered, paid model and is deliberately NOT supported. FREE_MODELS is a hard-coded
allow-list; generate() refuses any model name that is not on it, so a typo in the
environment or a wrong argument can never silently hit a billed model.

The "current price $0" advertised for the free models is a promotional price, so
it can change at any time. This module will not quietly keep using a model that
has stopped being free: if the pricing changes, empty or edit FREE_MODELS and
generate() will refuse to call anything until you have re-checked the pricing.

Usage:
    import agnes_client
    text = agnes_client.generate("Write a short paragraph about ...")   # primary model
    text = agnes_client.generate("Write ...", system="You are a careful editor")

Command line (python agnes_client.py "prompt text"):
    Sends a single request and prints the reply; handy for checking that the key
    and the models work.

Environment variables (.env):
    AGNES_API_KEY              required
    AGNES_BASE_URL             API base URL, default https://apihub.agnes-ai.com/v1
    AGNES_MODEL_NAME           primary model, default agnes-3.0-flash
    AGNES_FALLBACK_MODELS      comma-separated fallback models, default agnes-2.5-flash
    AGNES_TIMEOUT_SECONDS      per-request timeout in seconds, default 180
"""

import os
import time

import requests
from dotenv import load_dotenv

_CONFIGURED = False

# ============================================================
# Free-model allow-list: generate() may only call models listed here.
# agnes-2.5-pro is intentionally absent and can never be called.
# ============================================================
FREE_MODELS = ("agnes-3.0-flash", "agnes-2.5-flash")

DEFAULT_MODEL_NAME = "agnes-3.0-flash"       # newest generation
DEFAULT_FALLBACK_MODELS = ["agnes-2.5-flash"]
DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_TIMEOUT = 180.0


class AgnesAPIError(Exception):
    """Raised when the Agnes API fails (HTTP 4xx/5xx, retries exhausted, malformed reply...)."""
    pass


class AgnesPaidModelBlockedError(Exception):
    """Raised when the caller asks for a model that is not on the FREE_MODELS allow-list.
    This is a deliberate safety valve so that a paid model (e.g. agnes-2.5-pro) can
    never be called by accident."""
    pass


def configure(env_path=None, force=False):
    """Load .env and make sure the API key exists. Safe to call repeatedly
    (the real check only runs once unless force=True)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
    if not get_api_key():
        raise EnvironmentError(
            "Environment variable AGNES_API_KEY not found; please set it in your .env "
            "(create a key at https://platform.agnes-ai.com/settings/apiKeys)."
        )
    _CONFIGURED = True


def get_api_key():
    return (os.getenv("AGNES_API_KEY") or "").strip()


def get_base_url():
    return (os.getenv("AGNES_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def get_default_model_name():
    return (os.getenv("AGNES_MODEL_NAME") or DEFAULT_MODEL_NAME).strip()


def get_fallback_model_names(primary=None):
    """Fallback model list (excluding the primary itself). By default
    agnes-3.0-flash and agnes-2.5-flash act as each other's fallback."""
    raw = os.getenv("AGNES_FALLBACK_MODELS")
    names = [n.strip() for n in raw.split(",")] if raw else list(DEFAULT_FALLBACK_MODELS)
    primary = (primary or get_default_model_name()).strip()
    return [n for n in names if n and n != primary]


def get_timeout(explicit=None):
    if explicit is not None:
        return explicit
    raw = os.getenv("AGNES_TIMEOUT_SECONDS")
    try:
        return float(raw) if raw and str(raw).strip() else DEFAULT_TIMEOUT
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers():
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }


def _validate_prompt(prompt):
    """Never let an empty string reach the API (same guard as ai_client.py)."""
    if not prompt or not str(prompt).strip():
        raise ValueError("Refusing to send an empty prompt to the Agnes API; check the upstream prompt source.")


def _assert_free_model(model_name):
    if model_name not in FREE_MODELS:
        raise AgnesPaidModelBlockedError(
            f"Refusing to call '{model_name}': it is not on the free-model allow-list FREE_MODELS={FREE_MODELS}.\n"
            f"   This module is intentionally locked to free models so a paid model "
            f"(e.g. agnes-2.5-pro) is never hit by accident.\n"
            f"   If you have checked Agnes' current pricing and the model is free, edit FREE_MODELS in this file."
        )


def _extract_text(data):
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise AgnesAPIError(f"The Agnes response contains no usable text: {str(data)[:300]}")
    if isinstance(content, str):
        return content
    # Occasionally content is an array of blocks (e.g. mixed with a reasoning block); keep text blocks only.
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def _generate_once(model_name, messages, temperature, max_tokens, timeout, max_retries, label):
    """Send a request to a single model with built-in 429/5xx retries. It does not
    fall back between models (that is generate()'s job).
    Returns (text, ok); ok=False means this model exhausted its retries or hit a
    non-retryable error, so the caller should move on to the next model."""
    payload = {"model": model_name, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature

    url = f"{get_base_url()}/chat/completions"
    delay = 5.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=get_timeout(timeout))
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Agnes {model_name} connection error ({type(e).__name__}); retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = AgnesAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            retry_after = resp.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            print(f"   [wait] Agnes {model_name} returned HTTP {resp.status_code}; retrying in {wait:.0f}s ({attempt}/{max_retries})")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 404:
            print(f"   [skip] Agnes {model_name} returned 404 (model missing or retired); switching to the fallback model.")
            return None, False

        if resp.status_code >= 400:
            raise AgnesAPIError(f"Agnes API error HTTP {resp.status_code}: {resp.text[:300]}")

        latency = time.monotonic() - started
        data = resp.json()
        usage = data.get("usage") or {}
        text = _extract_text(data)
        label_note = f" | {label}" if label else ""
        print(f"   [ok] Agnes {model_name} | {latency:.1f}s | "
              f"tokens in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
              f" | est. cost $0{label_note}")

        finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
        if finish_reason == "length" and not text.strip():
            print(f"   [warn] The reply was cut off by max_tokens ({max_tokens}) before any text was produced. Raise max_tokens and retry.")
        return text, True

    print(f"   [fail] Agnes {model_name} still failing after {max_retries} attempts: {last_error}")
    return None, False


def generate(prompt, model=None, system=None, temperature=None, max_tokens=4096,
             timeout=None, max_retries=3, raise_on_failure=True, label=None,
             allow_model_fallback=True, fallback_models=None):
    """
    Call the Agnes Chat Completions API: a single request with no conversation memory.

    - prompt: the user message text
    - system: optional system prompt
    - model: optional; defaults to AGNES_MODEL_NAME (agnes-3.0-flash)
    - returns: the generated text
    - Only models on the FREE_MODELS allow-list may be called (see the module
      header); any other model raises AgnesPaidModelBlockedError and no HTTP
      request is sent.
    - 429 / 5xx (transient errors) are retried on the same model up to max_retries
      times, honouring Retry-After or exponential backoff; on a 404 (model missing
      or retired) or once retries are exhausted, fallback_models are tried in order
      (default: the other model on the allow-list).
    - raise_on_failure: True (default) raises AgnesAPIError after every model has
      failed; False prints the error and returns an empty string.
    """
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_model_name()).strip()
    _assert_free_model(model_name)

    models_to_try = [model_name]
    if fallback_models is not None:
        for m in fallback_models:
            m = m.strip()
            _assert_free_model(m)
            if m not in models_to_try:
                models_to_try.append(m)
    elif allow_model_fallback:
        for m in get_fallback_model_names(model_name):
            _assert_free_model(m)
            models_to_try.append(m)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for current_model in models_to_try:
        if current_model != models_to_try[0]:
            print(f"   [switch] Using fallback model: {current_model}")
        text, ok = _generate_once(current_model, messages, temperature, max_tokens,
                                  timeout, max_retries, label)
        if ok:
            return text

    msg = f"Agnes: all {len(models_to_try)} free model(s) ({', '.join(models_to_try)}) were tried and failed."
    if raise_on_failure:
        raise AgnesAPIError(msg)
    print(msg)
    return ""


if __name__ == "__main__":
    import sys

    # Some Windows consoles default to a legacy code page that cannot print every character.
    sys.stdout.reconfigure(encoding="utf-8")
    configure()
    prompt_arg = " ".join(sys.argv[1:]) or "Introduce yourself in one sentence."
    print(f"   Agnes primary model: {get_default_model_name()} | fallback: {get_fallback_model_names()}")
    print(f"   Sending: {prompt_arg}\n")
    result = generate(prompt_arg)
    print("\n" + "=" * 78)
    print(result)
    print("=" * 78)
