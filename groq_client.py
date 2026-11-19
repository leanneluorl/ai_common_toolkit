# -*- coding: utf-8 -*-
"""
groq_client.py - Client for the Groq (GroqCloud) free-tier API
==============================================================
Calls Groq's OpenAI-compatible Chat Completions API
(https://api.groq.com/openai/v1/chat/completions).
Note: this is Groq (groq.com, the LPU inference cloud), not xAI's Grok.

Groq has a free plan: no credit card is needed, and going over the limits never
bills you - it just returns HTTP 429. Limits are RPM / RPD / TPM / TPD per model and
are applied at the organisation level (keys in the same organisation share them).
See https://console.groq.com/settings/limits for the current numbers (they change
with policy, so they are not hard-coded here).

Because limits are per model, the fallback strategy of this module is:
  - Short 429s (per-minute limits): wait for Retry-After and retry the same model.
  - 429s with a very long wait (usually a daily RPD/TPD cap, Retry-After in the
    thousands of seconds): do not sit and wait - switch straight to a fallback
    model (a different model means a different quota pool).

Usage:
    import groq_client
    text = groq_client.generate("Write a short paragraph about ...")
    text = groq_client.generate("Write ...", system="You are a careful editor")

Command line (python groq_client.py "prompt text"):
    Sends a single request and prints the reply; handy for checking that the key
    and the models work.

Environment variables (.env):
    GROQ_API_KEY               required (create a key at https://console.groq.com/keys)
    GROQ_BASE_URL              API base URL, default https://api.groq.com/openai/v1
    GROQ_MODEL_NAME            primary model, default openai/gpt-oss-120b
    GROQ_FALLBACK_MODELS       comma-separated fallback models,
                               default openai/gpt-oss-20b,qwen/qwen3.8-27b
    GROQ_TIMEOUT_SECONDS       per-request timeout in seconds, default 180
    GROQ_MAX_WAIT_SECONDS      longest wait accepted on a 429 before switching to a
                               fallback model, default 90
"""

import os
import time

import requests
from dotenv import load_dotenv

_CONFIGURED = False

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_FALLBACK_MODELS = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_WAIT = 90.0


class GroqAPIError(Exception):
    """Raised when the Groq API fails (HTTP 4xx/5xx, retries exhausted, malformed reply...)."""
    pass


class GroqRequestFailedError(GroqAPIError):
    """Every model failed, but not because of quota (request too large, the model could
    not produce valid JSON, the model was retired... possibly a mix). Only this request
    is affected: re-running tomorrow may not help, but other requests can still be
    processed today."""
    pass


class GroqRequestTooLargeError(GroqAPIError):
    """Every model returned 413: the request itself exceeds the model's per-request token
    limit (e.g. the free tier's TPM cap of 8000). Unlike quota exhaustion, waiting until
    tomorrow will not help - the request has to be shrunk (shorter transcript / persona)
    or sent to a model or plan with a higher limit."""
    pass


# Why the last _generate_once attempt failed (lets generate() tell "request too large" apart).
_LAST_FAILURE = {"reason": "", "detail": ""}
# The model that served the most recent successful request (when a fallback took over, the
# caller relies on this to record the model that was really used).
_LAST_USED = {"model": ""}


def last_used_model():
    return _LAST_USED["model"]


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
            "Environment variable GROQ_API_KEY not found; please set it in your .env "
            "(create a key at https://console.groq.com/keys)."
        )
    _CONFIGURED = True


def get_api_key():
    return (os.getenv("GROQ_API_KEY") or "").strip()


def get_base_url():
    return (os.getenv("GROQ_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def get_default_model_name():
    return (os.getenv("GROQ_MODEL_NAME") or DEFAULT_MODEL_NAME).strip()


def get_fallback_model_names(primary=None):
    """Fallback model list (excluding the primary itself)."""
    raw = os.getenv("GROQ_FALLBACK_MODELS")
    names = [n.strip() for n in raw.split(",")] if raw is not None else list(DEFAULT_FALLBACK_MODELS)
    primary = (primary or get_default_model_name()).strip()
    return [n for n in names if n and n != primary]


def _env_float(name, default):
    raw = os.getenv(name)
    try:
        return float(raw) if raw and str(raw).strip() else default
    except ValueError:
        return default


def get_timeout(explicit=None):
    if explicit is not None:
        return explicit
    return _env_float("GROQ_TIMEOUT_SECONDS", DEFAULT_TIMEOUT)


def _headers():
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }


def _validate_prompt(prompt):
    """Never let an empty string reach the API (same guard as ai_client.py)."""
    if not prompt or not str(prompt).strip():
        raise ValueError("Refusing to send an empty prompt to the Groq API; check the upstream prompt source.")


def _extract_text(data):
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise GroqAPIError(f"The Groq response contains no usable text: {str(data)[:300]}")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def _generate_once(model_name, messages, temperature, max_tokens, timeout, max_retries, label,
                   json_mode=False):
    """Send a request to a single model with built-in 429/5xx retries. It does not fall
    back between models (that is generate()'s job).
    Returns (text, ok); ok=False means this model should be abandoned (retries
    exhausted, quota wait too long, 404), so move on to the next model.
    With json_mode=True the request carries response_format=json_object; if the model
    answers 400 saying it is unsupported, the field is dropped and the same model is
    retried immediately (this does not consume a max_retries attempt and does not
    switch to a fallback model)."""
    payload = {"model": model_name, "messages": messages, "max_tokens": max_tokens}
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    url = f"{get_base_url()}/chat/completions"
    max_wait = _env_float("GROQ_MAX_WAIT_SECONDS", DEFAULT_MAX_WAIT)
    delay = 5.0
    last_error = None

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        started = time.monotonic()
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=get_timeout(timeout))
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Groq {model_name} connection error ({type(e).__name__}); retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code in (413, 429) and "request too large" in resp.text.lower():
            # Some models (e.g. qwen) report "request too large" as a 429; that is not
            # rate limiting and no amount of waiting will make it pass.
            _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "too_large", resp.text[:300]
            print(f"   [skip] Groq {model_name} returned HTTP {resp.status_code} (this request exceeds the model's per-request token limit).")
            return None, False

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = GroqAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            retry_after = resp.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            if wait > max_wait:
                print(f"   [skip] Groq {model_name} quota exhausted; the wait would be {wait:.0f}s (limit {max_wait:.0f}s), "
                      f"not waiting - switching to a fallback model.")
                _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "quota", str(last_error)
                return None, False
            print(f"   [wait] Groq {model_name} returned HTTP {resp.status_code}; retrying in {wait:.0f}s ({attempt}/{max_retries})")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 404:
            print(f"   [skip] Groq {model_name} returned 404 (model missing or retired); switching to a fallback model.")
            _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "not_found", resp.text[:300]
            return None, False

        if resp.status_code == 413:
            # The request exceeds this model's per-request token limit; retrying the same
            # model is useless, so switch to a fallback model (its limit may differ).
            _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "too_large", resp.text[:300]
            print(f"   [skip] Groq {model_name} returned 413 (request exceeds the model's per-request token limit); switching to a fallback model.")
            return None, False

        if (resp.status_code == 400 and "response_format" in payload
                and "response_format" in resp.text.lower()):
            print(f"   [warn] Groq {model_name} does not support response_format; dropping JSON mode and retrying the same model.")
            del payload["response_format"]
            attempt -= 1
            continue

        if resp.status_code == 400 and ("failed to generate json" in resp.text.lower()
                                        or "json_validate_failed" in resp.text.lower()):
            # This model could not produce valid JSON this time (common with smaller
            # models) - try a fallback model; it is not a service-wide problem.
            _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "bad_output", resp.text[:300]
            print(f"   [skip] Groq {model_name} could not produce valid JSON (HTTP 400); switching to a fallback model.")
            return None, False

        if resp.status_code >= 400:
            raise GroqAPIError(f"Groq API error HTTP {resp.status_code}: {resp.text[:300]}")

        latency = time.monotonic() - started
        data = resp.json()
        usage = data.get("usage") or {}
        text = _extract_text(data)
        label_note = f" | {label}" if label else ""
        print(f"   [ok] Groq {model_name} | {latency:.1f}s | "
              f"tokens in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}{label_note}")

        finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
        if finish_reason == "length" and not text.strip():
            print(f"   [warn] The reply was cut off by max_tokens ({max_tokens}) before any text was produced. Raise max_tokens and retry.")
        _LAST_USED["model"] = model_name
        return text, True

    print(f"   [fail] Groq {model_name} still failing after {max_retries} attempts: {last_error}")
    _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "quota", str(last_error)
    return None, False


def generate(prompt, model=None, system=None, temperature=None, max_tokens=4096,
             timeout=None, max_retries=3, raise_on_failure=True, label=None,
             allow_model_fallback=True, fallback_models=None, json_mode=False,
             fallback_on_too_large=True):
    """
    Call the Groq Chat Completions API: a single request with no conversation memory.

    - prompt: the user message text
    - system: optional system prompt
    - model: optional; defaults to GROQ_MODEL_NAME
    - returns: the generated text
    - 429 / 5xx are retried up to max_retries times, honouring Retry-After or
      exponential backoff; if Retry-After exceeds GROQ_MAX_WAIT_SECONDS (usually a
      daily quota), or on a 404, or once retries are exhausted, fallback_models are
      used (default: read from GROQ_FALLBACK_MODELS).
    - raise_on_failure: True (default) raises GroqAPIError after every model has
      failed; False prints the error and returns an empty string.
    - fallback_on_too_large: True (default) = when the primary model answers
      "request too large", try a fallback model; False = raise
      GroqRequestTooLargeError immediately without switching. Fallback models
      usually have the same or a lower limit and lower quality, so set this to False
      if the caller has a better fix (e.g. shrink the request and resend it to the
      primary model).
    - json_mode: True = ask for a JSON object (response_format=json_object). The
      OpenAI-compatible API requires the word "JSON" to appear in the prompt or the
      system message, otherwise it answers 400; if neither contains it, a sentence is
      appended to the system message automatically. If the model does not support
      response_format, the field is dropped and the same model is retried (see
      _generate_once).
    """
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_model_name()).strip()

    models_to_try = [model_name]
    if fallback_models is not None:
        for m in fallback_models:
            m = m.strip()
            if m and m not in models_to_try:
                models_to_try.append(m)
    elif allow_model_fallback:
        models_to_try.extend(get_fallback_model_names(model_name))

    if json_mode and "json" not in ((system or "") + "\n" + str(prompt)).lower():
        system = (system + "\n" if system else "") + "Output valid JSON only."

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    failures = []
    for current_model in models_to_try:
        if current_model != models_to_try[0]:
            print(f"   [switch] Using fallback model: {current_model}")
        _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "", ""
        text, ok = _generate_once(current_model, messages, temperature, max_tokens,
                                  timeout, max_retries, label, json_mode=json_mode)
        if ok:
            return text
        failures.append(_LAST_FAILURE["reason"])
        if _LAST_FAILURE["reason"] == "too_large" and not fallback_on_too_large:
            raise GroqRequestTooLargeError(
                f"Request too large: Groq {current_model} exceeded its per-request token limit, "
                f"and the caller asked not to switch to a fallback model. "
                f"{_LAST_FAILURE['detail']}")

    tried = f"{len(models_to_try)} model(s) ({', '.join(models_to_try)})"
    detail = _LAST_FAILURE["detail"]
    if "quota" in failures:
        exc = GroqAPIError(f"Groq: all {tried} were tried and failed (quota exhausted / connection problems included). {detail}")
    elif failures and all(r == "too_large" for r in failures):
        exc = GroqRequestTooLargeError(
            f"Request too large: all of Groq's {tried} returned 413 (per-request token limit exceeded). {detail}")
    else:
        exc = GroqRequestFailedError(
            f"Groq: none of the {tried} could complete this request "
            f"(reasons: {', '.join(f or 'unknown' for f in failures)}). {detail}")
    msg = str(exc)
    if raise_on_failure:
        raise exc
    print(msg)
    return ""


DEFAULT_VISION_MODEL = "qwen/qwen3.8-27b"
DEFAULT_VISION_FALLBACK_MODELS = []
# Checked against https://console.groq.com/docs/vision (2026-10): this is currently the
# only Groq model that accepts image input (the previously used llama-4-scout/maverick
# now return 404 because they were retired). At most 3 images per request, each costing
# a fixed 2048 tokens. There is no other vision model to fall back to, so the fallback
# list is empty - if this model fails, it is a real failure, not a "switch model and it
# works" situation.


def _image_to_data_url(image):
    """Accept a file path (str/Path) or a PIL.Image and return a data: URL string (base64 PNG)."""
    import base64
    import io

    from PIL import Image as PILImage

    if isinstance(image, PILImage.Image):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        raw = buf.getvalue()
    else:
        with open(image, "rb") as f:
            raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/png;base64,{b64}"


def generate_vision(prompt, images, model=None, system=None, temperature=None, max_tokens=4096,
                    timeout=None, max_retries=3, raise_on_failure=True, label=None,
                    allow_model_fallback=True, fallback_models=None):
    """Like generate(), but with images attached (OpenAI-compatible vision format).

    Groq's own primary/fallback text models (GROQ_MODEL_NAME etc.) cannot read images,
    so this function does NOT share generate()'s model-list logic - it defaults to
    DEFAULT_VISION_MODEL. Deciding which images to send is the caller's job; no
    compression or cropping is done here, so pre-process large images before calling.

    - images: one or more items, each a file path or a PIL.Image
    - the remaining parameters mean the same as in generate()
    """
    _validate_prompt(prompt)
    configure()
    model_name = (model or DEFAULT_VISION_MODEL).strip()

    models_to_try = [model_name]
    if fallback_models is not None:
        for m in fallback_models:
            m = m.strip()
            if m and m not in models_to_try:
                models_to_try.append(m)
    elif allow_model_fallback:
        for m in DEFAULT_VISION_FALLBACK_MODELS:
            if m not in models_to_try:
                models_to_try.append(m)

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    failures = []
    for current_model in models_to_try:
        if current_model != models_to_try[0]:
            print(f"   [switch] Using fallback vision model: {current_model}")
        _LAST_FAILURE["reason"], _LAST_FAILURE["detail"] = "", ""
        text, ok = _generate_once(current_model, messages, temperature, max_tokens,
                                  timeout, max_retries, label, json_mode=False)
        if ok:
            return text
        failures.append(_LAST_FAILURE["reason"])

    tried = f"{len(models_to_try)} vision model(s) ({', '.join(models_to_try)})"
    detail = _LAST_FAILURE["detail"]
    exc = GroqAPIError(f"Groq: none of the {tried} could complete this request. {detail}")
    if raise_on_failure:
        raise exc
    print(str(exc))
    return ""


if __name__ == "__main__":
    import sys

    # Some Windows consoles default to a legacy code page that cannot print every character.
    sys.stdout.reconfigure(encoding="utf-8")
    configure()
    prompt_arg = " ".join(sys.argv[1:]) or "Introduce yourself in one sentence."
    print(f"   Groq primary model: {get_default_model_name()} | fallback: {get_fallback_model_names()}")
    print(f"   Sending: {prompt_arg}\n")
    result = generate(prompt_arg)
    print("\n" + "=" * 78)
    print(result)
    print("=" * 78)
