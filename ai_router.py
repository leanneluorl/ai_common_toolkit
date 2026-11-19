# -*- coding: utf-8 -*-
"""
ai_router.py - One interface for switching between Gemini and Groq
==================================================================
Usage:
    import ai_router
    try:
        text, used_model = ai_router.generate_json_text(prompt, label="episode 3")
    except ai_router.AIQuotaExhaustedError as e:
        ...   # whether the provider is Gemini or Groq, "quota exhausted / rate limited"
              # always surfaces as this single exception type

    ai_router.get_provider()                      # "gemini" / "groq", handy for logging
    ai_router.get_default_model_name()            # the provider's default model, for logging

Command-line smoke test (python ai_router.py "Return the JSON {\\"ok\\": true}"):
    AI_PROVIDER=gemini python ai_router.py ...
    AI_PROVIDER=groq   python ai_router.py ...

Tunable via .env:
    AI_PROVIDER    gemini (default) or groq; a provider argument from the caller overrides it

Note: only free providers are wired in here. Paid models (for example the Qwen
client in qwen_client.py) are deliberately NOT part of PROVIDERS: the router's
provider can be switched by .env, a runtime menu or an argument, and if a billed
model were in the list a single wrong value would start charging money without
anyone noticing. Paid models are always used by importing their module explicitly.
"""

import os

import ai_client
import groq_client

PROVIDERS = ("gemini", "groq")
DEFAULT_PROVIDER = "gemini"


class AIQuotaExhaustedError(Exception):
    """Raised when quota is exhausted or the request is rate limited so that
    further calls in this round are pointless.

    Unifies the provider-specific exceptions so callers only need to catch this one:
      - ai_client.AllApiKeysExhaustedError (all Gemini quota used up today)
      - ai_client.RateLimitedError (Gemini keeps hitting per-minute limits, usually
        because a single request is too large)
      - groq_client.GroqAPIError (every Groq model failed; this includes quota
        exhaustion but can also be another HTTP error; the original message is kept
        in the text and in __cause__)"""
    pass


class AIRequestFailedError(Exception):
    """No model could complete this request, but the cause is not quota (for example
    the model did not return valid JSON, the model was retired, or a mix of these).
    Only this one item is affected: the caller can retry it or let it fail and keep
    processing other items. Do not treat it as "quota is gone for today" and abort
    the whole batch."""
    pass


class AIRequestTooLargeError(Exception):
    """The request itself is too large (above the model's per-request token limit,
    e.g. Groq's free-tier TPM cap of 8000), and every fallback model would reject it
    just the same. Deliberately NOT a subclass of AIQuotaExhaustedError: this is not a
    quota problem, waiting until tomorrow will not help, and only this one item is
    affected - the caller should fail this item and carry on with the others instead
    of aborting the batch. Fix it by shrinking the request, using a model with a
    higher limit, or switching provider."""
    pass


# Runtime override: after the main program (e.g. an interactive menu) has chosen a
# provider it calls set_runtime_choice(); from then on every module in the same
# process that goes through the router (including imported helper modules) uses the
# same AI without each one having to pass provider/model. Explicit arguments still win.
_RUNTIME_PROVIDER = None
_RUNTIME_MODEL = None


def set_runtime_choice(provider=None, model=None):
    """Set this process's default provider and model (the model only applies when the
    resolved provider equals this provider)."""
    global _RUNTIME_PROVIDER, _RUNTIME_MODEL
    _RUNTIME_PROVIDER = _resolve_provider(provider) if provider else None
    _RUNTIME_MODEL = (model or "").strip() or None


def _resolve_model(provider, model):
    if model:
        return model
    if _RUNTIME_MODEL and _RUNTIME_PROVIDER == provider:
        return _RUNTIME_MODEL
    return None


def _resolve_provider(provider=None):
    if (provider is None or not str(provider).strip()) and _RUNTIME_PROVIDER:
        return _RUNTIME_PROVIDER
    if provider is None or not str(provider).strip():
        # Make sure .env is loaded first so AI_PROVIDER can live in .env.
        from dotenv import load_dotenv
        load_dotenv()
        provider = os.getenv("AI_PROVIDER") or DEFAULT_PROVIDER
    provider = str(provider).strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported AI_PROVIDER '{provider}'; it must be one of {' / '.join(PROVIDERS)}.")
    return provider


def get_provider(provider=None):
    """The provider in effect (caller argument > AI_PROVIDER env var > gemini)."""
    return _resolve_provider(provider)


def get_default_model_name(provider=None):
    """The provider's default model name (for logging; when Gemini falls back to a
    different model the one actually used may differ)."""
    p = _resolve_provider(provider)
    if p == "groq":
        groq_client.configure()
        return groq_client.get_default_model_name()
    ai_client.configure()
    return ai_client.get_default_model_name()


def _success_counts_by_model():
    """Read the cumulative per-model success counts from ai_client's usage stats, so
    the model actually used by a call can be inferred from the before/after difference."""
    counts = {}
    try:
        snap = ai_client.usage_snapshot()
    except Exception:
        return counts
    for owner in snap.get("owners", []):
        for m in owner.get("models", []):
            counts[m["model"]] = counts.get(m["model"], 0) + int(m.get("total_success", 0))
    return counts


def _generate_gemini(prompt, model, system, temperature, label, json_mode=True, max_tokens=None):
    if system:
        # ai_client.generate has no system parameter, so prepend it to the prompt.
        prompt = f"{system}\n\n{prompt}"
    generation_config = {}
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    if temperature is not None:
        generation_config["temperature"] = temperature
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens

    before = _success_counts_by_model()
    text = ai_client.generate(prompt, model_name=model, label=label,
                              generation_config=generation_config)
    after = _success_counts_by_model()

    used_model = ""
    for name, count in after.items():
        if count > before.get(name, 0):
            used_model = name
            break
    if not used_model:
        used_model = model or ai_client.get_default_model_name()
    return text, used_model


def _generate_groq(prompt, model, system, temperature, max_tokens, label, json_mode=True,
                   fallback_on_too_large=True):
    text = groq_client.generate(prompt, model=model, system=system, temperature=temperature,
                                max_tokens=max_tokens, label=label, json_mode=json_mode,
                                fallback_on_too_large=fallback_on_too_large)
    # When a fallback model took over, the model actually used is last_used_model(), not the one passed in.
    return text, (groq_client.last_used_model() or model or groq_client.get_default_model_name())


def generate_text(prompt, provider=None, model=None, system=None, temperature=None,
                  max_tokens=4096, label=None):
    """
    Ask the AI for plain text (JSON is not enforced) and return (text, used_model).
    The parameters match generate_json_text; max_tokens applies to both providers
    (for Gemini it maps to max_output_tokens).
    Quota exhausted / rate limited -> always raises AIQuotaExhaustedError.
    """
    p = _resolve_provider(provider)
    model = _resolve_model(p, model)
    try:
        if p == "groq":
            return _generate_groq(prompt, model, system, temperature, max_tokens, label, json_mode=False)
        return _generate_gemini(prompt, model, system, temperature, label,
                                json_mode=False, max_tokens=max_tokens)
    except groq_client.GroqRequestTooLargeError as e:
        raise AIRequestTooLargeError(f"[{p}] {e}") from e
    except groq_client.GroqRequestFailedError as e:
        raise AIRequestFailedError(f"[{p}] {e}") from e
    except (ai_client.AllApiKeysExhaustedError, ai_client.RateLimitedError,
            groq_client.GroqAPIError) as e:
        raise AIQuotaExhaustedError(f"[{p}] {type(e).__name__}: {e}") from e


def generate_json_text(prompt, provider=None, model=None, system=None, temperature=None,
                       max_tokens=8192, label=None, fallback_on_too_large=True):
    """
    Ask the AI for JSON text and return (text, used_model). text is the raw string the
    model produced; the caller runs json.loads itself (fallback models occasionally
    append extra text after valid JSON, so parse tolerance is left to the caller).

    - provider: "gemini" / "groq"; if omitted, AI_PROVIDER is read, then gemini is used.
    - model: optional; defaults to the provider's default model.
    - system: optional. Groq receives it as a system message; Gemini has no system
      parameter so it is prepended to the prompt.
    - temperature: optional; the model default is used if omitted.
    - max_tokens: only takes effect for Groq. Gemini keeps ai_client's default output
      cap (no extra limit).
    - used_model: for Gemini it is inferred by diffing the usage stats before and
      after the call (so it can be a fallback model); for Groq it is the model passed
      in or the default (a Groq fallback switch is not reflected here, only in the logs).
    - fallback_on_too_large: only affects Groq. False = when the primary model says
      "request too large", raise AIRequestTooLargeError immediately instead of trying
      fallback models (the caller must shrink the request and resend).
    - Quota exhausted / rate limited -> always raises AIQuotaExhaustedError.
    """
    p = _resolve_provider(provider)
    model = _resolve_model(p, model)
    try:
        if p == "groq":
            return _generate_groq(prompt, model, system, temperature, max_tokens, label,
                                  fallback_on_too_large=fallback_on_too_large)
        return _generate_gemini(prompt, model, system, temperature, label)
    except groq_client.GroqRequestTooLargeError as e:
        raise AIRequestTooLargeError(f"[{p}] {e}") from e
    except groq_client.GroqRequestFailedError as e:
        raise AIRequestFailedError(f"[{p}] {e}") from e
    except (ai_client.AllApiKeysExhaustedError, ai_client.RateLimitedError,
            groq_client.GroqAPIError) as e:
        raise AIQuotaExhaustedError(f"[{p}] {type(e).__name__}: {e}") from e


if __name__ == "__main__":
    import json
    import sys

    # Some Windows consoles default to a legacy code page that cannot print every character.
    sys.stdout.reconfigure(encoding="utf-8")
    prompt_arg = " ".join(sys.argv[1:]) or 'Return the JSON {"ok": true}'
    provider_name = get_provider()
    print(f"   provider: {provider_name} | default model: {get_default_model_name(provider_name)}")
    print(f"   Sending: {prompt_arg}\n")
    text, used = generate_json_text(prompt_arg, label="ai_router smoke test")
    print("\n" + "=" * 78)
    print(f"used_model: {used}")
    print(text)
    try:
        print(f"Parsed JSON: {json.loads(text)}")
    except json.JSONDecodeError as e:
        print(f"Could not parse as JSON: {e}")
    print("=" * 78)
