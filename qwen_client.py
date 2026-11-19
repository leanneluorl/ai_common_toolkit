# -*- coding: utf-8 -*-
"""
qwen_client.py - Client for Alibaba Cloud Model Studio (DashScope) Qwen chat models
===================================================================================
Calls DashScope's OpenAI-compatible Chat Completions API ({base_url}/chat/completions).

Note: Qwen models are billed per token (new accounts on the international site
usually get a free quota first, after which billing starts). Prices and free quotas
are shown in the Model Studio console and are intentionally not hard-coded here.
This module is deliberately NOT wired into ai_router.py: the router only holds free
providers, so a paid model is only ever used when this module is imported explicitly.

Regions (an API key is bound to a region; using the wrong base URL returns 401):
  - International (Singapore, default): https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  - Mainland China (Beijing)          : https://dashscope.aliyuncs.com/compatible-mode/v1

Thinking mode: on the API side the Qwen3.5 family enables thinking by default
(enable_thinking=true), which makes responses slower and produces more output tokens.
This module turns it off by default (QWEN_ENABLE_THINKING=false); set it to true, or
pass enable_thinking=True per call, to turn it on. The reasoning trace
(reasoning_content) is never mixed into the returned text and is never stored in the
conversation history (so it is not re-sent and re-billed on every turn).

Everything is received as a stream (stream=True): the timeout is "the longest wait
between two chunks", not "the whole response must finish within N seconds", so a long
answer with thinking enabled is not cut off just because the model thinks for several
minutes. A progress line is printed every 30 seconds while generating.

Usage 1: single request (no memory)
    import qwen_client
    text = qwen_client.generate("Write a short paragraph about ...", system="You are a professional editor")

Usage 2: continuous conversation (with memory; the interface mirrors
         ai_client.start_chat / send_chat_message)
    chat = qwen_client.start_chat(system="You are a professional author", enable_thinking=True)
    t1 = qwen_client.send_chat_message(chat, "First prompt...")
    t2 = qwen_client.send_chat_message(chat, "Second prompt...")   # remembers the first
    qwen_client.estimate_chat_tokens(chat)   # roughly how many tokens the next request sends (value reported by the API)
    print(qwen_client.usage_summary())       # cumulative usage of this process

Usage 3: pick a model by tier / task type (lists: MODEL_TIERS and TASK_TIERS below)
    model = qwen_client.get_task_model("metadata")            # -> default model of the low tier
    model = qwen_client.get_task_model("long_form_writing")   # -> default model of the high tier
    model = qwen_client.get_tier_default("upper_mid")         # -> default model of the upper-mid tier
    model = qwen_client.pick_model(tiers=("upper_mid", "low"))   # interactive menu grouped by tier
    text = qwen_client.generate("...", model=model)

Usage 4: translation (qwen-mt family, TRANSLATION_MODELS; a different API shape from
         generate(), see the block comment before translate())
    text = qwen_client.translate("Translate this sentence into English.", target_lang="English")
    text = qwen_client.translate("Mix well.", target_lang="German", source_lang="English",
                                  terms=[{"source": "mix well", "target": "gut vermischen"}])

Command line:
    python qwen_client.py "prompt text"                    send one request and print the reply (checks key / model)
    python qwen_client.py --usage                          list cumulative usage / limits per model (no API call)
    python qwen_client.py --set-used qwen3.5-plus 12345    calibrate a model's used amount against the console

Budget guard: the cumulative tokens (input + output + reasoning) of every model are
recorded in _log/qwen_usage.json, with a default limit of 1M tokens (= a typical free
quota). Before sending, the worst case of the request is estimated; if it would exceed
"limit - reserve", QwenBudgetExceededError is raised and nothing is sent.
Note: the local ledger only counts calls that go through this module; usage elsewhere
(console playground, other programs) has to be added with --set-used.

Environment variables (.env):
    DASHSCOPE_API_KEY          required (create it on the API Key page of the Model Studio console)
    QWEN_BASE_URL              API base URL, default the international site https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    QWEN_MT_BASE_URL           base_url used only by the qwen-mt family; unset = share QWEN_BASE_URL
                               (the official docs use a workspace-specific URL; override here if you see 404s)
    QWEN_MODEL_NAME            primary model, default qwen3.5-plus
    QWEN_MODEL_HIGH / _UPPER_MID / _MID / _LOW   default model per tier; unset = first entry of that tier in MODEL_TIERS
    QWEN_FALLBACK_MODELS       comma-separated fallback models, default none (empty)
    QWEN_ENABLE_THINKING       enable thinking mode, default false
    QWEN_THINKING_BUDGET       max tokens for the reasoning trace (optional, unset = model decides); caps thinking-mode cost
    QWEN_MAX_TOKENS            output limit per response, default 8192
    QWEN_TIMEOUT_SECONDS       longest wait between two stream chunks in seconds, default 180
    QWEN_MAX_WAIT_SECONDS      longest time to wait on a 429 before switching to a fallback model, default 90
    QWEN_TOKEN_LIMIT           cumulative token limit per model, default 1000000 (0 = no limit, billing will start)
    QWEN_TOKEN_RESERVE         reserve (buffer for drift between the local ledger and the console), default 50000
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

_CONFIGURED = False

DEFAULT_MODEL_NAME = "qwen3.5-plus"
# Model lists for four tiers (the entry recommended first comes first within a tier). They all
# live under the same DashScope account / key. Each model has its own free quota on the
# provider's pricing page, which differs per model and per region, so check the console
# before relying on it.
#   high      = flagship (strongest; once the free quota is gone the unit price is
#               several times that of the upper-mid tier)
#   upper_mid = Plus family and large open-weight models (good at writing and structured output)
#   mid       = newer-generation Flash snapshots and large MoE models (cheap and fast)
#   low       = stable Flash aliases (cheapest; short copy, classification, small batch jobs)
# Specialty models (code / vision / OCR / translation / role play / video) are not part of any
# tier, see SPECIALTY_MODELS below. A model missing from these lists is not necessarily
# retired - it may simply not be listed here - so check the console for usage and availability.
MODEL_TIERS = {
    "high": [
        ("qwen3.8-2.4t-a95b", "Qwen 3.8 open-weight flagship, 2.4T"),
        ("qwen3.7-max-2026-06-08", "Qwen 3.7 Max snapshot (recommended by the provider for long-form writing)"),
        ("qwen3.7-max-2026-05-20", "Qwen 3.7 Max older snapshot"),
        ("qwen3.7-max-2026-05-17", "Qwen 3.7 Max older snapshot"),
        ("glm-5.3", "GLM 5.3 (flagship, recommended by the provider for long-form writing)"),
        ("kimi-k3", "Kimi K3 (flagship)"),
        ("qwen-max", "Qwen Max (stable flagship alias)"),
        ("deepseek-v4-pro-0813", "DeepSeek V4 Pro 0813 snapshot (hosted by Alibaba Cloud)"),
        ("qwen3.5-397b-a17b", "Qwen3.5 open-weight flagship 397B"),
        ("qwen3.5-122b-a10b", "Qwen3.5 open-weight flagship 122B"),
        ("qwen3.8-max", "Qwen 3.8 Max (newest flagship)"),
        ("qwen3.8-max-0902", "Qwen 3.8 Max 0902 snapshot"),
        ("qwen3.7-max", "Qwen 3.7 Max (flagship)"),
        ("qwen3.7-max-preview", "Qwen 3.7 Max preview (flagship)"),
        ("glm-5.2", "GLM 5.2 (flagship)"),
        ("glm-5.1", "GLM 5.1 (flagship)"),
    ],
    "upper_mid": [
        # Dated snapshots come first (pinned version, good for tests; "latest" aliases drift
        # with provider updates).
        ("qwen-plus-2025-12-01", "Qwen Plus 2025-12-01 snapshot"),
        ("qwen-plus-2025-07-14", "Qwen Plus 2025-07-14 snapshot"),
        ("qwen-plus-2025-04-28", "Qwen Plus 2025-04-28 snapshot"),
        ("qwen-plus-latest", "Qwen Plus latest alias"),
        ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash (upper-mid)"),
        ("qwen3.8-27b", "Qwen 3.8 open-weight 27B"),
        ("qwen3.6-27b", "Qwen 3.6 open-weight 27B"),
        ("qwen3.5-27b", "Qwen 3.5 open-weight 27B"),
        ("qwen-plus", "Qwen Plus (stable alias)"),
    ],
    "mid": [
        ("qwen3.8-flash", "Qwen 3.8 Flash (newest generation, strongest Flash)"),
        ("qwen3.7-flash-2026-07-15", "Qwen 3.7 Flash snapshot"),
        ("qwen3.6-flash-2026-04-16", "Qwen 3.6 Flash snapshot"),
        ("qwen3.5-flash-2026-02-23", "Qwen 3.5 Flash snapshot"),
        ("deepseek-v4-flash-0731", "DeepSeek V4 Flash 0731 snapshot"),
        ("deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash 1017 snapshot"),
        ("qwen3.6-35b-a3b", "Qwen 3.6 open-weight MoE 35B-A3B"),
        ("qwen3.5-35b-a3b", "Qwen 3.5 open-weight MoE 35B-A3B"),
    ],
    "low": [
        ("qwen3.7-flash", "Qwen 3.7 Flash (stable alias)"),
        ("qwen3.6-flash", "Qwen 3.6 Flash (stable alias)"),
        ("qwen3.5-flash", "Qwen 3.5 Flash (stable alias)"),
        ("qwen-flash", "Qwen Flash (stable alias, cheapest)"),
        ("qwen-flash-2025-07-28", "Qwen Flash older snapshot"),
    ],
}
TIER_LABELS = {"high": "High", "upper_mid": "Upper-mid", "mid": "Mid", "low": "Low"}
# Specialty models (not part of any tier; call them with model= according to the task type).
# Video-generation models use a different asynchronous API that this module does not call.
SPECIALTY_MODELS = {
    "code": ["kimi-k2.7-code", "qwen3-coder-flash", "qwen3-coder-flash-2025-07-28"],
    "vision": ["qwen3-vl-plus"],
    "ocr": ["qwen-vl-ocr-2025-11-20", "qwen-vl-ocr"],
    "translation": ["qwen-mt-plus", "qwen-mt-flash", "qwen-mt-lite"],
    "roleplay": ["qwen-plus-character", "qwen-flash-character"],
}
# Translation only: the qwen-mt family. NOTE: a completely different API shape from the regular
# chat models; do not call it through generate(), use translate() below (call limits and
# caveats are in the block comment before translate()).
TRANSLATION_MODELS = ("qwen-mt-plus", "qwen-mt-flash", "qwen-mt-lite")
_MT_FULL_TEXT_MODELS = {"qwen-mt-plus"}   # these models stream "non-incrementally", see full_text_stream in _read_stream
ALL_MODELS = [m for tier in ("high", "upper_mid", "mid", "low") for m in MODEL_TIERS[tier]]
# Instruct-only models without thinking: sending enable_thinking=true would be rejected, so it is
# always forced off (currently only the code-specialty models are instruct-only).
_NO_THINKING_MODELS = {"kimi-k2.7-code", "qwen3-coder-flash", "qwen3-coder-flash-2025-07-28"}

# Which tier is recommended for which kind of work (get_task_model("metadata") returns the model;
# a task that is not listed = upper_mid).
# NOTE: do not route translation through here: get_task_model() returns a plain model name for
# generate(); the qwen-mt family has a different API shape (see the block comment before
# translate()), and passing such a name to generate() would fail completely. For translation
# call qwen_client.translate(text, target_lang=...) directly.
TASK_TIERS = {
    "long_form_writing": "high",       # long, voice-driven writing such as scripts and chapters
    "storyboard": "upper_mid",         # splitting a script into shots
    "image_prompt": "upper_mid",       # writing image prompts
    "outline": "upper_mid",            # outlines
    "rewrite": "upper_mid",            # rewriting and polishing
    "metadata": "low",                 # titles, descriptions, tags, chapter timestamps
    "summary": "low",                  # summaries
    "classify": "low",                 # classification, tagging, field extraction
    "format": "low",                   # format conversion (JSON, tables, SRT)
}


def get_model_tier(model_name):
    """Which tier a model belongs to ("high" / "upper_mid" / "mid" / "low"); None if it is not listed."""
    for tier, models in MODEL_TIERS.items():
        if any(name == model_name for name, _note in models):
            return tier
    return None


def get_tier_models(tier):
    """Model names of a tier (recommended first)."""
    return [name for name, _note in MODEL_TIERS[tier]]


def get_tier_default(tier):
    """Default model of a tier: QWEN_MODEL_HIGH / _UPPER_MID / _MID / _LOW from .env, otherwise the first entry of the list."""
    return (os.getenv(f"QWEN_MODEL_{tier.upper()}") or "").strip() or get_tier_models(tier)[0]


def get_task_model(task):
    """Recommended model for a kind of work, following TASK_TIERS; tasks that are not listed use the upper-mid tier."""
    return get_tier_default(TASK_TIERS.get(task, "upper_mid"))


def pick_model(tiers=("high", "upper_mid", "mid", "low"), default=None, title="Choose a model"):
    """Interactive menu: lists the models grouped by tier and returns the model name; plain Enter = default; invalid input returns ""."""
    default = default or get_tier_default(tiers[0])
    options = []
    print(f"\n{title} (Alibaba Cloud Model Studio paid models, each with its own free quota):")
    for tier in tiers:
        print(f"  -- {TIER_LABELS[tier]} --")
        for name, note in MODEL_TIERS[tier]:
            options.append(name)
            mark = " (default)" if name == default else ""
            print(f"  [{len(options)}] {name} - {note}{mark}")
    raw = input(f"Enter a number (plain Enter = {default}): ").strip()
    if not raw:
        return default
    try:
        return options[int(raw) - 1]
    except (ValueError, IndexError):
        print("Invalid input.")
        return ""
DEFAULT_FALLBACK_MODELS = []
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_WAIT = 90.0
HEARTBEAT_SECONDS = 30.0

# Error-code keywords DashScope uses when quota is exhausted / the account is in arrears
# (HTTP 400/403 can carry these too, not only 429)
_QUOTA_MARKERS = ("arrearage", "insufficient_quota", "quota", "freetieronly")


class QwenAPIError(Exception):
    """Raised when the Qwen (DashScope) API fails (HTTP 4xx/5xx, retries exhausted, malformed reply...)."""
    pass


# The model that served the most recent successful request (with fallbacks, callers use this to
# record the model that was really used)
_LAST_USED = {"model": ""}
# Cumulative usage of this process (numbers actually reported by the API; with a paid model you
# want to see what you spent)
USAGE = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
         "reasoning_tokens": 0, "seconds": 0.0}


def last_used_model():
    return _LAST_USED["model"]


def usage_summary():
    u = USAGE
    return (f"{u['calls']} calls | input {u['prompt_tokens']:,} tokens (cache hits {u['cached_tokens']:,}) | "
            f"output {u['completion_tokens']:,} tokens (reasoning {u['reasoning_tokens']:,}) | "
            f"{u['seconds'] / 60:.1f} min")


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
            "Environment variable DASHSCOPE_API_KEY not found; please set it in your .env "
            "(create an API key in the Alibaba Cloud Model Studio console)."
        )
    _CONFIGURED = True


def get_api_key():
    return (os.getenv("DASHSCOPE_API_KEY") or "").strip()


def get_base_url():
    return (os.getenv("QWEN_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def get_default_model_name():
    return (os.getenv("QWEN_MODEL_NAME") or DEFAULT_MODEL_NAME).strip()


def get_fallback_model_names(primary=None):
    """Fallback model list (excluding the primary model itself)."""
    raw = os.getenv("QWEN_FALLBACK_MODELS")
    names = [n.strip() for n in raw.split(",")] if raw is not None else list(DEFAULT_FALLBACK_MODELS)
    primary = (primary or get_default_model_name()).strip()
    return [n for n in names if n and n != primary]


def get_enable_thinking(explicit=None):
    if explicit is not None:
        return bool(explicit)
    return (os.getenv("QWEN_ENABLE_THINKING") or "false").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    raw = os.getenv(name)
    try:
        return float(raw) if raw and str(raw).strip() else default
    except ValueError:
        return default


def get_thinking_budget(explicit=None):
    if explicit is not None:
        return int(explicit) or None
    return int(_env_float("QWEN_THINKING_BUDGET", 0)) or None


def get_max_tokens(explicit=None):
    if explicit is not None:
        return explicit
    return int(_env_float("QWEN_MAX_TOKENS", DEFAULT_MAX_TOKENS))


def get_timeout(explicit=None):
    if explicit is not None:
        return explicit
    return _env_float("QWEN_TIMEOUT_SECONDS", DEFAULT_TIMEOUT)


def _headers():
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }


def _validate_prompt(prompt):
    """Never let an empty string reach the API (same guard as ai_client.py)."""
    if not prompt or not str(prompt).strip():
        raise ValueError("Refusing to send empty content to the Qwen API; check the upstream prompt source.")


def _read_stream(resp, model_name, started, full_text_stream=False):
    """Read one SSE stream to the end; returns (text, reasoning character count, finish_reason, usage).
    The reasoning trace is only counted (for progress messages), its content is not kept.

    full_text_stream=True: this model streams "non-incrementally" (currently only qwen-mt-plus) -
    every chunk's delta.content carries the complete text so far, so it has to REPLACE the
    previous text instead of being appended like for ordinary models; otherwise the content
    would be duplicated."""
    content_parts, reasoning_chars = [], 0
    full_text = ""
    finish_reason, usage = None, {}
    next_beat = started + HEARTBEAT_SECONDS

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if chunk.get("error"):
            raise QwenAPIError(f"Qwen reported an error in the middle of the stream: {str(chunk['error'])[:300]}")
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if full_text_stream:
                    full_text = delta["content"]
                else:
                    content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_chars += len(delta["reasoning_content"])
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        now = time.monotonic()
        if now >= next_beat:
            content_chars = len(full_text) if full_text_stream else sum(len(p) for p in content_parts)
            stage = "thinking" if reasoning_chars and not content_chars else "writing"
            print(f"   [wait] Qwen {model_name} {stage}... {now - started:.0f}s elapsed"
                  f" (reasoning {reasoning_chars:,} chars / text {content_chars:,} chars)")
            next_beat = now + HEARTBEAT_SECONDS

    return (full_text if full_text_stream else "".join(content_parts)), reasoning_chars, finish_reason, usage


# ============================================================
# Budget guard: cumulative tokens per model stay below a limit (default 1M)
# ============================================================
# The ledger lives in qwen_usage.json and accumulates across runs:
# {"qwen3.5-plus": {"used_tokens": 12345, "updated": "..."}}
# Counting = prompt_tokens + completion_tokens reported by the API (reasoning included),
# the same way the console deducts free quota. Before sending, the worst case of this request
# is estimated; if adding it would exceed "limit - reserve" the request is refused and not sent.
# The reserve covers drift between the local ledger and the console (for example a stream that
# broke off, or a billed request whose usage was never received).
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_log")
os.makedirs(_LOG_DIR, exist_ok=True)
USAGE_LEDGER_PATH = os.path.join(_LOG_DIR, "qwen_usage.json")
DEFAULT_TOKEN_LIMIT = 1_000_000
DEFAULT_TOKEN_RESERVE = 50_000


class QwenBudgetExceededError(Exception):
    """The cumulative usage of this model is about to exceed QWEN_TOKEN_LIMIT (default 1M),
    so the request was NOT sent. Deliberately not a subclass of QwenAPIError: this is not an
    API error, and retrying or switching to another item would be blocked the same way."""
    pass


def get_token_limit():
    return int(_env_float("QWEN_TOKEN_LIMIT", DEFAULT_TOKEN_LIMIT))


def get_token_reserve():
    return int(_env_float("QWEN_TOKEN_RESERVE", DEFAULT_TOKEN_RESERVE))


def _load_ledger():
    try:
        with open(USAGE_LEDGER_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_ledger(ledger):
    tmp = USAGE_LEDGER_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)
    os.replace(tmp, USAGE_LEDGER_PATH)


def get_used_tokens(model_name):
    return int((_load_ledger().get(model_name) or {}).get("used_tokens") or 0)


def set_used_tokens(model_name, tokens):
    """Calibrate the ledger by hand (for example against the used amount shown in the console)."""
    ledger = _load_ledger()
    ledger[model_name] = {"used_tokens": int(tokens), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_ledger(ledger)


def _add_to_ledger(model_name, tokens):
    if tokens:
        set_used_tokens(model_name, get_used_tokens(model_name) + tokens)


def _check_budget(model_name, worst_case):
    limit = get_token_limit()
    if limit <= 0:
        return   # QWEN_TOKEN_LIMIT=0 = no limit (only set this when you deliberately choose paid usage)
    used = get_used_tokens(model_name)
    allowed = limit - get_token_reserve()
    if used + worst_case > allowed:
        raise QwenBudgetExceededError(
            f"{model_name} has used {used:,} tokens and this request may use up to {worst_case:,} more, "
            f"which would exceed the budget limit {allowed:,} (QWEN_TOKEN_LIMIT {limit:,} - reserve {get_token_reserve():,}); "
            f"the request was NOT sent. Switch to a model that still has quota, or check the console usage and adjust the limit.")


def budget_status(model_name):
    used, limit = get_used_tokens(model_name), get_token_limit()
    if limit <= 0:
        return f"{model_name}: used {used:,} tokens (no limit set)"
    return f"{model_name}: used {used:,} / {limit:,} tokens ({used / limit:.1%}), {max(limit - used, 0):,} left"


def _record_usage(usage, latency, model_name):
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cached = int(prompt_details.get("cached_tokens") or 0)
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    _add_to_ledger(model_name, int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0))
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
    USAGE["cached_tokens"] += cached
    USAGE["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    USAGE["reasoning_tokens"] += reasoning
    USAGE["seconds"] += latency
    return cached, reasoning


def _generate_once(model_name, messages, temperature, max_tokens, timeout, max_retries, label,
                   enable_thinking, thinking_budget=None, json_mode=False, extra_payload=None,
                   full_text_stream=False, include_enable_thinking=True, base_url=None):
    """Send a request to a single model, with built-in 429/5xx/connection-drop retries; no
    fallback between models (that is the caller's job).
    Returns (text, usage, ok); ok=False means this model should be given up on (retries
    exhausted, quota exhausted, model missing) and the next model should be tried.

    extra_payload: extra parameters merged into the top level of the payload (translate() uses
    it for translation_options; per the provider's docs such non-standard OpenAI parameters go at
    the same level as model / messages over HTTP, not inside messages).
    include_enable_thinking=False: do not send enable_thinking / thinking_budget at all (for the
    qwen-mt family; the docs do not mention thinking support there, so omitting the key is safer
    than sending false).
    full_text_stream: see _read_stream (qwen-mt-plus only; its stream is non-incremental).
    base_url: override the base_url for this request (the qwen-mt family may need a different
    workspace URL, see get_mt_base_url)."""
    if model_name in _NO_THINKING_MODELS:
        enable_thinking = False
    payload = {"model": model_name, "messages": messages, "max_tokens": max_tokens,
               "stream": True, "stream_options": {"include_usage": True}}
    if include_enable_thinking:
        payload["enable_thinking"] = enable_thinking
        if enable_thinking and thinking_budget:
            payload["thinking_budget"] = thinking_budget
    if temperature is not None:
        payload["temperature"] = temperature
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if extra_payload:
        payload.update(extra_payload)

    url = f"{(base_url or get_base_url()).rstrip('/')}/chat/completions"
    max_wait = _env_float("QWEN_MAX_WAIT_SECONDS", DEFAULT_MAX_WAIT)
    read_timeout = get_timeout(timeout)
    delay = 5.0
    last_error = None

    # Worst-case tokens this request can use: input (1 character ~ at most 1 token, deliberately
    # overestimated) + output limit + thinking limit
    worst_case = (sum(len(str(m.get("content", ""))) for m in messages) + max_tokens
                  + ((thinking_budget or max_tokens) if enable_thinking else 0))

    for attempt in range(1, max_retries + 1):
        _check_budget(model_name, worst_case)   # check before every retry: an interrupted stream may already have been billed
        started = time.monotonic()
        try:
            resp = requests.post(url, headers=_headers(), json=payload, stream=True,
                                 timeout=(15, read_timeout))
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Qwen {model_name} connection error ({type(e).__name__}); retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        with resp:
            if resp.status_code >= 400:
                body_lower = resp.text.lower()

                if resp.status_code in (400, 403) and any(m in body_lower for m in _QUOTA_MARKERS):
                    print(f"   [skip] Qwen {model_name} quota exhausted or account in arrears (HTTP {resp.status_code}); switching to the fallback model.")
                    return None, {}, False

                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = QwenAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                    retry_after = resp.headers.get("retry-after")
                    try:
                        wait = float(retry_after) if retry_after else delay
                    except ValueError:
                        wait = delay
                    if wait > max_wait:
                        print(f"   [skip] Qwen {model_name} is rate limited and would need {wait:.0f}s (above the {max_wait:.0f}s cap); "
                              f"not waiting it out, switching to the fallback model.")
                        return None, {}, False
                    print(f"   [wait] Qwen {model_name} returned HTTP {resp.status_code}; retrying in {wait:.0f}s ({attempt}/{max_retries})")
                    time.sleep(wait)
                    delay = min(delay * 2, 60)
                    continue

                if resp.status_code == 404 or "model_not_found" in body_lower or "model not exist" in body_lower:
                    hint = "; if the qwen-mt family keeps returning 404, try setting QWEN_MT_BASE_URL to the workspace-specific URL from the official docs" \
                        if model_name in TRANSLATION_MODELS else ""
                    print(f"   [skip] Qwen {model_name} model missing or retired (HTTP {resp.status_code}); switching to the fallback model{hint}.")
                    return None, {}, False

                if resp.status_code == 401:
                    raise QwenAPIError(
                        f"Qwen API 401: the key is invalid, or its region does not match base_url ({base_url or get_base_url()}). {resp.text[:300]}")

                if "accessdenied" in body_lower:
                    raise QwenAPIError(
                        f"Qwen API {resp.status_code}: this key has no permission to use {model_name}. "
                        f"Check in the Model Studio console that the service is activated, that the workspace "
                        f"the key belongs to is authorized for this model, and that the account is not in "
                        f"arrears or restricted to free quota only. {resp.text[:300]}")

                raise QwenAPIError(f"Qwen API error HTTP {resp.status_code}: {resp.text[:300]}")

            try:
                text, reasoning_chars, finish_reason, usage = _read_stream(resp, model_name, started,
                                                                            full_text_stream=full_text_stream)
            except requests.RequestException as e:
                # The connection dropped mid-stream / the gap between chunks exceeded read_timeout:
                # resend the whole request (the part already generated is still billed)
                last_error = e
                print(f"   [warn] Qwen {model_name} stream interrupted ({type(e).__name__}); resending in {delay:.0f}s ({attempt}/{max_retries})")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

        latency = time.monotonic() - started
        cached, reasoning = _record_usage(usage, latency, model_name)
        label_note = f" | {label}" if label else ""
        cache_note = f" (cached {cached:,})" if cached else ""
        thinking_note = f" | reasoning {reasoning:,} tokens" if enable_thinking else ""
        print(f"   [ok] Qwen {model_name} | {latency:.1f}s | "
              f"tokens in={usage.get('prompt_tokens', 0):,}{cache_note} out={usage.get('completion_tokens', 0):,}"
              f"{thinking_note}{label_note}")

        if finish_reason == "length":
            print(f"   [warn] The reply was cut off by max_tokens ({max_tokens:,}) and is incomplete. Raise QWEN_MAX_TOKENS"
                  f"{' or lower QWEN_THINKING_BUDGET' if enable_thinking else ''} and retry.")
        if not text.strip():
            last_error = QwenAPIError(f"Empty content returned (finish_reason={finish_reason})")
            print(f"   [warn] Qwen {model_name} returned empty content; retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        _LAST_USED["model"] = model_name
        return text, usage, True

    print(f"   [fail] Qwen {model_name} still failing after {max_retries} attempts: {last_error}")
    return None, {}, False


def _models_to_try(model_name, allow_model_fallback, fallback_models):
    models = [model_name]
    if fallback_models is not None:
        for m in fallback_models:
            m = m.strip()
            if m and m not in models:
                models.append(m)
    elif allow_model_fallback:
        models.extend(get_fallback_model_names(model_name))
    return models


def _call_with_fallback(models, messages, temperature, max_tokens, timeout, max_retries, label,
                        thinking, thinking_budget, json_mode=False, extra_payload=None,
                        full_text_models=frozenset(), include_enable_thinking=True, base_url=None):
    for current_model in models:
        if current_model != models[0]:
            print(f"   [switch] Using fallback model: {current_model}")
        text, usage, ok = _generate_once(current_model, messages, temperature, max_tokens, timeout,
                                         max_retries, label, thinking, thinking_budget, json_mode=json_mode,
                                         extra_payload=extra_payload,
                                         full_text_stream=current_model in full_text_models,
                                         include_enable_thinking=include_enable_thinking, base_url=base_url)
        if ok:
            return text, usage
    raise QwenAPIError(f"Qwen: all {len(models)} model(s) ({', '.join(models)}) were tried and failed.")


def generate(prompt, model=None, system=None, temperature=None, max_tokens=None,
             timeout=None, max_retries=3, raise_on_failure=True, label=None,
             allow_model_fallback=True, fallback_models=None, json_mode=False,
             enable_thinking=None, thinking_budget=None):
    """
    Call the Qwen (DashScope) Chat Completions API: a single request with no conversation memory.

    - prompt: the user message text
    - system: optional system prompt
    - model: optional; defaults to QWEN_MODEL_NAME (qwen3.5-plus)
    - returns: the generated text (without the reasoning trace)
    - 429 / 5xx / dropped connections are retried max_retries times; when quota is exhausted,
      the model is missing, or retries are exhausted, fallback_models are tried (default: read
      from QWEN_FALLBACK_MODELS, empty by default = no fallback).
    - enable_thinking: None = read QWEN_ENABLE_THINKING (default false); True/False sets it directly.
    - json_mode: True = ask for a JSON object (response_format=json_object); thinking mode is
      switched off automatically (thinking mode does not support structured output). If neither
      the prompt nor the system prompt contains the word "JSON", a sentence is appended to the
      system prompt automatically.
    - raise_on_failure: True (default) raises QwenAPIError after every model has failed;
      False prints the error and returns an empty string.
    """
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_model_name()).strip()
    thinking = get_enable_thinking(enable_thinking)
    if json_mode and thinking:
        print("   [info] json_mode does not support thinking mode; thinking is switched off for this request.")
        thinking = False
    if json_mode and "json" not in ((system or "") + "\n" + str(prompt)).lower():
        system = (system + "\n" if system else "") + "Output valid JSON only."

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        text, _usage = _call_with_fallback(
            _models_to_try(model_name, allow_model_fallback, fallback_models), messages,
            temperature, get_max_tokens(max_tokens), timeout, max_retries, label,
            thinking, get_thinking_budget(thinking_budget), json_mode=json_mode)
        return text
    except QwenAPIError as e:
        if raise_on_failure:
            raise
        print(e)
        return ""


# ============================================================
# Translation: the qwen-mt family (qwen-mt-plus / qwen-mt-flash / qwen-mt-lite, see TRANSLATION_MODELS)
# ============================================================
# According to the official documentation (the qwen-mt-api and machine-translation pages of the
# Model Studio docs) these models are called in a completely different way from regular chat
# models: do not use generate(), always use translate().
#   - Only a single user message is accepted: no system role, no multi-turn memory; every call is
#     an independent one-shot translation.
#   - Translation behaviour (languages, glossary, translation memory, domain hint) goes into
#     translation_options - a top-level request parameter at the same level as model / messages,
#     not inside messages (translate() takes care of that).
#   - Thinking mode is not supported: the docs do not mention it at all, so the enable_thinking key
#     is not sent, which is safer than sending false.
#   - The stream format depends on the model: qwen-mt-flash / qwen-mt-lite are "incremental" like
#     regular models; qwen-mt-plus is "non-incremental" and every chunk carries the complete
#     translation so far - handled by the full_text_stream parameter of _read_stream, so callers
#     do not have to care.
#   - The official examples use a region-specific workspace URL as base_url (for example
#     https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1); it is unclear
#     whether it is the same as the dashscope-intl.aliyuncs.com URL shared by the other Qwen
#     models. QWEN_BASE_URL is reused here first; if you run into 404 / model_not_found, point
#     QWEN_MT_BASE_URL at the workspace-specific URL (see get_mt_base_url below). On first use,
#     test with a short text to confirm the account can reach it.
def get_mt_base_url():
    """base_url of the qwen-mt family: QWEN_MT_BASE_URL from .env; if unset it shares QWEN_BASE_URL with the other models."""
    return (os.getenv("QWEN_MT_BASE_URL") or "").strip().rstrip("/") or get_base_url()


def translate(text, target_lang, source_lang="auto", model=None, terms=None, tm_list=None, domains=None,
             max_tokens=None, timeout=None, max_retries=3, raise_on_failure=True, label=None,
             allow_model_fallback=True, fallback_models=None):
    """Translate with the qwen-mt family (see the block comment above for its call limits; it is
    not the same kind of API as generate()).

    - text: the source text to translate
    - target_lang / source_lang: full language names in English, e.g. "English" / "Chinese";
      source_lang defaults to "auto" = detect the source language automatically
    - terms: glossary, format [{"source": "source term", "target": "required translation"}, ...]
      (optional; forces the given translation)
    - tm_list: translation memory, same format as terms (optional; example translations the model
      may refer to, not enforced)
    - domains: domain hint text, written in English (optional, e.g. "This is a traditional Chinese medicine book.")
    - model: optional, defaults to the first of TRANSLATION_MODELS (qwen-mt-plus); fallbacks only
      ever switch within TRANSLATION_MODELS, never to a regular chat model (different API shape,
      the switch would fail completely)
    - returns: the translated text as a string
    """
    _validate_prompt(text)
    configure()
    model_name = (model or TRANSLATION_MODELS[0]).strip()
    if model_name not in TRANSLATION_MODELS:
        raise ValueError(f"translate() only accepts models from TRANSLATION_MODELS ({', '.join(TRANSLATION_MODELS)}), "
                         f"got: {model_name}; use generate() for regular text models.")

    translation_options = {"source_lang": source_lang, "target_lang": target_lang}
    if terms:
        translation_options["terms"] = terms
    if tm_list:
        translation_options["tm_list"] = tm_list
    if domains:
        translation_options["domains"] = domains

    messages = [{"role": "user", "content": text}]
    models = [m for m in _models_to_try(model_name, allow_model_fallback, fallback_models) if m in TRANSLATION_MODELS]

    try:
        text_out, _usage = _call_with_fallback(
            models, messages, None, get_max_tokens(max_tokens), timeout, max_retries, label,
            False, None, extra_payload={"translation_options": translation_options},
            full_text_models=_MT_FULL_TEXT_MODELS, include_enable_thinking=False,
            base_url=get_mt_base_url())
        return text_out
    except QwenAPIError as e:
        if raise_on_failure:
            raise
        print(e)
        return ""


# ============================================================
# Usage 2: continuous conversation with memory (the interface mirrors ai_client's
# start_chat / send_chat_message)
# ============================================================
class QwenChat:
    """State of one conversation. The API itself is stateless: every message re-sends system + the whole history.

    - history only stores the visible text, never the reasoning trace, so thinking mode does not
      make the history grow.
    - Keep the system prompt and the start of the history unchanged so that DashScope's implicit
      cache can hit (cached input tokens are cheaper). When you need to cut tokens, trim from
      the middle / older part of the conversation and leave the fixed beginning alone.
    """

    def __init__(self, model, system=None, history=None, enable_thinking=False, thinking_budget=None,
                 max_tokens=None, temperature=None):
        self.model = model
        self.system = system
        self.history = list(history or [])
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.turns = 0
        self.last_request_tokens = 0   # prompt + completion reported by the API for the previous turn = the minimum the next turn must send

    def build_messages(self, prompt):
        messages = [{"role": "system", "content": self.system}] if self.system else []
        return messages + self.history + [{"role": "user", "content": prompt}]


def start_chat(model=None, model_name=None, history=None, system=None, enable_thinking=None,
               thinking_budget=None, max_tokens=None, temperature=None):
    """Create a conversation; keep reusing the same chat object for the same task so the memory stays coherent.
    history format: [{"role": "user"/"assistant", "content": "..."}...]."""
    configure()
    return QwenChat((model or model_name or get_default_model_name()).strip(), system=system,
                    history=history, enable_thinking=get_enable_thinking(enable_thinking),
                    thinking_budget=get_thinking_budget(thinking_budget),
                    max_tokens=get_max_tokens(max_tokens), temperature=temperature)


def send_chat_message(chat, prompt, max_retries=3, timeout=None, label=None,
                      allow_model_fallback=True, **_ignored):
    """Send a message on an existing conversation and return the text of this turn.
    The turn is only written into chat.history on success; a failure raises QwenAPIError and
    leaves the history unchanged, so it can simply be re-sent.
    (**_ignored swallows parameters that only the ai_client version has, such as initial_delay,
    so callers can swap the two modules directly.)"""
    _validate_prompt(prompt)
    configure()
    messages = chat.build_messages(prompt)
    text, usage = _call_with_fallback(
        _models_to_try(chat.model, allow_model_fallback, None), messages,
        chat.temperature, chat.max_tokens, timeout, max_retries, label or "chat",
        chat.enable_thinking, chat.thinking_budget)

    chat.history.append({"role": "user", "content": prompt})
    chat.history.append({"role": "assistant", "content": text})
    chat.turns += 1
    chat.last_request_tokens = int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    return text


def estimate_chat_tokens(chat):
    """Estimate how many tokens the NEXT message on this conversation has to send at minimum (= the whole current history).
    Uses the value actually reported by the API when there is one; only a conversation that has
    not sent anything yet (for example one created with a history) falls back to a character-count estimate."""
    if chat.last_request_tokens:
        return chat.last_request_tokens
    text = (chat.system or "") + "".join(m.get("content", "") for m in chat.history)
    return int(len(text) / 1.5)


if __name__ == "__main__":
    import sys

    # Some Windows consoles default to a legacy code page that cannot print every character.
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:2] == ["--usage"]:
        load_dotenv()
        for tier in ("high", "upper_mid", "mid", "low"):
            print(f"  -- {TIER_LABELS[tier]} --")
            for name in get_tier_models(tier):
                print(f"   [budget] {budget_status(name)}")
        sys.exit(0)
    if sys.argv[1:2] == ["--set-used"] and len(sys.argv) == 4:
        load_dotenv()
        set_used_tokens(sys.argv[2], int(sys.argv[3].replace(",", "")))
        print(f"   [ok] Ledger calibrated: {budget_status(sys.argv[2])}")
        sys.exit(0)
    configure()
    prompt_arg = " ".join(sys.argv[1:]) or "Introduce yourself in one sentence."
    print(f"   Qwen primary model: {get_default_model_name()} | fallback: {get_fallback_model_names()}"
          f" | thinking mode: {get_enable_thinking()} | {get_base_url()}")
    print(f"   Sending: {prompt_arg}\n")
    result = generate(prompt_arg)
    print("\n" + "=" * 78)
    print(result)
    print("=" * 78)
    print(f"   [usage] {usage_summary()}")
    print(f"   [budget] {budget_status(last_used_model() or get_default_model_name())}")
