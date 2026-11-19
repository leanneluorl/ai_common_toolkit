# -*- coding: utf-8 -*-
"""
ai_client.py - quota-aware Gemini API client
=================================================
A shared client module for any program that talks to the Gemini API. It centralizes:
  - API key / model configuration (multiple keys, distributed across independent quota pools)
  - Quota-aware scheduling: proactively stays under the RPM / TPM / RPD limits, and when a limit
    is hit it "waits + rotates" instead of simply stopping
  - Usage statistics: call counts and token counts per key x model, persisted across runs
  - Model fallback: when the primary model's daily RPD is used up, a fallback model takes over
  - Empty-content guard (an empty string is never sent to the API)
  - Safe response-text extraction (avoids obscure errors when the safety filter blocks a reply)

Usage 1: one-shot request (no memory; suited to one-off tasks such as outline / copy generation)
    import ai_client
    model = ai_client.get_model()
    text = ai_client.generate("Please write a paragraph about...", model=model)

Usage 2: multi-turn conversation (with memory; suited to long, coherent writing)
    import ai_client
    chat = ai_client.start_chat()
    text1 = ai_client.send_chat_message(chat, "First prompt...")
    text2 = ai_client.send_chat_message(chat, "Second prompt...")  # remembers the first one

Command line (python ai_client.py ...):
    (no arguments)      show usage statistics + model list      no API call
    --models            show the model list only                no API call
    --modelupdate       refresh the model list via list_models()  does not consume generateContent quota, safe to run often
    --modelupdate --verify
                        additionally verify "can the new model be used on the free tier", costing 1 request per model (asks first)
    --verify-only pro   verify only whether pro models can be used (1 request per model, asks first)
    --diagnose          empirically test whether the quotas of several keys are independent   [warn] consumes dozens of requests

Optional use of pro models (when quality matters most):
    pros = ai_client.get_pro_model_names()     # list of pro models verified usable; empty list if none
    text = ai_client.generate_pro(prompt)      # generate with pro; returns None when no pro model is usable
    In a test run, gemini-2.5-pro / 3-pro-preview / 3.1-pro-preview / pro-latest all had a
    free-tier quota of 0, so these two functions currently return an empty list / None.
    Pro models are deliberately kept out of the automatic fallback chain, so a downgrade never
    silently switches to a model with different cost and speed characteristics.

High / low tier classification (3.6-flash, 3.5-flash and equivalents = high; flash-lite = low):
    ai_client.get_model_tier(name)             # returns "high" / "low" / None (non-text model)
    ai_client.get_high_tier_models()           # known usable high-tier models
    ai_client.get_low_tier_models()            # known usable low-tier models
    ai_client.get_fallback_model_names(primary, tier="high")  # fallback chain of one tier only
    When the Gemini free quota is used up, another provider option is agnes_client.py in the same
    directory (Agnes 3.0 Flash / 2.5 Flash, free models only; agnes-2.5-pro is a paid model and is
    never used). Another free option is groq_client.py (Groq free tier, no credit card, quota per
    model; note this is Groq, not xAI's Grok).

------------------------------------------------------------
[important] How Gemini quota works - and how this module distributes load across it:
   Gemini quota is counted per "Google Cloud project x model", NOT per API key.
   Several keys created under the SAME project share one quota allocation, so rotating between
   them gains nothing. Load distribution therefore only helps across keys that belong to
   DIFFERENT projects (for example separate projects per workload, environment or team member,
   each with its own quota allocation).
   This module treats every key as a scheduling target: it tracks RPM / TPM / RPD per
   (quota pool x model), sends each request to the pool that can serve it soonest, and spreads
   traffic so that no single pool is driven into its limit while others sit idle.
   When it detects that a freshly rotated-in key is immediately rate limited (429) again, it
   concludes that the keys share one quota, switches to a single shared-pool mode and tells you.
   You can also declare it in .env: GEMINI_QUOTA_SCOPE=shared (one shared allocation) or
   project (each key has its own allocation, the default).
------------------------------------------------------------

.env settings (all have defaults; it runs without setting any of them):
    GEMINI_API_KEY-1 / -2 / -3 ...        multiple keys (any number, scanned automatically)
    GEMINI_MODEL_NAME                     primary model, default gemini-3-flash
    GEMINI_FALLBACK_MODELS                fallback model list (comma separated); overrides the automatic order in gemini_models.json
    GEMINI_ALLOW_LITE_FALLBACK            default 0: flash-lite is kept out of automatic fallback (when the primary and
                                          the regular flash quotas are all exhausted, AllApiKeysExhaustedError is raised
                                          instead of silently degrading to the weaker lite model).
                                          Set 1 to restore "flash exhausted -> flash-lite takes over". A primary model
                                          explicitly chosen by the caller is not affected (pass model_name to use lite).
    GEMINI_MODEL_REGISTRY_PATH            path of the model registry file (default gemini_models.json next to this file)
    GEMINI_QUOTA_SCOPE                    project (default, each key has its own allocation) / shared (all keys share one)
    GEMINI_LIMIT_RPM / _TPM / _RPD        manual override of the limits (without it the built-in Free Tier table is used)
    GEMINI_MAX_WAIT_SECONDS_PER_REQUEST   maximum cumulative wait per request in seconds (default 1800)
    GEMINI_WAIT_FOR_DAILY_RESET           true = even wait for Pacific midnight when the daily quota is exhausted (default false = stop)
    GEMINI_MIN_REQUEST_INTERVAL_SECONDS   extra global minimum interval (default 0, superseded by RPM scheduling)
    GEMINI_USAGE_LOG_PATH                 usage statistics JSON path (default _log/gemini_usage.json next to this file)
    GEMINI_USAGE_CSV_PATH                 per-call log CSV path (default _log/gemini_usage_log.csv next to this file; set to an empty string to disable)
    GEMINI_PRINT_USAGE_ON_EXIT            print a usage summary automatically when the program exits (default true)
    GEMINI_DESKTOP_NOTIFY                 show a Windows desktop notification when today's quota is exhausted (default true)
    GEMINI_DISPLAY_TZ                     IANA time zone used when showing reset times (default UTC)
"""

import atexit
import csv
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timedelta

from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions
import google.api_core.retry

_CONFIGURED = False


class AllApiKeysExhaustedError(Exception):
    """Raised only when the "daily quota" (RPD) of ALL keys x ALL models is used up.

    Note: short-term "per-minute rate limits" (RPM/TPM) never end up here - for those this module
    waits and rotates keys automatically until the call succeeds (unless the protective cap
    GEMINI_MAX_WAIT_SECONDS_PER_REQUEST is exceeded). Getting this exception means today's volume
    is really used up and retrying makes no sense."""
    pass


class RateLimitedError(Exception):
    """Raised when ALL keys x ALL models have been cycled through the configured number of times
    and per-minute rate limits (RPM/TPM) keep being hit.

    Difference from AllApiKeysExhaustedError: that one means "today's volume is used up", this one
    means "cannot squeeze in right now". The most common cause is not too few keys but a SINGLE
    REQUEST THAT IS ITSELF TOO LARGE - for example a chat that has accumulated the context of a
    whole book, so the token count of one request approaches or exceeds the TPM limit. In that case
    switching keys, switching models or waiting will not help; only making the request smaller
    (shorter context / splitting / starting a new conversation) works, so we must stop and let the
    caller know instead of cycling forever."""
    pass


# ============================================================
# Key discovery
# ============================================================
# Primary naming rule: GEMINI_API_KEY-1, GEMINI_API_KEY-2, GEMINI_API_KEY-3... (any number, gaps allowed)
_NUMBERED_API_KEY_PATTERN = re.compile(r"^GEMINI_API_KEY-(\d+)$")
# When no GEMINI_API_KEY-N is found, fall back to the legacy names (a single key)
_LEGACY_API_KEY_ENV_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]

DEFAULT_MODEL_NAME_ENV = "GEMINI_MODEL_NAME"
DEFAULT_MODEL_NAME_FALLBACK = "gemini-3.8-flash"

# Transient errors that trigger "retry after a cooldown" (timeouts, temporary service
# unavailability, ...). 429 quota exhaustion (ResourceExhausted) is handled by the quota scheduler.
# ============================================================
# Per-request timeout and "disable the SDK's built-in retry"
# ============================================================
# google-generativeai retries transient errors internally, and the timeout in request_options is
# the total deadline of the WHOLE built-in retry loop. The measured consequence is severe: with
# timeout=600 a single call may retry silently inside the SDK for 10 minutes before the error
# surfaces (the logs showed a 599-second DeadlineExceeded and a 429 that took 441 seconds to come
# back), during which this module cannot intervene at all - it can neither switch keys nor models,
# only wait. For comparison, the same request takes 2~8 seconds when it succeeds.
# So the SDK's built-in retry is switched off explicitly, so that errors surface immediately and
# this module decides whether to switch keys or models.
DEFAULT_REQUEST_TIMEOUT = 180.0


def get_request_timeout(explicit=None):
    if explicit is not None:
        return explicit
    return _env_float("GEMINI_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT)


def _build_request_options(timeout):
    opts = {"timeout": get_request_timeout(timeout)}
    if _env_bool("GEMINI_DISABLE_SDK_RETRY", True):
        # predicate always returns False = never retry, raise the exception straight up
        opts["retry"] = google.api_core.retry.Retry(predicate=lambda exc: False)
    return opts


TRANSIENT_ERRORS = (
    google.api_core.exceptions.DeadlineExceeded,
    google.api_core.exceptions.ServiceUnavailable,
    google.api_core.exceptions.InternalServerError,
    google.api_core.exceptions.Aborted,
)

# 403 PermissionDenied: "Your project has been denied access."
#
# This is completely different from 429 - a 429 means "this minute / today is used up, it will be
# fine in a moment", while a 403 means "this project itself cannot use this API": the Generative
# Language API is not enabled, the project is disabled, the key has API / origin restrictions, or
# the region does not allow this model. Waiting, switching models or retrying will not help; only
# switching to another key does.
#
# Why we count "consecutive" 403s instead of disabling on the first one:
# Occasional 403s do exist (a permission change that is still propagating), so disabling forever
# on one is too hasty. But two in a row means the project state of this key is simply like that -
# in a measured case one key hit the same 403 ten times within 20 seconds, each one burning a
# retry attempt, and it never succeeded once.
DENIED_ERRORS = (google.api_core.exceptions.PermissionDenied,)
DENY_STREAK_LIMIT = max(1, int(os.getenv("GEMINI_DENY_STREAK_LIMIT", "2")))

# key index -> reason for disabling. Valid for this run only and forgotten when the program ends -
# a project's permissions may be fixed between runs, so the disabled state is not written to a file.
_DENIED_KEYS = {}
# key index -> number of consecutive 403s currently received (reset by one success)
_DENY_STREAK = {}


def _note_denied(index, exc):
    """Record one 403. Reaching the threshold in a row disables this key until the end of the run. Returns whether it is disabled."""
    _DENY_STREAK[index] = _DENY_STREAK.get(index, 0) + 1
    name = _API_KEYS[index].env_name
    if _DENY_STREAK[index] < DENY_STREAK_LIMIT:
        print(f"   [warn] {name} got a 403 (project denied access), switching to the next key. "
              f"(consecutive {_DENY_STREAK[index]}/{DENY_STREAK_LIMIT})")
        return False
    if index not in _DENIED_KEYS:
        _DENIED_KEYS[index] = str(exc)[:200]
        print(f"   [disabled] {name} got {_DENY_STREAK[index]} consecutive 403s; this key is not used again in this run.\n"
              f"      A 403 is a project-level problem; waiting and switching models do not help. Check the project of this key:\n"
              f"      is the Generative Language API enabled, is billing attached, does the key have API / origin restrictions.")
    return True


def _live_key_indexes():
    """Indexes of the keys that can still be used in this run."""
    return [i for i in range(len(_API_KEYS)) if i not in _DENIED_KEYS]


# ============================================================
# Free Tier limits table (counted per "project x model"; RPD resets at midnight US Pacific time)
#    The numbers change with Google's policy; the most accurate source is https://aistudio.google.com/rate-limit
#    To override, set GEMINI_LIMIT_RPM / GEMINI_LIMIT_TPM / GEMINI_LIMIT_RPD in .env
# ============================================================
# [note] The numbers here are only conservative starting values used until real limits are
#    observed; most free-quota tables circulating online are wrong. The real limits are learned
#    directly from the `quota_id` + `limit:` in Google's response when a 429 is received
#    (see _learn_limit_from_429), saved in the learned_limits of gemini_usage.json, and from then
#    on the observed values take precedence.
MODEL_FREE_TIER_LIMITS = {
    "gemini-3.6-flash":        {"rpm": 10, "tpm": 250_000, "rpd": 20},
    "gemini-3.5-flash":        {"rpm": 10, "tpm": 250_000, "rpd": 20},
    "gemini-3.5-flash-lite":   {"rpm": 15, "tpm": 250_000, "rpd": 50},
    "gemini-3.1-flash-lite":   {"rpm": 15, "tpm": 250_000, "rpd": 50},
    "gemini-3.1-pro":          {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3-pro":            {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-2.5-flash":        {"rpm": 10, "tpm": 250_000, "rpd": 50},
    "gemini-2.5-pro":          {"rpm": 5,  "tpm": 250_000, "rpd": 20},
}
DEFAULT_FREE_TIER_LIMITS = {"rpm": 10, "tpm": 250_000, "rpd": 20}

# When the primary model's daily quota is used up, these fallback models are tried in order
# (different model = different quota pool, which is the biggest lever of the free quota).
# This is only the built-in default used before a model registry file exists; running
# `python ai_client.py --modelupdate` regenerates the list via list_models(), and from then on the
# registry file takes precedence.
DEFAULT_FALLBACK_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]

# ============================================================
# Model registry (gemini_models.json)
#    Updated with `python ai_client.py --modelupdate`.
#    list_models() is a metadata endpoint and does not consume the free generateContent quota, so
#    it is safe to run often. "Can this model actually be used on the free tier" cannot be found
#    out for free - a real request has to be sent - so that part additionally requires --verify,
#    and only verifies new models that have not been verified yet.
# ============================================================
# Names containing these markers are not plain text-generation models and are kept out of automatic fallback
_NON_TEXT_MODEL_MARKERS = (
    "image", "tts", "audio", "lyria", "veo", "imagen", "nano-banana",
    "robotics", "embedding", "aqa", "computer-use", "live", "omni",
    "deep-research", "antigravity", "customtools", "transcribe",
)


def _classify_model_kind(name):
    lowered = _normalize_model_name(name).lower()
    for marker in _NON_TEXT_MODEL_MARKERS:
        if marker in lowered:
            return "non-text"
    if lowered.startswith("gemma"):
        return "gemma"
    return "text"


def _is_pro_model(name):
    """Pro family (better reasoning, but usually a smaller free quota).
    Deliberately kept out of the automatic fallback chain - silently switching to pro on a
    downgrade would upset cost and speed expectations; to use pro the caller must ask for it
    explicitly (see get_pro_model_names / generate_pro)."""
    lowered = _normalize_model_name(name).lower()
    return "pro" in lowered and _classify_model_kind(name) == "text"


# ============================================================
# High / low tier classification (only regular flash / pro models with kind="text")
#    High: 3.6-flash, 3.5-flash and equivalent regular flash / pro - quality first.
#    Low: the flash-lite family - usually a looser quota but weaker reasoning.
#    Gemma (kind="gemma") and non-text models such as TTS / image are not in the automatic
#    fallback chain anyway and have no tier.
# ============================================================
def _model_tier(name):
    """Returns "high" / "low" / None (non-text model, including Gemma)."""
    if _classify_model_kind(name) != "text":
        return None
    lowered = _normalize_model_name(name).lower()
    if "lite" in lowered:
        return "low"
    return "high"


def get_model_tier(name):
    """Return the tier of the given model: "high" / "low" / None (non-text model, for example TTS or image)."""
    return _model_tier(name)


def _models_by_tier(tier):
    configure()
    models = _MODEL_REGISTRY.get("models") or {}
    names = [n for n, e in models.items()
             if e.get("kind") == "text"
             and e.get("status") not in ("unavailable", "retired")
             and _model_tier(n) == tier]
    names.sort(key=_model_sort_key)
    return names


def get_high_tier_models():
    """Known usable high-tier text models (regular flash / pro such as 3.6-flash, 3.5-flash), in automatic order."""
    return _models_by_tier("high")


def get_low_tier_models():
    """Known usable low-tier text models (the flash-lite family), in automatic order."""
    return _models_by_tier("low")


def _model_sort_key(name):
    """Automatic fallback order: flash -> flash-lite -> pro -> gemma; within a kind, newer versions first."""
    lowered = _normalize_model_name(name).lower()
    if lowered.startswith("gemma"):
        tier = 3
    elif "flash-lite" in lowered:
        tier = 1
    elif "flash" in lowered:
        tier = 0
    elif "pro" in lowered:
        tier = 2
    else:
        tier = 4
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", lowered)
    version = float(m.group(1)) if m else 0.0
    is_preview = 1 if "preview" in lowered else 0
    return (tier, is_preview, -version, lowered)


# ============================================================
# US Pacific time helpers (the Gemini Free Tier daily quota resets at midnight Pacific time)
# ============================================================
def _pt_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles"))
    except Exception:
        from datetime import timezone
        return datetime.now(timezone.utc) - timedelta(hours=8)


def _pt_today_str():
    return _pt_now().strftime("%Y-%m-%d")


def _pt_to_local_str(dt_pt):
    """Convert a Pacific-time datetime into a display string in the configured time zone
    (GEMINI_DISPLAY_TZ, default UTC). Showing only Pacific time makes it easy to lose track of
    when the reset really happens in local time (PT midnight shifts with US daylight saving time),
    so the reset time is also shown in a time zone the operator picks. On failure (for example
    zoneinfo is unavailable) an empty string is returned and the main flow is unaffected."""
    try:
        from zoneinfo import ZoneInfo
        tz_name = (os.getenv("GEMINI_DISPLAY_TZ") or "UTC").strip() or "UTC"
        return dt_pt.astimezone(ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _seconds_until_pt_midnight():
    now = _pt_now()
    next_reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1.0, (next_reset - now).total_seconds())


def _fmt_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


# ============================================================
# Reading limit settings
# ============================================================
def _normalize_model_name(name):
    n = str(name or "").strip()
    if n.startswith("models/"):
        n = n[len("models/"):]
    return n


# Real limits learned from actual 429 responses: {model_name: {"rpm"/"tpm"/"rpd": value}}, saved in gemini_usage.json
_LEARNED_LIMITS = {}


def get_model_limits(model_name, include_learned=True):
    """Get the Free Tier limits dict of a model: {"rpm":..,"tpm":..,"rpd":..}

    Priority: manual .env override > observed values learned from 429 > built-in conservative defaults."""
    n = _normalize_model_name(model_name).lower()
    base = None
    if n in MODEL_FREE_TIER_LIMITS:
        base = MODEL_FREE_TIER_LIMITS[n]
    else:
        # Match by "longest prefix" so that suffixed names such as gemini-2.5-flash-preview-xxx also resolve
        for prefix in sorted(MODEL_FREE_TIER_LIMITS, key=len, reverse=True):
            if n.startswith(prefix):
                base = MODEL_FREE_TIER_LIMITS[prefix]
                break
    limits = dict(base or DEFAULT_FREE_TIER_LIMITS)
    if include_learned:
        limits.update(_LEARNED_LIMITS.get(_normalize_model_name(model_name), {}))
    for field, env_name in (("rpm", "GEMINI_LIMIT_RPM"), ("tpm", "GEMINI_LIMIT_TPM"), ("rpd", "GEMINI_LIMIT_RPD")):
        raw = os.getenv(env_name)
        if raw and str(raw).strip():
            try:
                limits[field] = int(float(raw))
            except ValueError:
                pass
    return limits


def _env_float(name, default):
    try:
        raw = os.getenv(name)
        return float(raw) if raw and str(raw).strip() else default
    except (TypeError, ValueError):
        return default


def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def get_default_model_name():
    """Read the model name configured in .env; use the default if not set"""
    return _normalize_model_name(os.getenv(DEFAULT_MODEL_NAME_ENV) or DEFAULT_MODEL_NAME_FALLBACK)


def get_fallback_model_names(primary, tier=None):
    """Fallback model list (excluding the primary model itself). They take over in order when the primary model's quota is used up.

    Priority: GEMINI_FALLBACK_MODELS in .env > fallback_order in gemini_models.json > built-in defaults.
    Whatever the source, models known to be unusable (404 or free-tier limit=0) are filtered out.
    tier: optional, "high" or "low"; keep only models of that tier (see get_model_tier).
    """
    raw = os.getenv("GEMINI_FALLBACK_MODELS")
    if raw is None:
        names = _MODEL_REGISTRY.get("fallback_order") or list(DEFAULT_FALLBACK_MODELS)
    elif str(raw).strip() == "":
        return []  # explicitly set to an empty string = model fallback disabled
    else:
        names = [_normalize_model_name(x) for x in str(raw).split(",")]

    primary = _normalize_model_name(primary)
    allow_lite = _env_bool("GEMINI_ALLOW_LITE_FALLBACK", False)
    seen, result = {primary}, []
    for n in names:
        n = _normalize_model_name(n)
        if not n or n in seen:
            continue
        seen.add(n)
        if _model_status(n) in ("unavailable", "retired"):
            continue
        if tier is not None and _model_tier(n) != tier:
            continue
        if _classify_model_kind(n) != "text":
            continue   # old registry records may still be labelled text (for example transcribe); the fallback chain holds only text models meant for writing
        # lite is clearly weaker (copywriting and proofreading get worse), so regardless of where the fallback list comes from it is never degraded to automatically
        if not allow_lite and _model_tier(n) == "low":
            continue
        result.append(n)
    return result


def _model_status(name):
    entry = (_MODEL_REGISTRY.get("models") or {}).get(_normalize_model_name(name))
    return (entry or {}).get("status", "unknown")


# ============================================================
# Usage and quota state
# ============================================================
class _Bucket:
    """Usage and cooldown state of one quota pool x one model.

    Gemini quota is counted per "project x model", so switching one key to another model means a brand-new allocation."""

    def __init__(self, model_name):
        self.model_name = model_name
        self.request_times = deque()   # monotonic timestamps, kept for 60 seconds only (guards RPM)
        self.token_events = deque()    # (monotonic, tokens), kept for 60 seconds only (guards TPM)
        self.cooldown_until = 0.0      # monotonic; cooldown deadline after being blocked by a 429
        self.cooldown_reason = ""
        self.day = _pt_today_str()
        self.day_requests = 0
        self.day_tokens = 0
        self.day_blocked = False       # today's RPD is confirmed to be used up
        self.total_requests = 0
        self.total_success = 0
        self.total_429 = 0
        self.total_errors = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_thought_tokens = 0
        self.total_tokens = 0
        self.unknown_429_streak = 0
        self.day_labeled_429_streak = 0   # consecutive 429s carrying a PerDay label (used to decide "waiting is useless")

    def roll_day(self):
        today = _pt_today_str()
        if today != self.day:
            self.day = today
            self.day_requests = 0
            self.day_tokens = 0
            self.day_blocked = False
            self.cooldown_until = 0.0
            self.cooldown_reason = ""

    def prune(self, now):
        while self.request_times and now - self.request_times[0] > 60.0:
            self.request_times.popleft()
        while self.token_events and now - self.token_events[0][0] > 60.0:
            self.token_events.popleft()

    def tokens_in_window(self):
        return sum(tok for _, tok in self.token_events)

    def rpd_left(self, limits):
        self.roll_day()
        if self.day_blocked:
            return 0
        rpd = limits.get("rpd")
        if rpd is None:
            return 10 ** 9   # no limit configured = treated as unlimited
        # An rpd of 0 is meaningful: it means this model has no quota on the free tier at all
        # (measured for the pro family) and must not be treated as "no limit".
        return max(0, int(rpd) - self.day_requests)

    def ready_at(self, now, limits, est_tokens):
        """Return the earliest monotonic time at which this quota pool can send the next request."""
        self.roll_day()
        self.prune(now)
        ready = self.cooldown_until

        rpm = limits.get("rpm") or 0
        if rpm > 0 and len(self.request_times) >= rpm:
            ready = max(ready, self.request_times[0] + 60.5)

        tpm = limits.get("tpm") or 0
        if tpm > 0 and self.token_events:
            if self.tokens_in_window() + est_tokens > tpm:
                ready = max(ready, self.token_events[0][0] + 60.5)

        return ready


class _QuotaOwner:
    """One independent quota pool. Normally = one key (backed by one Google Cloud project);
    in shared-quota mode all keys share a single _QuotaOwner."""

    def __init__(self, label, fingerprint):
        self.label = label
        self.fingerprint = fingerprint
        self.buckets = {}

    def bucket(self, model_name):
        model_name = _normalize_model_name(model_name)
        if model_name not in self.buckets:
            self.buckets[model_name] = _Bucket(model_name)
        return self.buckets[model_name]


class _ApiKey:
    def __init__(self, env_name, key):
        self.env_name = env_name
        self.key = key
        self.fingerprint = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
        self.owner = _QuotaOwner(env_name, self.fingerprint)


_API_KEYS = []            # list[_ApiKey]
_CURRENT_KEY_INDEX = 0
_SHARED_OWNER = _QuotaOwner("(all keys share one quota)", "shared")
_QUOTA_SCOPE = "project"  # "project" = each key has its own allocation; "shared" = all keys share one
_SHARED_WARNED = False
_SUSPECT_SHARED_FINGERPRINTS = {}  # fingerprint -> monotonic time of the latest "429 although usage was still low"
_MODEL_CACHE = {}         # (fingerprint, model_name) -> GenerativeModel
_SESSION_STARTED_AT = time.time()
_EXIT_HOOK_REGISTERED = False


def _owner_of(api_key):
    return _SHARED_OWNER if _QUOTA_SCOPE == "shared" else api_key.owner


def _all_owners():
    if _QUOTA_SCOPE == "shared":
        return [_SHARED_OWNER]
    return [k.owner for k in _API_KEYS]


# ============================================================
# Usage statistics persistence (accumulated across runs; the key point is that RPD must be remembered)
# ============================================================
def _usage_json_path():
    raw = os.getenv("GEMINI_USAGE_LOG_PATH")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip()
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_log")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "gemini_usage.json")


def _usage_csv_path():
    raw = os.getenv("GEMINI_USAGE_CSV_PATH")
    if raw is not None:
        raw = str(raw).strip()
        if raw == "":
            return None  # explicitly disabled
        return raw
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_log")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "gemini_usage_log.csv")


_PERSISTED_FIELDS = (
    "day_requests", "day_tokens", "day_blocked",
    "total_requests", "total_success", "total_429", "total_errors",
    "total_prompt_tokens", "total_output_tokens", "total_thought_tokens", "total_tokens",
)


def _load_usage_state():
    """Restore cumulative usage from JSON. Today's (Pacific time) day_* counters are restored only if the date matches."""
    path = _usage_json_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"   [warn] Failed to read the usage statistics file (counting starts over): {e}")
        return

    for model_name, dims in (data.get("learned_limits") or {}).items():
        _LEARNED_LIMITS[model_name] = {k: int(v) for k, v in dims.items()}

    today = _pt_today_str()
    saved_day = data.get("pt_date")
    owners_by_fp = {o.fingerprint: o for o in [k.owner for k in _API_KEYS] + [_SHARED_OWNER]}

    for fp, entry in (data.get("owners") or {}).items():
        owner = owners_by_fp.get(fp)
        if owner is None:
            continue  # this key is no longer in .env
        for model_name, stats in (entry.get("models") or {}).items():
            b = owner.bucket(model_name)
            for field in _PERSISTED_FIELDS:
                if field.startswith("day_") and saved_day != today:
                    continue
                if field in stats:
                    setattr(b, field, stats[field])
            b.day = today
            if saved_day != today:
                b.day_blocked = False
            elif b.day_blocked:
                # already judged earlier today that RPD is used up -> cool down until Pacific midnight
                b.cooldown_until = time.monotonic() + _seconds_until_pt_midnight()
                b.cooldown_reason = "daily quota (RPD) exhausted"


def _save_usage_state():
    path = _usage_json_path()
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pt_date": _pt_today_str(),
        "quota_scope": _QUOTA_SCOPE,
        "learned_limits": _LEARNED_LIMITS,
        "owners": {},
    }
    for api_key in _API_KEYS:
        owner = api_key.owner
        models = {m: {f: getattr(b, f) for f in _PERSISTED_FIELDS} for m, b in owner.buckets.items()}
        if models:
            payload["owners"][owner.fingerprint] = {"env_name": api_key.env_name, "models": models}
    if _SHARED_OWNER.buckets:
        payload["owners"]["shared"] = {
            "env_name": _SHARED_OWNER.label,
            "models": {m: {f: getattr(b, f) for f in _PERSISTED_FIELDS} for m, b in _SHARED_OWNER.buckets.items()},
        }
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"   [warn] Failed to write the usage statistics file: {e}")


# ============================================================
# Reading / writing / updating the model registry file
# ============================================================
_MODEL_REGISTRY = {"models": {}, "fallback_order": []}


def _model_registry_path():
    raw = os.getenv("GEMINI_MODEL_REGISTRY_PATH")
    if raw is not None and str(raw).strip() != "":
        return str(raw).strip()
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemini_models.json")


def load_model_registry():
    global _MODEL_REGISTRY
    path = _model_registry_path()
    if not os.path.exists(path):
        _MODEL_REGISTRY = {"models": {}, "fallback_order": []}
        return _MODEL_REGISTRY
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _MODEL_REGISTRY = {
            "updated_at": data.get("updated_at"),
            "models": data.get("models") or {},
            "fallback_order": data.get("fallback_order") or [],
        }
    except Exception as e:
        print(f"   [warn] Failed to read the model registry file (using built-in defaults): {e}")
        _MODEL_REGISTRY = {"models": {}, "fallback_order": []}
    return _MODEL_REGISTRY


def save_model_registry():
    path = _model_registry_path()
    payload = dict(_MODEL_REGISTRY)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        print(f"   [warn] Failed to write the model registry file: {e}")


def mark_model_unavailable(name, reason):
    """Record in the registry a model that turned out to be unusable after a real call, so automatic fallback skips it from now on."""
    name = _normalize_model_name(name)
    models = _MODEL_REGISTRY.setdefault("models", {})
    entry = models.setdefault(name, {"kind": _classify_model_kind(name)})
    if entry.get("status") == "unavailable" and entry.get("status_reason") == reason:
        return
    entry["status"] = "unavailable"
    entry["status_reason"] = reason
    entry["checked_at"] = datetime.now().isoformat(timespec="seconds")
    _MODEL_REGISTRY["fallback_order"] = [
        m for m in (_MODEL_REGISTRY.get("fallback_order") or []) if m != name
    ]
    print(f"   [registry] Marked {name} as unavailable ({reason}) and saved it to the registry; automatic fallback will not use it again.")
    save_model_registry()


def mark_model_ok(name):
    name = _normalize_model_name(name)
    entry = _MODEL_REGISTRY.setdefault("models", {}).setdefault(name, {"kind": _classify_model_kind(name)})
    if entry.get("status") != "ok":
        entry["status"] = "ok"
        entry.pop("status_reason", None)
        entry["checked_at"] = datetime.now().isoformat(timespec="seconds")
        save_model_registry()


def _rebuild_fallback_order(limit=8):
    """Pick the usable text models from the registry and order them into the automatic fallback chain.

    Stable releases of flash / flash-lite with an explicit version number go first, newest
    version first, even if they have not been verified yet - in practice older flash models
    tend to be overloaded (503) as a whole, while newer models are usually less crowded.
    Previews, -latest aliases and other special models stay at the back as a last resort;
    an unverified model only reveals its status when it is really tried, and failures are
    recorded automatically."""
    models = _MODEL_REGISTRY.get("models") or {}
    usable = [
        name for name, entry in models.items()
        if entry.get("kind") == "text"
        and entry.get("status") not in ("unavailable", "retired")
        and not _is_pro_model(name)   # pro is listed separately and kept out of automatic fallback
        and (_model_tier(name) != "low" or _env_bool("GEMINI_ALLOW_LITE_FALLBACK", False))   # lite is kept out of fallback
    ]

    def _key(n):
        tier, is_preview, neg_version, lowered = _model_sort_key(n)
        secondary = bool(is_preview or neg_version == 0 or tier > 1)
        return (secondary, tier, neg_version, 0 if models[n].get("status") == "ok" else 1, lowered)

    usable.sort(key=_key)
    _MODEL_REGISTRY["fallback_order"] = usable[:limit]
    _rebuild_pro_models()
    return _MODEL_REGISTRY["fallback_order"]


def _rebuild_pro_models():
    """Build the list of verified-usable pro models, for callers that opt in."""
    models = _MODEL_REGISTRY.get("models") or {}
    pros = [n for n, e in models.items() if _is_pro_model(n) and e.get("status") == "ok"]
    pros.sort(key=_model_sort_key)
    _MODEL_REGISTRY["pro_models"] = pros
    return pros


def get_pro_model_names():
    """List of verified-usable pro models (for quality-first use; an empty list if none is usable).

        pros = ai_client.get_pro_model_names()
        if pros:
            text = ai_client.generate(prompt, model_name=pros[0], allow_model_fallback=False)
    """
    configure()
    return list(_MODEL_REGISTRY.get("pro_models") or [])


def generate_pro(prompt, fallback_to_flash=False, **kwargs):
    """Generate with a pro model. Returns None when no verified-usable pro model exists (the caller decides whether to use flash instead).

    - By default it only degrades between pro models (when one is used up, the next pro takes
      over) and never silently drops to flash, because asking for pro is usually about quality.
    - fallback_to_flash=True allows falling back to the regular flash chain once every pro is out of quota.
    """
    pros = get_pro_model_names()
    if not pros:
        models = _MODEL_REGISTRY.get("models") or {}
        tested = [n for n, e in models.items()
                  if _is_pro_model(n) and e.get("status") == "unavailable"]
        if tested:
            print(f"   [warn] No usable pro model: all {len(tested)} were tested and have a free-tier quota of 0. "
                  "(Upgrade to a paid tier and re-run --verify-only pro to enable them.)")
        else:
            print("   [warn] Pro models have not been verified yet. Run first: python ai_client.py --verify-only pro")
        return None
    chain = pros[1:]
    if fallback_to_flash:
        chain = chain + get_fallback_model_names(pros[0])
    return generate(prompt, model_name=pros[0], fallback_models=chain, **kwargs)


def verify_models(spec=None, assume_yes=False):
    """Verification only; does not re-fetch list_models(). See resolve_verify_targets for spec."""
    configure()
    load_model_registry()
    models = _MODEL_REGISTRY.setdefault("models", {})
    if not models:
        print("   There is no model registry yet. Run first: python ai_client.py --modelupdate")
        return
    targets = resolve_verify_targets(models, spec)
    _verify_models(models, targets, assume_yes=assume_yes)
    print_model_registry()


def refresh_model_registry(verify=False, verify_spec=None, assume_yes=False):
    """Refresh the model registry.

    - By default it only calls list_models(): a metadata endpoint that does NOT consume the
      free generateContent quota, so it is safe to run any time. It tells you which models
      exist and whether they support generateContent, but not "can it be used on the free
      tier" (for example gemini-2.0-flash exists but has a free quota of 0).
    - verify=True additionally sends one minimal request for every "not yet verified" new
      model to find out; each model costs 1 request (the cost is listed and confirmation is
      requested first).
    """
    configure()
    load_model_registry()
    models = _MODEL_REGISTRY.setdefault("models", {})
    today = datetime.now().strftime("%Y-%m-%d")

    print("\n" + "=" * 78)
    print(" Refreshing model registry (list_models(), consumes no generateContent quota)")
    print("=" * 78)

    listed = None
    last_error = None
    for idx in range(len(_API_KEYS)):
        try:
            genai.configure(api_key=_API_KEYS[idx].key)
            listed = list(genai.list_models())
            print(f"   Fetched the list with {_API_KEYS[idx].env_name}: {len(listed)} models in total.")
            break
        except Exception as e:
            last_error = e
            print(f"   [warn] {_API_KEYS[idx].env_name} failed to fetch the list ({type(e).__name__}), trying the next key...")
    if listed is None:
        print(f"   [error] No key could fetch the model list: {last_error}")
        return _MODEL_REGISTRY

    seen_names = set()
    new_names = []
    for m in listed:
        methods = list(getattr(m, "supported_generation_methods", []) or [])
        if "generateContent" not in methods:
            continue
        name = _normalize_model_name(getattr(m, "name", ""))
        if not name:
            continue
        seen_names.add(name)
        entry = models.get(name)
        is_new = entry is None
        if is_new:
            entry = {"status": "unverified"}
            new_names.append(name)
        entry["kind"] = _classify_model_kind(name)
        entry["display_name"] = getattr(m, "display_name", "")
        entry["input_token_limit"] = getattr(m, "input_token_limit", None)
        entry["output_token_limit"] = getattr(m, "output_token_limit", None)
        entry["last_seen"] = today
        if entry.get("status") == "retired":
            entry["status"] = "unverified"
            entry.pop("status_reason", None)
        models[name] = entry

    # Models Google has withdrawn: keep the record but mark them retired and drop them from fallback
    retired = []
    for name, entry in models.items():
        if name not in seen_names and entry.get("status") != "retired":
            entry["status"] = "retired"
            entry["status_reason"] = "no longer returned by list_models()"
            retired.append(name)

    # Automatically mark models that have succeeded before as ok (inferred from usage stats, no API call needed)
    promoted = []
    for owner in [k.owner for k in _API_KEYS] + [_SHARED_OWNER]:
        for model_name, bucket in owner.buckets.items():
            if bucket.total_success > 0:
                entry = models.get(model_name)
                if entry and entry.get("status") == "unverified":
                    entry["status"] = "ok"
                    entry["status_reason"] = "has succeeded before"
                    promoted.append(model_name)

    if verify:
        _verify_models(models, resolve_verify_targets(models, verify_spec), assume_yes=assume_yes)

    order = _rebuild_fallback_order()
    save_model_registry()

    text_models = {n: e for n, e in models.items() if e.get("kind") == "text"}
    print(f"\n   {len(text_models)} text-generation models:")
    for name in sorted(text_models, key=_model_sort_key):
        e = text_models[name]
        icon = _STATUS_ICONS.get(e.get("status"), "[?]")
        tier_label = {"high": "high", "low": "low"}.get(_model_tier(name), "")
        note = e.get("status_reason") or ""
        print(f"     {icon:<8}{name:<34}{e.get('status',''):<12}{tier_label:<6}{note}")
    if new_names:
        print(f"\n   New models this time: {', '.join(sorted(new_names))}")
    if retired:
        print(f"   Withdrawn models: {', '.join(sorted(retired))}")
    if promoted:
        print(f"   Marked usable based on past successes: {', '.join(sorted(set(promoted)))}")
    print(f"\n   Automatic fallback order (takes over in sequence when the primary model has no quota):\n     {' -> '.join(order)}")
    print(f"   Registry file: {_model_registry_path()}")
    if not verify and any(e.get("status") == "unverified" for e in text_models.values()):
        unverified = [n for n, e in text_models.items() if e.get("status") == "unverified"]
        print(f"\n   {len(unverified)} models have not been verified for free-tier use yet.")
        print("      list_models() cannot tell; a real request has to be sent. To verify, run:")
        print(f"      python ai_client.py --modelupdate --verify   (consumes {len(unverified)} requests)")
    print("=" * 78 + "\n")
    return _MODEL_REGISTRY


_STATUS_ICONS = {"ok": "[ok]", "unverified": "[?]", "unavailable": "[x]", "retired": "[gone]"}


def resolve_verify_targets(models, spec=None, include_verified=False):
    """Resolve the --verify-only argument into the list of model names that will actually be verified.

    Supported spec values:
      None           -> every text model that has not been verified yet
      "pro"          -> all pro models
      "flash"        -> all flash / flash-lite models
      "a,b,c"        -> the named models (a models/ prefix is accepted)
    """
    text_models = {n: e for n, e in models.items() if e.get("kind") == "text"}

    if spec is None:
        picked = [n for n, e in text_models.items() if e.get("status") == "unverified"]
    else:
        keyword = str(spec).strip().lower()
        if keyword == "pro":
            picked = [n for n in text_models if _is_pro_model(n)]
        elif keyword in ("flash", "flash-lite"):
            picked = [n for n in text_models if "flash" in n and not _is_pro_model(n)]
        elif keyword == "all":
            picked = list(text_models)
        else:
            picked = []
            for raw in str(spec).split(","):
                name = _normalize_model_name(raw)
                if not name:
                    continue
                if name not in models:
                    print(f"   [warn] {name} is not in the registry; run --modelupdate to refresh it, or check the spelling.")
                    continue
                picked.append(name)

    if not include_verified:
        picked = [n for n in picked if models[n].get("status") != "ok"]
    # No point spending quota again on models already known to be unusable
    picked = [n for n in picked if models[n].get("status") not in ("unavailable", "retired")]
    return sorted(set(picked), key=_model_sort_key)


def _verify_models(models, targets, assume_yes=False):
    """Send one minimal request to each given model to confirm free-tier availability. Costs 1 request per model."""
    if not targets:
        print("   No models need verification; no quota will be consumed.")
        return

    print(f"\n   [warn] About to send 1 minimal request to each of {len(targets)} models, consuming {len(targets)} free requests in total:")
    for n in targets:
        print(f"      - {n}")
    if not assume_yes:
        answer = input("   Proceed with verification? (y/N): ").strip().lower()
        if answer not in ("y", "yes"):
            print("   Verification cancelled; no quota was consumed.")
            return

    for name in targets:
        entry = models[name]
        try:
            genai.configure(api_key=_API_KEYS[_CURRENT_KEY_INDEX].key)
            resp = genai.GenerativeModel(name).generate_content("ok", request_options=_build_request_options(60))
            tokens = getattr(getattr(resp, "usage_metadata", None), "total_token_count", 0)
            entry["status"] = "ok"
            entry.pop("status_reason", None)
            print(f"      [ok] {name:<34}usable (tokens={tokens})")
        except google.api_core.exceptions.NotFound as e:
            entry["status"] = "unavailable"
            entry["status_reason"] = "404 model not usable (may be closed to new users)"
            print(f"      [x]  {name:<34}404 not usable")
        except google.api_core.exceptions.ResourceExhausted as e:
            learned = _learn_limit_from_429(name, e)
            if learned and learned[1] == 0:
                entry["status"] = "unavailable"
                entry["status_reason"] = f"free-tier {learned[0].upper()} quota is 0"
                print(f"      [x]  {name:<34}no free-tier quota (limit=0)")
            else:
                entry["status"] = "ok"
                entry["status_reason"] = "hit a rate limit while verifying, but the model itself is usable"
                print(f"      [ok] {name:<34}usable (happened to hit a rate limit while verifying)")
        except Exception as e:
            entry["status"] = "unverified"
            entry["status_reason"] = f"{type(e).__name__} while verifying"
            print(f"      [warn] {name:<34}{type(e).__name__}: {str(e)[:60]}")
        entry["checked_at"] = datetime.now().isoformat(timespec="seconds")
        save_model_registry()

    _rebuild_fallback_order()
    save_model_registry()


_CSV_HEADER = [
    "time", "key_env", "key_fingerprint", "model", "status",
    "prompt_tokens", "output_tokens", "thought_tokens", "total_tokens",
    "latency_seconds", "label", "note",
]


def _append_csv_row(row):
    path = _usage_csv_path()
    if not path:
        return
    try:
        is_new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(_CSV_HEADER)
            writer.writerow(row)
    except Exception as e:
        print(f"   [warn] Failed to write the per-call CSV log: {e}")


# ============================================================
# Initialization
# ============================================================
def _discover_api_keys():
    """Scan the environment for every GEMINI_API_KEY-N (sorted by number); fall back to the legacy single-key names if none is found."""
    numbered = []
    for name, value in os.environ.items():
        m = _NUMBERED_API_KEY_PATTERN.match(name)
        if m and value and value.strip():
            numbered.append((int(m.group(1)), name, value.strip()))
    numbered.sort(key=lambda item: item[0])
    keys = [(name, value) for _, name, value in numbered]

    if not keys:
        for name in _LEGACY_API_KEY_ENV_NAMES:
            value = os.getenv(name)
            if value and value.strip():
                keys.append((name, value.strip()))
                break

    return keys


def configure(env_path=None, force=False):
    """Load .env, set up the keys and restore historical usage. Safe to call repeatedly (by default it only really configures once)."""
    global _CONFIGURED, _API_KEYS, _CURRENT_KEY_INDEX, _QUOTA_SCOPE, _MODEL_CACHE, _EXIT_HOOK_REGISTERED
    if _CONFIGURED and not force:
        return

    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()

    raw_keys = _discover_api_keys()
    if not raw_keys:
        raise EnvironmentError(
            "[error] No API key found. Set at least one GEMINI_API_KEY-1 in .env "
            "(add GEMINI_API_KEY-2, -3... as needed)."
        )

    _API_KEYS = [_ApiKey(name, value) for name, value in raw_keys]
    _MODEL_CACHE = {}
    _CURRENT_KEY_INDEX = 0

    scope = str(os.getenv("GEMINI_QUOTA_SCOPE", "project")).strip().lower()
    _QUOTA_SCOPE = "shared" if scope in ("shared", "same", "one", "single") else "project"

    _load_usage_state()
    load_model_registry()

    genai.configure(api_key=_API_KEYS[0].key)
    _CONFIGURED = True

    primary = get_default_model_name()
    limits = get_model_limits(primary)
    if len(_API_KEYS) > 1:
        print(f"   [keys] Loaded {len(_API_KEYS)} API keys, currently using: {_API_KEYS[0].env_name}")
    print(f"   [model] Primary model: {primary} (reference free-tier limits RPM={limits['rpm']} / TPM={limits['tpm']:,} / RPD={limits['rpd']})")
    if _QUOTA_SCOPE == "shared":
        print("   [warn] Configured in 'shared quota' mode: all keys are treated as one allocation (use this when every key belongs to the same GCP project).")
    print(f"   [quota] Estimated requests still available today: {_describe_remaining_capacity(primary)}")

    if not _EXIT_HOOK_REGISTERED:
        _EXIT_HOOK_REGISTERED = True
        atexit.register(_on_exit)


def _on_exit():
    try:
        _save_usage_state()
        if not _env_bool("GEMINI_PRINT_USAGE_ON_EXIT", True):
            return
        if any(b.total_requests for o in _all_owners() for b in o.buckets.values()):
            print_usage_summary()
    except Exception:
        pass


# ============================================================
# Quota scheduler: pick a key that can serve the request now, wait if necessary
# ============================================================
def _start_heartbeat(key_name, model_name):
    """While waiting for an API response, print a "still running" line at intervals.

    A large prompt can take several minutes to come back, and with no output at all the program
    looks hung. A heartbeat message tells "still waiting for the response" apart from "really stuck".
    Set GEMINI_HEARTBEAT_SECONDS=0 to turn it off. Returns an Event; the caller must .set() it when done."""
    interval = _env_float("GEMINI_HEARTBEAT_SECONDS", 30.0)
    stop = threading.Event()
    if interval <= 0:
        return stop
    started = time.monotonic()

    def _beat():
        while not stop.wait(interval):
            elapsed = time.monotonic() - started
            print(f"   [wait] {key_name} x {model_name} still waiting for the API response... "
                  f"({_fmt_duration(elapsed)} elapsed)")

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def _estimate_prompt_tokens(prompt):
    """Rough token estimate, used only as an up-front TPM guard (the real number corrects it from usage_metadata after the response).
    CJK text is about 1 token per character and English about 1 token per 4 characters; this
    deliberately uses the conservative 1 token per 2 characters."""
    try:
        return max(32, int(len(str(prompt)) / 2))
    except Exception:
        return 512


def _select_key(model_name, est_tokens):
    """Returns (key_index, wait_seconds).

    The scheduling decision: among all keys (quota pools) that still have daily capacity, choose
    the one that becomes ready soonest; ties favour the current key to avoid pointless hopping.
    wait_seconds > 0 means the short-term limit of that key has to clear first.
    If the DAILY quota of every key is used up, returns (None, None)."""
    now = time.monotonic()
    limits = get_model_limits(model_name)

    # If any dimension is 0, the model has no free-tier quota at all (in practice some pro
    # models return RPD=0 and others TPM=0; both count). Switching keys does not help, so
    # treat it as having no capacity.
    if any(limits.get(dim) == 0 for dim in ("rpm", "tpm", "rpd")):
        return None, None

    # The built-in RPD is only a guess, and using it to block keys in advance is too risky (a
    # key could be skipped although it still has quota). Block up front only when the number
    # is trustworthy: a value learned from a 429, or one set explicitly in .env.
    rpd_is_authoritative = bool(os.getenv("GEMINI_LIMIT_RPD")) or (
        "rpd" in _LEARNED_LIMITS.get(_normalize_model_name(model_name), {})
    )

    best_index, best_ready = None, None
    any_rpd_left = False
    # Start from the current key so equal candidates stay sticky
    order = list(range(len(_API_KEYS)))
    order = order[_CURRENT_KEY_INDEX:] + order[:_CURRENT_KEY_INDEX]

    for idx in order:
        # A key disabled by 403 is skipped entirely: that is a project-level problem, and a different model would fail the same way.
        if idx in _DENIED_KEYS:
            continue
        owner = _owner_of(_API_KEYS[idx])
        bucket = owner.bucket(model_name)
        bucket.roll_day()
        if bucket.day_blocked:
            continue
        if rpd_is_authoritative and bucket.rpd_left(limits) <= 0:
            continue
        any_rpd_left = True
        ready = bucket.ready_at(now, limits, est_tokens)
        if best_ready is None or ready < best_ready - 0.01:
            best_index, best_ready = idx, ready

    if not any_rpd_left:
        return None, None
    return best_index, max(0.0, best_ready - now)


def _switch_to_key(index):
    global _CURRENT_KEY_INDEX
    if index != _CURRENT_KEY_INDEX:
        _CURRENT_KEY_INDEX = index
        api_key = _API_KEYS[index]
        genai.configure(api_key=api_key.key)
        print(f"   [rotate] Switching API key to {api_key.env_name} (#{index + 1} of {len(_API_KEYS)})")
    else:
        genai.configure(api_key=_API_KEYS[index].key)


def _model_for(index, model_name):
    """Get the GenerativeModel bound to the given key.
    google-generativeai caches the client on the model object, so each key keeps its own model instance."""
    api_key = _API_KEYS[index]
    cache_key = (api_key.fingerprint, model_name)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = genai.GenerativeModel(model_name)
    return _MODEL_CACHE[cache_key]


# ============================================================
# 429 classification: short per-minute limits vs an exhausted daily quota
# ============================================================
def _extract_retry_delay_seconds(exc):
    """Try to parse Google's suggested retry delay (RetryInfo.retry_delay) out of a 429 error."""
    try:
        details = exc.details
    except Exception:
        details = None

    for item in details or []:
        # gRPC transport: details are protobuf objects (RetryInfo.retry_delay is a google.protobuf.Duration)
        if type(item).__name__ == "RetryInfo" and hasattr(item, "retry_delay"):
            rd = item.retry_delay
            return rd.seconds + rd.nanos / 1e9
        # REST transport: details are raw dicts and retryDelay is a string such as "38s"
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            raw = str(item.get("retryDelay", "")).strip()
            if raw.endswith("s"):
                raw = raw[:-1]
            try:
                return float(raw)
            except ValueError:
                continue

    # Last resort: scrape "retry_delay { seconds: 38 }" or "retryDelay: 38s" from the error message
    text = str(exc)
    m = re.search(r"retry[_ ]?delay\s*[{:]\s*(?:seconds\s*:\s*)?(\d+)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


SHORT_RETRY_THRESHOLD_SECONDS = 300


def _429_blob(exc):
    blob = str(exc)
    try:
        blob += " " + repr(exc.details)
    except Exception:
        pass
    return blob


def _classify_429(exc):
    """Returns (kind, retry_seconds). kind is one of {"minute", "day", "unknown"}.

    Note: the order of the checks matters, and it was learned from real traffic.
    Google's free-tier 429s sometimes carry a quota_id of `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
    (which looks like "no quota left today") together with `retry_delay { seconds: 4 }`, and
    really do recover after a few seconds. So whenever Google gives an explicit and short
    retry_delay, treat it as a short-term limit and retry shortly; a PerDay label alone must
    not block the key until Pacific midnight - that was the original cause of false stops.
    """
    retry_seconds = _extract_retry_delay_seconds(exc)
    if retry_seconds is not None and retry_seconds <= SHORT_RETRY_THRESHOLD_SECONDS:
        return "minute", retry_seconds

    flat = _429_blob(exc).lower().replace("_", "").replace(" ", "")
    has_minute = ("perminute" in flat) or ("permin" in flat)
    has_day = "perday" in flat
    if has_minute and not has_day:
        return "minute", retry_seconds
    if has_day:
        # The label says daily, and Google gave no (or a very long) retry delay -> this time the daily quota really is used up
        return "day", retry_seconds
    if retry_seconds is not None:
        return "day", retry_seconds
    return "unknown", None


_QUOTA_DIMENSION_PATTERNS = (
    ("rpd", re.compile(r"requestsperday", re.IGNORECASE)),
    ("rpm", re.compile(r"requestsperminute", re.IGNORECASE)),
    ("tpm", re.compile(r"tokensper\w*minute", re.IGNORECASE)),
)


def _learn_limit_from_429(model_name, exc):
    """Learn the limit Google really applies from a 429 response, overriding the built-in guess.

    The response looks like:
        Quota exceeded for metric: ... limit: 20, model: gemini-3.6-flash
        violations { quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier" quota_value: 20 }
    Returns (dimension, value) or None. A limit of 0 means the model has no free-tier quota at all.
    """
    blob = _429_blob(exc)
    quota_id = ""
    m = re.search(r'quota_id:\s*"?([A-Za-z0-9\-]+)', blob)
    if m:
        quota_id = m.group(1)

    value = None
    m = re.search(r"quota_value:\s*(\d+)", blob) or re.search(r"\blimit:\s*(\d+)", blob)
    if m:
        value = int(m.group(1))
    if value is None:
        return None

    dimension = None
    flat_id = quota_id.replace("_", "")
    for dim, pattern in _QUOTA_DIMENSION_PATTERNS:
        if pattern.search(flat_id):
            dimension = dim
            break
    if dimension is None:
        return None

    model_name = _normalize_model_name(model_name)
    current = _LEARNED_LIMITS.setdefault(model_name, {})
    if current.get(dimension) != value:
        current[dimension] = value
        old = get_model_limits(model_name, include_learned=False).get(dimension)
        print(f"   [learned] Observed the real quota from Google's response: {model_name} {dimension.upper()} = {value}"
              f" (the built-in default was {old})")
    return dimension, value


def _note_429(index, model_name, exc):
    """Record one 429 and set a cooldown on this quota pool according to its type. Returns (kind, cooldown_seconds)."""
    global _QUOTA_SCOPE, _SHARED_WARNED

    api_key = _API_KEYS[index]
    owner = _owner_of(api_key)
    bucket = owner.bucket(model_name)
    bucket.total_429 += 1
    now = time.monotonic()
    bucket.prune(now)

    # First learn the real quota numbers Google applies (scheduling then uses real numbers instead of built-in guesses)
    learned = _learn_limit_from_429(model_name, exc)
    if learned and learned[1] == 0:
        # A limit of 0 means this model has no free-tier quota at all (for example gemini-2.0-* has been closed); do not try again today
        bucket.day_blocked = True
        bucket.cooldown_until = now + _seconds_until_pt_midnight()
        bucket.cooldown_reason = "this model has no free-tier quota (limit=0)"
        print(f"   [blocked] {model_name} has a free-tier {learned[0].upper()} quota of 0 and cannot be used.")
        mark_model_unavailable(model_name, f"free-tier {learned[0].upper()} quota is 0")
        return "day", bucket.cooldown_until - now
    limits = get_model_limits(model_name)

    kind, retry_seconds = _classify_429(exc)

    if kind == "unknown":
        bucket.unknown_429_streak += 1
        # Three undiagnosable 429s in a row on one pool -> conservatively treat the daily quota as used up, to avoid spinning forever
        kind = "day" if bucket.unknown_429_streak >= 3 else "minute"
    else:
        bucket.unknown_429_streak = 0

    # "Waiting will not help" escalation (needed in practice):
    # When the daily quota is used up, Google still attaches a retry_delay of about 57 seconds,
    # and after waiting, the next call returns the same PerDay 429 again. We follow its advice a
    # few times (short-term limits sometimes look like this too), but after 3 consecutive
    # PerDay-labelled 429s we conclude this pool really is empty for today, block it and let
    # the upper layer switch models instead of spinning for another 30 minutes.
    if "perday" in _429_blob(exc).lower().replace("_", ""):
        bucket.day_labeled_429_streak += 1
        if bucket.day_labeled_429_streak >= 3:
            kind = "day"
    else:
        bucket.day_labeled_429_streak = 0

    if kind == "day":
        bucket.day_blocked = True
        cooldown = _seconds_until_pt_midnight()
        bucket.cooldown_until = now + cooldown
        bucket.cooldown_reason = "daily quota (RPD) exhausted"
        print(f"   [blocked] {api_key.env_name} x {model_name}: daily quota (RPD) exhausted "
              f"({bucket.day_requests} requests sent today, limit about {limits['rpd']}).\n"
              f"      {_estimate_next_reset_text()}")
    else:
        cooldown = (retry_seconds if retry_seconds is not None else 60.0) + 2.0
        bucket.cooldown_until = now + cooldown
        bucket.cooldown_reason = f"per-minute rate limit, cooling down {cooldown:.0f}s"
        print(f"   [paused] {api_key.env_name} x {model_name}: hit the per-minute rate limit (RPM/TPM), "
              f"cooling down {cooldown:.0f}s; continuing on another key meanwhile.")

        # Shared-quota hint: when several keys are blocked by the per-minute limit while their
        # usage is still low, they may belong to the same GCP project.
        # This is only a hint and never changes behaviour automatically - the signal is easy to
        # misread (for example the key's quota was consumed by earlier runs, so this run's
        # counters look low). A false positive is expensive (N independent keys would be treated
        # as 1), so confirm with `python ai_client.py --diagnose` and then set
        # GEMINI_QUOTA_SCOPE=shared yourself in .env.
        if _QUOTA_SCOPE == "project" and len(_API_KEYS) > 1 and not _SHARED_WARNED:
            rpm = limits.get("rpm") or 0
            is_minute_quota = "perminute" in _429_blob(exc).lower().replace("_", "")
            if is_minute_quota and rpm and len(bucket.request_times) <= max(1, rpm // 2):
                # Only count suspicious events from the last 90 seconds, so stray 429s do not accumulate into a false alarm
                for fp, seen_at in list(_SUSPECT_SHARED_FINGERPRINTS.items()):
                    if now - seen_at > 90:
                        _SUSPECT_SHARED_FINGERPRINTS.pop(fp, None)
                _SUSPECT_SHARED_FINGERPRINTS[api_key.fingerprint] = now
                if len(_SUSPECT_SHARED_FINGERPRINTS) >= 3:
                    _SHARED_WARNED = True
                    print("\n   [hint] 3 or more keys hit the per-minute limit at the same time while their usage was low.")
                    print("      Gemini quota is counted per [Google Cloud project x model], not per key;")
                    print("      if these keys belong to the same project, spreading load across them does not help.")
                    print("      To confirm, run: python ai_client.py --diagnose")
                    print("      (once confirmed shared, set GEMINI_QUOTA_SCOPE=shared in .env to use the throttling strategy)\n")

    return kind, bucket.cooldown_until - now


def _estimate_next_reset_text(exc=None):
    """Return a human-readable description of the estimated quota recovery time (Pacific time, plus the configured display time zone to avoid confusion)."""
    if exc is not None:
        retry_seconds = _extract_retry_delay_seconds(exc)
        if retry_seconds is not None and retry_seconds <= 300:
            return f"Google suggests waiting about {retry_seconds:.0f}s before retrying (a short per-minute rate limit)."
    now_pt = _pt_now()
    next_reset_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    wait_hours = (next_reset_pt - now_pt).total_seconds() / 3600
    local_str = _pt_to_local_str(next_reset_pt)
    tz_name = (os.getenv("GEMINI_DISPLAY_TZ") or "UTC").strip() or "UTC"
    local_note = f" ({local_str} {tz_name})" if local_str else ""
    return (
        "This looks like the daily Free Tier total (RPD) being used up. The Gemini API Free Tier quota officially resets "
        "at midnight US Pacific time; "
        f"the next reset is estimated at {next_reset_pt.strftime('%Y-%m-%d %H:%M')} Pacific time{local_note} "
        f"(about {wait_hours:.1f} more hours; for reference only, check your Google account quota page for the actual value)."
    )


# ============================================================
# Desktop notification: best-effort alert when the quota is used up
#    (a user reported not being notified at all and missing the window after the reset time had passed)
# ============================================================
def _ps_escape(text):
    return str(text).replace("'", "''")


def _notify_desktop(title, message):
    """Best-effort Windows desktop notification. It is not a phone push (that would need a
    third-party service such as ntfy.sh or a Telegram bot, none of which is configured), but
    it pops up locally at once so you need not keep watching the terminal.
    On non-Windows systems, when PowerShell is missing, or on any failure it is silently
    skipped - a failed notification must never affect the main flow.
    Set GEMINI_DESKTOP_NOTIFY=false in .env to turn it off."""
    if not _env_bool("GEMINI_DESKTOP_NOTIFY", True):
        return
    if os.name != "nt":
        return
    try:
        ps = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "$ni = New-Object System.Windows.Forms.NotifyIcon;"
            "$ni.Icon = [System.Drawing.SystemIcons]::Information;"
            "$ni.Visible = $true;"
            f"$ni.BalloonTipTitle = '{_ps_escape(title)}';"
            f"$ni.BalloonTipText = '{_ps_escape(message)}';"
            "$ni.ShowBalloonTip(10000);"
            "Start-Sleep -Seconds 6;"
            "$ni.Dispose();"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ============================================================
# Recording a call
# ============================================================
def _record_attempt(index, model_name):
    owner = _owner_of(_API_KEYS[index])
    bucket = owner.bucket(model_name)
    now = time.monotonic()
    bucket.prune(now)
    bucket.request_times.append(now)
    bucket.day_requests += 1
    bucket.total_requests += 1
    return bucket


def _extract_usage(response):
    um = getattr(response, "usage_metadata", None)

    def _num(field):
        try:
            return int(getattr(um, field, 0) or 0)
        except Exception:
            return 0

    prompt_tokens = _num("prompt_token_count")
    output_tokens = _num("candidates_token_count")
    thought_tokens = _num("thoughts_token_count")
    total_tokens = _num("total_token_count") or (prompt_tokens + output_tokens + thought_tokens)
    return prompt_tokens, output_tokens, thought_tokens, total_tokens


def _record_success(index, model_name, response, latency, label):
    bucket = _owner_of(_API_KEYS[index]).bucket(model_name)
    p, o, th, total = _extract_usage(response)
    bucket.total_success += 1
    bucket.total_prompt_tokens += p
    bucket.total_output_tokens += o
    bucket.total_thought_tokens += th
    bucket.total_tokens += total
    bucket.day_tokens += total
    bucket.token_events.append((time.monotonic(), total))
    bucket.unknown_429_streak = 0
    bucket.day_labeled_429_streak = 0
    _DENY_STREAK.pop(index, None)   # one success resets the consecutive-403 count
    mark_model_ok(model_name)

    api_key = _API_KEYS[index]
    _append_csv_row([
        datetime.now().isoformat(timespec="seconds"), api_key.env_name, api_key.fingerprint,
        model_name, "ok", p, o, th, total, f"{latency:.2f}", label or "", "",
    ])
    _save_usage_state()
    return total


def _record_failure(index, model_name, status, note, latency, label):
    bucket = _owner_of(_API_KEYS[index]).bucket(model_name)
    if status != "429":
        bucket.total_errors += 1
    api_key = _API_KEYS[index]
    _append_csv_row([
        datetime.now().isoformat(timespec="seconds"), api_key.env_name, api_key.fingerprint,
        model_name, status, 0, 0, 0, 0, f"{latency:.2f}", label or "", str(note)[:300],
    ])
    _save_usage_state()


# ============================================================
# Extra global minimum interval (legacy parameter; default 0 because RPM scheduling supersedes it)
# ============================================================
MIN_REQUEST_INTERVAL_ENV = "GEMINI_MIN_REQUEST_INTERVAL_SECONDS"
DEFAULT_MIN_REQUEST_INTERVAL = 0.0
_last_request_started_at = None


def _throttle_before_request():
    global _last_request_started_at
    min_interval = _env_float(MIN_REQUEST_INTERVAL_ENV, DEFAULT_MIN_REQUEST_INTERVAL)
    now = time.monotonic()
    if min_interval > 0 and _last_request_started_at is not None:
        wait_needed = min_interval - (now - _last_request_started_at)
        if wait_needed > 0:
            time.sleep(wait_needed)
    _last_request_started_at = time.monotonic()


# ============================================================
# Response handling and guard rails
# ============================================================
def get_response_text(response):
    """Safely extract text from an API response; if the safety filter blocked it, raise a clear error instead of leaving the caller to trip over it."""
    try:
        return response.text
    except Exception:
        try:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", "unknown")
        except Exception:
            finish_reason = "unknown"
        raise Exception(
            f"[error] The API response has no usable text (possibly blocked by the safety filter, finish_reason={finish_reason}). "
            "Check whether the prompt or material you sent contains sensitive wording."
        )


def _validate_prompt(prompt):
    """Never let an empty string reach the API: it is the root cause of the 'content argument must not be empty' error."""
    if not prompt or not str(prompt).strip():
        raise ValueError(
            "[error] The content about to be sent to the API is an empty string! "
            "Check that the upstream prompt source (for example a spreadsheet column or a source file) is filled in correctly."
        )


def get_model(model_name=None):
    """Get a GenerativeModel instance bound to the current key.
    Without model_name, GEMINI_MODEL_NAME from .env is used."""
    configure()
    return _model_for(_CURRENT_KEY_INDEX, _normalize_model_name(model_name) or get_default_model_name())


# ============================================================
# Core: the quota-aware request runner
# ============================================================
def _run_with_quota(make_call, model_name, prompt, label,
                    max_retries=6, initial_delay=30, allow_model_fallback=True,
                    fallback_models=None):
    """Wraps the whole "pick a key -> wait -> call -> account -> handle 429 / transient errors" cycle.

    The strategy has two passes (the order was tuned on real traffic and matters):
      Pass 1 "no waiting": sweep the primary model plus every fallback model, and use any
                           "key x model" that can serve the request right now. When the primary
                           model is rate limited this goes to a fallback model first instead of
                           blindly waiting a minute.
      Pass 2 "may wait":   only when everything is cooling down, sleep for the shortest
                           cooldown and continue until it succeeds.
    - Short per-minute limits (RPM/TPM): switch key / switch model / wait and retry, never aborting.
    - Daily quota (RPD) exhausted: that "key x model" is blocked until Pacific midnight and other
      keys or fallback models take over. AllApiKeysExhaustedError is raised only when every
      combination is used up.
    """
    configure()
    est_tokens = _estimate_prompt_tokens(prompt)
    max_wait = _env_float("GEMINI_MAX_WAIT_SECONDS_PER_REQUEST", 1800.0)
    wait_for_daily = _env_bool("GEMINI_WAIT_FOR_DAILY_RESET", False)

    models_to_try = [_normalize_model_name(model_name)]
    if fallback_models is not None:
        # An explicit degradation chain from the caller (for example generate_pro only wants to degrade between pro models)
        models_to_try += [_normalize_model_name(m) for m in fallback_models
                          if _normalize_model_name(m) != models_to_try[0]]
    elif allow_model_fallback:
        models_to_try += get_fallback_model_names(model_name)

    waited_total = 0.0
    transient_attempts = 0
    delay = initial_delay
    exhausted_models = []
    announced_models = set()
    dead_models = set()   # models such as 404s where switching keys is pointless; skipped in both passes
    key_count = max(1, len(_API_KEYS))
    # The maximum number of attempts for one "key x model" combination within one pass. A value
    # of 1 means: try each key once, then move on to the next model, instead of hammering the
    # same model over and over (which is what used to cause stalls).
    sweeps_per_model = max(1, int(_env_float("GEMINI_KEY_SWEEPS_PER_MODEL", 1.0)))
    rate_limited_models = set()
    # A 503 "model overloaded" means the whole model is congested, and switching keys does not
    # help (in a measured case, 10 keys in a row all returned 503). On one, go straight to the
    # next model and push this one to the back of the queue, to be retried after all other
    # models. GEMINI_503_ROUNDS controls how many times a model may be re-queued per request.
    overload_rounds = max(1, int(_env_float("GEMINI_503_ROUNDS", 2.0)))
    overload_counts = {}
    last_overload_error = None

    for allow_wait in (False, True):
        exhausted_models = []
        if allow_wait:
            # Pass 1 may have queued a 503 model several times; pass 2 keeps just one copy
            models_to_try = list(dict.fromkeys(models_to_try))
        for current_model in models_to_try:
            if current_model in dead_models:
                continue
            if allow_wait and overload_counts.get(current_model, 0) > overload_rounds:
                continue  # already came back to it and still got 503; this request will not touch it again
            # How many attempts remain on this model in this pass. When used up, move to the
            # next model; only after every model has been through does the outer loop decide
            # whether to wait another round.
            attempts_left = key_count * sweeps_per_model
            while True:
                index, wait_seconds = _select_key(current_model, est_tokens)

                if index is None:
                    exhausted_models.append(current_model)
                    break  # the daily quota of every key is used up for this model -> next fallback model

                if wait_seconds > 0.5:
                    if not allow_wait:
                        break  # pass 1 does not wait; first see whether another model can serve right now
                    if attempts_left <= 0:
                        # This model has been swept completely and still hit the wall; move on
                        # to the next model instead of repeating "wait -> retry -> wait" on it.
                        rate_limited_models.add(current_model)
                        break
                    if waited_total + wait_seconds > max_wait:
                        raise Exception(
                            f"[error] Cumulative wait for quota recovery exceeded the safety cap of {max_wait:.0f}s; giving up on this request. "
                            f"(adjustable via GEMINI_MAX_WAIT_SECONDS_PER_REQUEST in .env)"
                        )
                    if wait_seconds > 2:
                        print(f"   [wait] All keys for {current_model} are cooling down, "
                              f"resuming automatically after {_fmt_duration(wait_seconds)}...")
                    time.sleep(wait_seconds)
                    waited_total += wait_seconds

                if current_model != models_to_try[0] and current_model not in announced_models:
                    announced_models.add(current_model)
                    limits = get_model_limits(current_model)
                    print(f"   [fallback] {models_to_try[0]} has no quota available right now; using fallback model "
                          f"{current_model} (RPD about {limits['rpd']})")

                _switch_to_key(index)
                model_instance = _model_for(index, current_model)
                _record_attempt(index, current_model)
                _throttle_before_request()

                started = time.monotonic()
                heartbeat = _start_heartbeat(_API_KEYS[index].env_name, current_model)
                try:
                    response = make_call(model_instance)
                except google.api_core.exceptions.ResourceExhausted as e:
                    latency = time.monotonic() - started
                    _record_failure(index, current_model, "429", e, latency, label)
                    _note_429(index, current_model, e)
                    # Move on to the next key that can serve right now and keep trying the same
                    # model, but only for one sweep. When the whole sweep is 429, switch to the
                    # next model - far more effective than sticking to one model, because
                    # RPM/TPM are counted per "project x model".
                    attempts_left -= 1
                    if attempts_left <= 0:
                        rate_limited_models.add(current_model)
                        break
                    continue
                except DENIED_ERRORS as e:
                    # 403: the project of this key cannot use the API. Waiting and switching models do not help; only switching keys does.
                    latency = time.monotonic() - started
                    _record_failure(index, current_model, "denied", e, latency, label)
                    _note_denied(index, e)
                    if not _live_key_indexes():
                        raise Exception(
                            f"[error] All {len(_API_KEYS)} keys received a 403 "
                            f"(Your project has been denied access).\n"
                            f"   This is not a quota problem and no amount of waiting will fix it. Check the projects these keys belong to:\n"
                            f"   is the Generative Language API enabled, is billing attached, "
                            f"does the key have API / origin restrictions."
                        ) from e
                    attempts_left -= 1
                    if attempts_left <= 0:
                        break
                    continue
                except google.api_core.exceptions.NotFound as e:
                    # The model name does not exist, or it has been closed to new users
                    latency = time.monotonic() - started
                    _record_failure(index, current_model, "not_found", e, latency, label)
                    mark_model_unavailable(current_model, "404 model not usable (may be closed to new users)")
                    dead_models.add(current_model)  # a problem of the model itself; switching keys does not help
                    break  # next fallback model
                except google.api_core.exceptions.ServiceUnavailable as e:
                    latency = time.monotonic() - started
                    _record_failure(index, current_model, "transient", e, latency, label)
                    last_overload_error = e
                    count = overload_counts.get(current_model, 0) + 1
                    overload_counts[current_model] = count
                    # This key cools down briefly; coming back to this model later automatically picks another one
                    bucket = _owner_of(_API_KEYS[index]).bucket(current_model)
                    bucket.cooldown_until = max(bucket.cooldown_until, time.monotonic() + initial_delay)
                    if count <= overload_rounds:
                        models_to_try.append(current_model)
                        print(f"   [warn] {current_model} is overloaded (503, took {latency:.0f}s), "
                              f"switching to the next model and coming back to it last. ({count}/{overload_rounds})")
                    else:
                        print(f"   [warn] {current_model} is overloaded (503, took {latency:.0f}s), "
                              f"already retried {overload_rounds} times; this request will not use it again.")
                    break
                except TRANSIENT_ERRORS as e:
                    latency = time.monotonic() - started
                    _record_failure(index, current_model, "transient", e, latency, label)
                    transient_attempts += 1
                    if transient_attempts > max_retries:
                        raise Exception(f"[error] Transient API error still failing after {max_retries} retries ({type(e).__name__}): {e}")
                    # This used to "sleep in place for a fixed time and hit the same key again",
                    # which keeps running into the same wall. Instead, put a short cooldown on
                    # this "key x model" and let _select_key pick the next key that can serve
                    # right now - it only really waits when everything is cooling down.
                    bucket = _owner_of(_API_KEYS[index]).bucket(current_model)
                    bucket.cooldown_until = max(bucket.cooldown_until, time.monotonic() + delay)
                    print(f"   [warn] Transient API error ({type(e).__name__}, took {latency:.0f}s), "
                          f"{_API_KEYS[index].env_name} cooling down {delay}s; continuing on another key. "
                          f"({transient_attempts}/{max_retries})")
                    delay = min(delay + 30, 120)
                    continue
                except Exception as e:
                    latency = time.monotonic() - started
                    _record_failure(index, current_model, "error", e, latency, label)
                    print(f"   [error] API error: {e}")
                    raise
                finally:
                    heartbeat.set()   # always stop the heartbeat thread, whether it succeeded, failed or raised

                latency = time.monotonic() - started
                # Account first, extract text second: even if the safety filter blocks the reply, the request's quota and tokens are already spent
                _record_success(index, current_model, response, latency, label)
                return get_response_text(response)

    # Reaching here means neither pass succeeded. First tell "today's volume is used up" apart
    # from "cannot squeeze in right now" - the two are handled completely differently, and
    # mixing them in one error makes people keep adding keys to no effect.
    if overload_counts and not rate_limited_models:
        raise Exception(
            f"[error] Every available model returned 503 overloaded ({', '.join(overload_counts)}); "
            f"each model was retried in turn {overload_rounds} times. Google's servers are too busy, please run again later. "
            f"(adjustable via GEMINI_503_ROUNDS in .env)"
        ) from last_overload_error

    if rate_limited_models and not wait_for_daily:
        msg = (
            f"[error] Swept all {len(_API_KEYS)} keys x {len(models_to_try)} models "
            f"({', '.join(models_to_try)}) and kept hitting the per-minute rate limit (RPM/TPM).\n"
            f"   Models that kept hitting the wall: {', '.join(sorted(rate_limited_models))}\n"
            f"   This request is estimated at about {est_tokens:,} tokens.\n"
            f"   [warn] If that number is close to or above the model's TPM limit, switching keys, switching models and waiting will all fail -\n"
            f"      because a single request cannot fit into the per-minute token quota. This is common when a chat-style generation has accumulated too much context.\n"
            f"   Suggestions: shorten the content of a single request, split a long conversation or start a new one;\n"
            f"      you can also tune GEMINI_KEY_SWEEPS_PER_MODEL in .env (currently {sweeps_per_model}) to set how many key sweeps each model gets."
        )
        print(msg)
        raise RateLimitedError(msg)

    # The daily quota of every model is used up
    if wait_for_daily:
        wait_seconds = _seconds_until_pt_midnight() + 30
        print(f"   [sleep] The daily quota of all keys x all models is used up. "
              f"Per GEMINI_WAIT_FOR_DAILY_RESET, sleeping {_fmt_duration(wait_seconds)} until the Pacific-midnight reset...")
        time.sleep(wait_seconds)
        for owner in _all_owners():
            for b in owner.buckets.values():
                b.roll_day()
        return _run_with_quota(make_call, model_name, prompt, label,
                               max_retries=max_retries, initial_delay=initial_delay,
                               allow_model_fallback=allow_model_fallback)

    reset_text = _estimate_next_reset_text()
    msg = (
        f"[error] The daily quota of {len(_API_KEYS)} API keys x {len(models_to_try)} models "
        f"({', '.join(models_to_try)}) is used up (429 ResourceExhausted).\n"
        f"   {reset_text}\n"
        f"   To keep going today: add other models that still have quota to GEMINI_FALLBACK_MODELS in .env, "
        f"or set GEMINI_WAIT_FOR_DAILY_RESET=true so the program sleeps until the quota resets and resumes by itself."
    )
    print(msg)
    print_usage_summary()
    _notify_desktop("Gemini free quota used up", reset_text)
    raise AllApiKeysExhaustedError(msg)


# ============================================================
# Usage 1: one-shot request, no memory (suited to outline / copy generation)
# ============================================================
def generate(prompt, model=None, model_name=None, max_retries=6, initial_delay=30,
             timeout=None, raise_on_failure=True, label=None, allow_model_fallback=True,
             fallback_models=None, generation_config=None):
    """
    A single model.generate_content() call with built-in quota scheduling, multi-key load
    distribution, model fallback and an empty-content guard.
    - model: an existing GenerativeModel instance may be passed (it is only used to infer the model name; the real instance is managed by this module)
    - prompt: a string, or a "multimodal content list" (for example [PIL.Image, "text prompt"])
    - generation_config: generation parameters dict, for example
      {"temperature": 0.7, "max_output_tokens": 8192, "response_mime_type": "application/json"}.
      The model defaults are used if omitted. Set response_mime_type="application/json" for strict JSON output.
    - raise_on_failure: after a general failure (all retries used up), True = raise an exception; False = print the error and return an empty string
    - Short per-minute rate limits: waits automatically and rotates keys, never aborting
    - Daily quota exhausted: AllApiKeysExhaustedError is raised only once every key and fallback model is used up (unaffected by raise_on_failure)
    """
    _validate_prompt(prompt)
    configure()
    resolved_model_name = (_normalize_model_name(model_name)
                           or _normalize_model_name(getattr(model, "model_name", None))
                           or get_default_model_name())

    def _call(m):
        return m.generate_content(prompt,
                                  generation_config=generation_config,
                                  request_options=_build_request_options(timeout))

    try:
        return _run_with_quota(_call, resolved_model_name, prompt, label or "generate",
                               max_retries=max_retries, initial_delay=initial_delay,
                               allow_model_fallback=allow_model_fallback,
                               fallback_models=fallback_models)
    except AllApiKeysExhaustedError:
        raise
    except Exception as e:
        if raise_on_failure:
            raise
        print(f"[error] This request failed, returning an empty string: {e}")
        return ""


# ============================================================
# Usage 2: multi-turn conversation with memory (suited to long, coherent writing)
# ============================================================
def start_chat(model=None, model_name=None, history=None):
    """Create a ChatSession. Reuse the same chat object for the same task afterwards to keep the conversation memory coherent."""
    m = model or get_model(model_name)
    return m.start_chat(history=history or [])


def _chat_history_text(chat):
    """Join the conversation history of a ChatSession into plain text, used only to estimate the token count (it does not change what is actually sent).

    The Gemini chat API is stateless: every send_message sends the whole history together with
    the new message. So "how big is this request" = history + new message, and leaving the
    history out of the estimate would underestimate it completely."""
    parts = []
    try:
        for message in getattr(chat, "history", None) or []:
            for part in getattr(message, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
    except Exception:
        return ""   # if the history cannot be read, fall back to estimating just the new message; at least the estimate itself must not blow up
    return "\n".join(parts)


def estimate_chat_tokens(chat):
    """Estimate the token cost of every message sent on this ChatSession right now (= the whole conversation history).

    Lets a caller decide "is it time to start a new conversation". The chat API resends the
    whole history each time, so the longer the history, the more expensive each request is and
    the easier it hits the TPM; beyond a certain length it can no longer be sent at all."""
    return _estimate_prompt_tokens(_chat_history_text(chat))


def get_model_tpm(model_name=None):
    """Return the per-minute token limit (TPM) of the given model; 0 if it cannot be determined."""
    resolved = _normalize_model_name(model_name) or get_default_model_name()
    return get_model_limits(resolved).get("tpm") or 0


def send_chat_message(chat, prompt, max_retries=5, initial_delay=60, timeout=None, label=None,
                      allow_model_fallback=True):
    """
    Send a message on an existing ChatSession, with built-in quota scheduling, multi-key load distribution and an empty-content guard.
    The return value is the text already safely extracted (the caller need not handle response.text).
    - When the key / model switches, the chat is given the matching model instance; the conversation memory (history) is unaffected.
    - Short per-minute rate limits: waits automatically and rotates keys, never aborting.
    - timeout: the longest a single request may wait, in seconds. Always set it, otherwise a stuck
      connection waits forever and the whole program looks hung (a timeout becomes
      DeadlineExceeded, which the transient-error retry of the upper layer picks up).
    """
    _validate_prompt(prompt)
    configure()
    resolved_model_name = (_normalize_model_name(getattr(chat.model, "model_name", None))
                           or get_default_model_name())

    def _call(m):
        chat.model = m  # after a key / model switch, install the matching model instance; history stays on the chat object
        return chat.send_message(prompt, request_options=_build_request_options(timeout))

    # The token estimate must include the whole conversation history: chat-style generation
    # resends the history every time, and counting only the new message would badly
    # underestimate it. The scheduler would then think quota is still available, hit the TPM
    # as soon as the request is sent, and fall into useless rotation ("switch key -> 429 ->
    # switch model -> 429") because the problem is not the key but the request itself being too large.
    estimate_source = _chat_history_text(chat)
    estimate_source = f"{estimate_source}\n{prompt}" if estimate_source else prompt

    est_tokens = _estimate_prompt_tokens(estimate_source)
    tpm_limit = get_model_limits(resolved_model_name).get("tpm") or 0
    if tpm_limit and est_tokens > tpm_limit * 0.8:
        print(f"   [warn] This request together with the conversation history is estimated at about {est_tokens:,} tokens, "
              f"which is {est_tokens / tpm_limit:.0%} of the per-minute limit of {resolved_model_name} "
              f"(TPM {tpm_limit:,}).")
        print("      The conversation is already too long; whichever key or model is used it will most likely be blocked from now on. "
              "Start a new conversation (call start_chat again) and continue there.")

    return _run_with_quota(_call, resolved_model_name, estimate_source, label or "chat",
                           max_retries=max_retries, initial_delay=initial_delay,
                           allow_model_fallback=allow_model_fallback)


# ============================================================
# Usage reports
# ============================================================
def usage_snapshot():
    """Return the current usage statistics as a dict (convenient for writing to a file or analysing yourself)."""
    configure()
    owners = []
    for owner in _all_owners():
        models = []
        for model_name, b in sorted(owner.buckets.items()):
            b.roll_day()
            limits = get_model_limits(model_name)
            models.append({
                "model": model_name,
                "requests_today": b.day_requests,
                "tokens_today": b.day_tokens,
                "rpd_limit": limits["rpd"],
                "rpd_left": b.rpd_left(limits),
                "total_requests": b.total_requests,
                "total_success": b.total_success,
                "total_429": b.total_429,
                "total_errors": b.total_errors,
                "total_prompt_tokens": b.total_prompt_tokens,
                "total_output_tokens": b.total_output_tokens,
                "total_thought_tokens": b.total_thought_tokens,
                "total_tokens": b.total_tokens,
                "day_blocked": b.day_blocked,
            })
        owners.append({"label": owner.label, "fingerprint": owner.fingerprint, "models": models})
    return {
        "pt_date": _pt_today_str(),
        "quota_scope": _QUOTA_SCOPE,
        "keys": len(_API_KEYS),
        "owners": owners,
        "session_seconds": round(time.time() - _SESSION_STARTED_AT, 1),
    }


def _describe_remaining_capacity(model_name):
    """Estimate "how many more requests this model can take today"."""
    limits = get_model_limits(model_name)
    total = 0
    for owner in _all_owners():
        total += owner.bucket(model_name).rpd_left(limits)
    pools = 1 if _QUOTA_SCOPE == "shared" else len(_API_KEYS)
    learned = "rpd" in _LEARNED_LIMITS.get(_normalize_model_name(model_name), {})
    source = "observed value" if learned else "built-in default; the real limit is corrected on the first 429"
    return f"{total} requests ({model_name}, {pools} quota pools x RPD {limits['rpd']}; {source})"


def print_usage_summary():
    """Print the call count, token usage and remaining quota today for every key / model."""
    snap = usage_snapshot()
    print("\n" + "=" * 78)
    print(f" Gemini API usage (Pacific time {snap['pt_date']}; quota mode: {snap['quota_scope']})")
    print("=" * 78)
    header = f"{'key':<20}{'model':<24}{'today':>10}{'left':>10}{'total tokens':>14}"
    print(header)
    print("-" * 78)
    grand_requests = grand_tokens = grand_429 = 0
    for owner in snap["owners"]:
        printed_label = False
        for m in owner["models"]:
            label = owner["label"] if not printed_label else ""
            printed_label = True
            left = "0 (full)" if m["day_blocked"] else str(m["rpd_left"])
            print(f"{label:<20}{m['model']:<24}{m['requests_today']:>10}{left:>10}{m['total_tokens']:>14,}")
            grand_requests += m["requests_today"]
            grand_tokens += m["total_tokens"]
            grand_429 += m["total_429"]
        if not owner["models"]:
            print(f"{owner['label']:<20}{'(not used yet)':<24}")
    print("-" * 78)
    print(f" Requests today: {grand_requests}  |  total tokens: {grand_tokens:,}  |  blocked by 429: {grand_429}")
    csv_path = _usage_csv_path()
    if csv_path:
        print(f" Per-call log: {csv_path}")
    print(f" Stats file: {_usage_json_path()}")
    print("=" * 78 + "\n")


# ============================================================
# Diagnosis: do your keys really share one quota?
# ============================================================
def diagnose_keys(model_name=None, probe_prompt="Answer in one word: ok"):
    """Empirically test whether several keys share the quota of one Google Cloud project.

    Method: first fire requests quickly with key 1 until it is blocked by a 429 (per-minute
    limit); then immediately send exactly ONE request with each of the other keys -
      - if the other key is also 429 right away -> it shares the quota with key 1 (same project) and spreading load across it gains nothing.
      - if the other key answers normally        -> the quotas are independent (different projects) and distributing load across keys really adds capacity.

    This check consumes about (RPM limit + number of keys) requests and only needs to be run once.
    """
    configure()
    model_name = _normalize_model_name(model_name) or get_default_model_name()
    limits = get_model_limits(model_name)
    if len(_API_KEYS) < 2:
        print("   Only one key is configured; no shared-quota check is needed.")
        return

    print("\n" + "=" * 78)
    print(f" Key quota independence check (model {model_name}, expected RPM limit {limits['rpm']})")
    print(f"    This will consume about {limits['rpm'] + len(_API_KEYS)} requests, please wait...")
    print("=" * 78)

    def _probe(index):
        """Send one minimal request directly with the given key; returns "ok" / "429-minute" / "429-day" / an error string."""
        api_key = _API_KEYS[index]
        genai.configure(api_key=api_key.key)
        m = genai.GenerativeModel(model_name)
        started = time.monotonic()
        _record_attempt(index, model_name)
        try:
            resp = m.generate_content(probe_prompt, request_options=_build_request_options(60))
        except google.api_core.exceptions.ResourceExhausted as e:
            _record_failure(index, model_name, "429", e, time.monotonic() - started, "diagnose")
            kind, _ = _classify_429(e)
            return f"429-{kind}"
        except Exception as e:
            _record_failure(index, model_name, "error", e, time.monotonic() - started, "diagnose")
            return f"error: {type(e).__name__}"
        _record_success(index, model_name, resp, time.monotonic() - started, "diagnose")
        return "ok"

    # Phase 1: drive key 1 into a 429
    burst_limit = (limits["rpm"] or 10) + 4
    hit = None
    for i in range(burst_limit):
        result = _probe(0)
        print(f"   [{_API_KEYS[0].env_name}] request {i + 1} -> {result}")
        if result.startswith("429"):
            hit = result
            break

    if hit is None:
        print(f"\n   [warn] {burst_limit} rapid requests were not blocked, so the comparison cannot be completed (the real quota may be higher than the built-in table).")
        print("      Run it again another day when the quota is tighter, or rely on the automatic detection during normal runs.")
        return

    # Phase 2: while key 1 is still blocked, immediately send one request with each other key
    # (works whether Google labels it a per-minute or a daily limit - the point is only "key 1
    # is blocked right now; do the others get blocked with it?")
    print(f"\n   {_API_KEYS[0].env_name} is currently blocked by the quota ({hit}); testing the other keys right away:")
    shared, independent = [], []
    for idx in range(1, len(_API_KEYS)):
        result = _probe(idx)
        name = _API_KEYS[idx].env_name
        print(f"   [{name}] -> {result}")
        if result.startswith("429"):
            shared.append(name)
        elif result == "ok":
            independent.append(name)

    print("\n" + "-" * 78)
    if independent and not shared:
        limits = get_model_limits(model_name)
        total = len(_API_KEYS) * limits["rpd"]
        print(f"   Conclusion: the quotas of the {len(_API_KEYS)} keys are INDEPENDENT (different projects).")
        print(f"      Total capacity today is about {total} requests ({len(_API_KEYS)} x RPD {limits['rpd']}); distributing load across keys really works.")
    elif shared and not independent:
        print("   Conclusion: these keys SHARE one quota (the same Google Cloud project); distributing load across them does not help.")
        print("      Fix: create a key in a different Google Cloud project for each pool,")
        print("      or set GEMINI_QUOTA_SCOPE=shared in .env so the program uses the throttling strategy and avoids pointless spinning.")
    else:
        print(f"   [warn] Mixed result: sharing quota with key 1: {shared}; independent quota: {independent}.")
        print("      Move the shared keys to different projects so the capacity really adds up.")
    print("-" * 78 + "\n")
    print_usage_summary()


def print_model_registry():
    """Only reads the registry file; never calls any API."""
    load_model_registry()
    models = _MODEL_REGISTRY.get("models") or {}
    if not models:
        print("\n   There is no model registry yet. Run `python ai_client.py --modelupdate` to create it "
              "(it only uses list_models() and consumes no generateContent quota).\n")
        return
    print("\n" + "=" * 78)
    print(f" Model registry (updated {_MODEL_REGISTRY.get('updated_at')})")
    print("=" * 78)
    for name in sorted(models, key=_model_sort_key):
        e = models[name]
        if e.get("kind") != "text":
            continue
        icon = _STATUS_ICONS.get(e.get("status"), "[?]")
        tier_label = {"high": "high", "low": "low"}.get(_model_tier(name), "")
        limits = get_model_limits(name)
        print(f"   {icon:<8}{name:<34}{e.get('status',''):<12}{tier_label:<6}RPD~{limits['rpd']:<6}{e.get('status_reason','')}")
    print(f"\n   Automatic fallback order: {' -> '.join(_MODEL_REGISTRY.get('fallback_order') or [])}")
    print(f"   High-tier models (3.6/3.5-flash and equivalents): {', '.join(get_high_tier_models()) or '(none)'}")
    print(f"   Low-tier models (flash-lite family): {', '.join(get_low_tier_models()) or '(none)'}")
    pros = _MODEL_REGISTRY.get("pro_models") or []
    if pros:
        print(f"   Usable pro models (not in automatic fallback; the caller must ask for them): {' , '.join(pros)}")
        print("     Usage: ai_client.generate_pro(prompt)  or  ai_client.get_pro_model_names()")
    else:
        tested = [n for n, e in models.items()
                  if _is_pro_model(n) and e.get("status") == "unavailable"]
        if tested:
            print(f"   Usable pro models: (none) - {len(tested)} pro models were tested and all have a free-tier limit of 0, so they are ignored automatically")
        else:
            print("   Usable pro models: (none) - run `python ai_client.py --verify-only pro` to test them")
    print(f"   Registry file: {_model_registry_path()}")
    print("=" * 78 + "\n")


_CLI_HELP = """
Usage: python ai_client.py [command]

  (no arguments)    show usage statistics + model registry     no API call
  --models          show the model registry only               no API call
  --modelupdate     refresh the registry via list_models()     consumes no generateContent quota
  --help            show this help

  The following consume quota (1 request per model); the cost is listed first and confirmation is requested:
  --verify-only pro         verify all pro models only
  --verify-only flash       verify all flash / flash-lite models only
  --verify-only a,b,c       verify only the named models
  --modelupdate --verify    refresh the registry and verify every model not yet verified
  --yes                     skip the confirmation prompt (use together with any of the above)

  --diagnose        empirically test whether the quotas of several keys are independent    [warn] consumes dozens of requests
"""


def _cli_option_value(argv, flag):
    """Supports both `--verify-only pro` and `--verify-only=pro`."""
    for i, arg in enumerate(argv):
        if arg == flag:
            return argv[i + 1] if i + 1 < len(argv) else None
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    args = set(argv)
    assume_yes = "--yes" in args
    verify_only = _cli_option_value(argv, "--verify-only")

    if "--help" in args or "-h" in args:
        print(_CLI_HELP)
    elif "--models" in args:
        print_model_registry()
    elif verify_only is not None:
        configure()
        verify_models(verify_only, assume_yes=assume_yes)
    elif "--modelupdate" in args:
        configure()
        refresh_model_registry(verify="--verify" in args, assume_yes=assume_yes)
    elif "--diagnose" in args:
        configure()
        diagnose_keys()
    else:
        configure()
        print_usage_summary()
        print_model_registry()
        print(_CLI_HELP)
