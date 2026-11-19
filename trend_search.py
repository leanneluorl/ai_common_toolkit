# -*- coding: utf-8 -*-
"""
trend_search.py - Real-time trend lookup helpers (web grounding, YouTube, Reddit, Google Trends)
===============================================================================================
A small, general-purpose module that sits next to ai_client.py / qwen_client.py. Any script
can `import trend_search` to ask "what is the market / audience talking about right now?"
without knowing anything about the caller's data structures. The functions are independent
of each other, and none of them ever raises an exception into the caller: when a lookup
fails the function prints a warning and returns an empty result.

Usage:
    import trend_search
    digest = trend_search.google_search_digest("reader reviews of <some book>")
    research = trend_search.search_generate("Research the market trends for <topic>...")
    videos = trend_search.youtube_trend_search("spiritual fiction book review")
    trends = trend_search.google_trends_digest(["manifestation novel"])
    posts = trend_search.reddit_trend_search("financial independence 4 percent rule")

================================ 1. Google Search grounding ================================
Uses the Gemini API (the newer `google-genai` SDK) with the `google_search` tool so the
model can ground its answer in live web results.

Quota-aware request distribution:
  - Keys are discovered from GEMINI_API_KEY-1 / -2 / ... (the same naming convention as
    ai_client.py, so the same set of keys is picked up). Each key belongs to its own project
    with its own quota, so requests are spread across the configured keys instead of
    hammering one of them.
  - A local usage ledger (grounding_usage.json) records "calls today per key x model".
    Quota resets at midnight Pacific time, so "today" is computed with _pt_today_str(),
    never the machine's local time zone. When the Pacific date changes the whole file is
    rewritten from zero; no history is kept because the quota itself is reset per day.
  - A key whose per-model daily cap (GROUNDING_DAILY_CAP_PER_KEY) is reached is skipped for
    the rest of the day; when every key is at the cap the API is not called at all.
  - A key that returns 429 (quota) or 404 (model not available to that project) is marked as
    used up for that model for the day, so a round of calls does not waste requests retrying
    something that cannot succeed today. The marker is recomputed with the date.
  - Models form a chain: the primary model first, then GROUNDING_FALLBACK_MODELS. Daily
    limits are counted per key per model, so each fallback model adds an independent budget.
  - Availability of grounding differs per model, per API version and per project, and it
    changes over time, so only models that have been verified are used automatically.
    GROUNDING_PROBE_CANDIDATES lists candidates that a human can test on demand:
        python trend_search.py --probe-models
    Each candidate is tried once, the result is printed and saved to
    grounding_model_probe.json for reference.

search_generate() and google_search_digest() share the same key rotation and the same daily
ledger, so no matter which function is called first, the other sees how much has been used.

================================ 2. YouTube Data API v3 ================================
Reads YOUTUBE_API_KEY from .env. search.list costs 100 quota units per call (the default
daily quota is 10,000), so a local counter (trend_search_youtube_usage.json) caps the number
of calls per day (default 80), leaving headroom for anything else that uses the YouTube API.
Results are cached on disk so a repeated query does not spend quota again.

================================ 3. Google Trends (pytrends) ================================
The pytrends package was archived in April 2025 and may break at any time when Google
changes its backend (429 / captcha), so everything is wrapped in try/except with short
timeouts: a failure just means "no trend data this time" and never propagates.
"""

import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv

_CONFIGURED = False

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(_THIS_DIR, "_log")
os.makedirs(_LOG_DIR, exist_ok=True)


# ============================================================
# Pacific-time helper (Google's free-tier daily quota resets at midnight Pacific time,
# not in the machine's local time zone)
# ============================================================
# Same logic as _pt_today_str() in ai_client.py. Using the local date as "today" can drift
# by many hours from the real reset: the local counter may roll over to zero while Google
# has not reset yesterday's quota yet (so calls that look unused still get a 429), or the
# other way around (quota already reset but the local counter still holds yesterday's count).
def _pt_today_str():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:
        from datetime import timezone, timedelta
        return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")


def configure(env_path=None, force=False):
    """Load .env. Safe to call repeatedly (the real load only happens once unless
    force=True). Shares the same .env as ai_client.py / qwen_client.py; YOUTUBE_API_KEY and
    GEMINI_API_KEY-N are the variables used here."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
    _CONFIGURED = True


# ============================================================
# 1. Google Search grounding (Gemini + google_search tool)
# ============================================================
GROUNDING_MODEL = "gemini-2.5-flash"
# Fallback model chain: when every key has used up its daily budget for the primary model,
# fall back to this one. Daily limits are counted per key per model, so this is an
# independent budget.
GROUNDING_FALLBACK_MODELS = ["gemini-2.5-flash-lite"]
GROUNDING_DAILY_CAP_PER_KEY = 18      # a little below the observed free-tier limit, as a safety margin
GROUNDING_TIMEOUT_SECONDS = 60

# Candidate grounding models. This list is only for the manual probe_grounding_models() run;
# regular calls (google_search_digest / search_generate) never try these automatically, so
# an unverified model cannot burn budget that production traffic needs.
GROUNDING_PROBE_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]

_GROUNDING_USAGE_PATH = os.path.join(_LOG_DIR, "grounding_usage.json")
_GROUNDING_PROBE_PATH = os.path.join(_LOG_DIR, "grounding_model_probe.json")

_NUMBERED_API_KEY_PATTERN = re.compile(r"^GEMINI_API_KEY-(\d+)$")
_LEGACY_API_KEY_ENV_NAMES = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]


def _discover_grounding_keys():
    """Scan GEMINI_API_KEY-1 / -2 / ... (falling back to a single legacy GEMINI_API_KEY /
    GOOGLE_API_KEY if there are none). Same naming convention as ai_client.py, so the same
    keys are picked up. Returns [(env_name, key_value), ...] sorted by number."""
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


def _load_grounding_usage():
    """Read today's usage ledger; when the date changes (or the structure does not match)
    the whole file is rewritten from zero and no history is kept.
    Structure: {"date": "YYYY-MM-DD", "per_key": {env_name: {model_name: count, ...}, ...}}"""
    today = _pt_today_str()
    try:
        with open(_GROUNDING_USAGE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}

    per_key = data.get("per_key")
    is_valid_shape = isinstance(per_key, dict) and all(isinstance(v, dict) for v in per_key.values())
    if data.get("date") != today or not is_valid_shape:
        data = {"date": today, "per_key": {}}
    data.setdefault("per_key", {})
    return data


def _save_grounding_usage(data):
    try:
        with open(_GROUNDING_USAGE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"      [warn] [trend_search] could not write the grounding usage ledger (the lookup itself is unaffected): {e}")


def _bump_grounding_usage(env_name, model_name):
    data = _load_grounding_usage()
    bucket = data["per_key"].setdefault(env_name, {})
    bucket[model_name] = bucket.get(model_name, 0) + 1
    _save_grounding_usage(data)


def _mark_grounding_key_exhausted(env_name, model_name, cap):
    data = _load_grounding_usage()
    bucket = data["per_key"].setdefault(env_name, {})
    bucket[model_name] = cap
    _save_grounding_usage(data)


def _grounding_usage_for(env_name, model_name):
    data = _load_grounding_usage()
    return data.get("per_key", {}).get(env_name, {}).get(model_name, 0)


def _available_grounding_keys(model_name, cap):
    """Keys whose usage of this model today is still below the cap, in numeric order."""
    configure()
    return [
        (name, value) for name, value in _discover_grounding_keys()
        if _grounding_usage_for(name, model_name) < cap
    ]


def _extract_grounding(response, num_sources=5):
    """Dig the evidence that "the model really issued a search" out of the response;
    if nothing can be found, treat it as not grounded."""
    queries, sources = [], []
    try:
        candidates = response.candidates or []
        gm = candidates[0].grounding_metadata if candidates else None
        if gm:
            queries = list(gm.web_search_queries or [])
            for chunk in (gm.grounding_chunks or [])[:num_sources]:
                web = getattr(chunk, "web", None)
                if web and (web.uri or web.title):
                    sources.append({"title": web.title or "", "url": web.uri or ""})
    except Exception:
        pass   # no evidence found = not grounded; a failure here must not interrupt the main flow
    return queries, sources


def _resolve_redirect_url(url, timeout=8):
    """Grounding sources are often redirect links on vertexaisearch.cloud.google.com;
    resolve them to the real URL. On failure keep the original link."""
    try:
        import requests
        r = requests.get(url, allow_redirects=True, timeout=timeout, stream=True)
        r.close()
        return r.url
    except Exception:
        return url


def _grounded_generate(prompt, model_name, cap, label=None, timeout=None):
    """Core call: try every key that still has budget for this model today, in order, until
    one succeeds or all have been tried.

    This is the single entry point shared by google_search_digest(), search_generate() and
    probe_grounding_models(), and all of them record usage in the same grounding_usage.json,
    so the budget is accounted for in one place no matter which function was called.

    Returns (response, env_name, error). On success error is "". When everything fails,
    response and env_name are None. It never raises: failure is reflected in the return
    value and the caller decides whether to degrade gracefully.
    """
    configure()
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return None, None, f"failed to load the google-genai package, skipping web search: {e}"

    candidates = _available_grounding_keys(model_name, cap)
    if not candidates:
        return None, None, (
            f"every Gemini key has used up today's grounding budget for {model_name} (cap {cap} per key)"
        )

    tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(
        tools=[tool],
        http_options=types.HttpOptions(timeout=int((timeout or GROUNDING_TIMEOUT_SECONDS) * 1000)),
    )
    tag = f"[{label}] " if label else ""
    last_error = None

    for env_name, key_value in candidates:
        try:
            client = genai.Client(api_key=key_value)
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            _bump_grounding_usage(env_name, model_name)
            return response, env_name, ""
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            if "429" in msg or "quota" in msg or "resource_exhausted" in msg or "resourceexhausted" in msg:
                print(f"      [warn] [trend_search] {tag}{env_name} has used up its quota for {model_name}; trying the next key...")
                _mark_grounding_key_exhausted(env_name, model_name, cap)
            elif "404" in msg or "not_found" in msg:
                # A 404 means one of two account / model level conditions: the model is not
                # offered to this project (typically newer projects lose access to older
                # models), or the model has been retired for everyone. Neither is a quota
                # problem, but retrying today will not help either, so it is marked as
                # "used up today" to avoid re-trying keys that are bound to fail on every call.
                # The marker is recomputed with the date, so tomorrow it is re-evaluated.
                print(f"      [warn] [trend_search] {tag}{env_name} does not support {model_name} (model missing or no access for this project; not a quota issue); trying the next key...")
                _mark_grounding_key_exhausted(env_name, model_name, cap)
            else:
                print(f"      [warn] [trend_search] {tag}{env_name} lookup failed; trying the next key: {e}")
            continue

    error = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
    return None, None, error


def _grounded_generate_chain(prompt, models, cap, label=None, timeout=None):
    """Try every model in the list in order (inside each model every key that still has
    budget today is tried first); move on to the next model only when all keys for the
    previous one are used up / unsupported. Returns (response, env_name, model_name, error)."""
    last_error = None
    for model_name in models:
        response, env_name, error = _grounded_generate(prompt, model_name, cap, label=label, timeout=timeout)
        if response is not None:
            return response, env_name, model_name, ""
        last_error = error
    return None, None, (models[-1] if models else None), (last_error or "unknown error")


def google_search_digest(query, num_sources=5, preferred_domains=()):
    """Run one live web lookup (Gemini google_search grounding). Returns:
    {"text": the model's digest, "sources": [{"title": ..., "url": ...}], "model": model used}
    It never raises; when nothing is found / the budget is used up it returns
    {"text": "", "sources": [], "model": ""}.

    preferred_domains: optional tuple of domain fragments (e.g. ("amazon.", "goodreads."));
    sources from those domains are sorted first. Other sources are kept, because Google
    does not support hard domain restrictions."""
    models = [GROUNDING_MODEL] + GROUNDING_FALLBACK_MODELS
    response, _env_name, model_used, error = _grounded_generate_chain(query, models, GROUNDING_DAILY_CAP_PER_KEY)
    if response is None:
        print(f"      [warn] [trend_search] skipping this web lookup: \"{query}\" ({error})")
        return {"text": "", "sources": [], "model": ""}

    text = (response.text or "").strip() if hasattr(response, "text") else ""
    _, sources = _extract_grounding(response, num_sources)
    for s in sources:
        s["url"] = _resolve_redirect_url(s["url"])
    if preferred_domains:
        sources.sort(key=lambda s: 0 if any(d in urlparse(s["url"]).netloc.lower()
                                             for d in preferred_domains) else 1)
    return {"text": text, "sources": sources, "model": model_used}


def search_generate(prompt, model_name=None, label=None, timeout=None):
    """Grounded generation with a richer result than google_search_digest(); it shares the
    same key rotation and the same daily usage ledger.

    The returned dict always has these keys:
        text     - the model's text answer (including its digest of the search results);
                   an empty string on failure
        grounded - True = there is evidence that the model really issued a search request;
                   False = there is not (the model may have decided it did not need to
                   search, or the request did not trigger one; the two cannot be told
                   apart from the outside)
        queries  - the search queries the model actually issued (only filled when grounded,
                   useful for human review)
        sources  - [{"title": ..., "url": ...}, ...], the sources cited by the search results
        error    - the error message when every key failed; an empty string on success

    This function does not raise: failure is reflected in the error / grounded fields, and
    the caller decides how to degrade (for example fall back to the model's training
    knowledge) so that a failed search step does not abort a whole pipeline.
    """
    if not str(prompt or "").strip():
        raise ValueError("prompt is empty")

    # Without an explicit model, use the same primary -> fallback chain as
    # google_search_digest(); an explicit model_name is respected and only that one model
    # is tried (no automatic switch to the fallback).
    models = [model_name] if model_name else [GROUNDING_MODEL] + GROUNDING_FALLBACK_MODELS
    response, _env_name, _model_used, error = _grounded_generate_chain(
        prompt, models, GROUNDING_DAILY_CAP_PER_KEY, label=label, timeout=timeout,
    )
    if response is None:
        return {"text": "", "grounded": False, "queries": [], "sources": [], "error": error}

    queries, sources = _extract_grounding(response)
    return {
        "text": (response.text or "").strip(),
        "grounded": bool(queries),
        "queries": queries,
        "sources": sources,
        "error": "",
    }


def probe_grounding_models(models=None, query=None):
    """Manually test a round of candidate models to see which ones can really use the
    google_search tool to fetch live data.

    Each model borrows one key that still has budget for it today and is tested once. The
    quota used is recorded in grounding_usage.json like any other call (nothing is hidden),
    so run this when budget is plentiful (for example just after the Pacific-time midnight
    reset). Results are printed and also saved to grounding_model_probe.json for reference.

    Usage: python trend_search.py --probe-models
    Returns {model_name: {"ok": bool, "grounded": bool, "note": str}}.
    """
    models = models or GROUNDING_PROBE_CANDIDATES
    query = query or "Use Google search to look up the current USD to EUR exchange rate. Please cite your sources."
    results = {}
    print(f"   [probe] testing {len(models)} candidate grounding model(s)...")
    for model_name in models:
        response, env_name, error = _grounded_generate(
            query, model_name, GROUNDING_DAILY_CAP_PER_KEY, label="probe",
        )
        if response is None:
            results[model_name] = {"ok": False, "grounded": False, "note": error}
            print(f"   [x] {model_name}: {error}")
            continue
        queries, _sources = _extract_grounding(response)
        grounded = bool(queries)
        results[model_name] = {
            "ok": True, "grounded": grounded,
            "note": f"key={env_name}" + ("" if grounded else " (succeeded but no search was triggered; the model may have judged it unnecessary)"),
        }
        print(f"   {'[ok]' if grounded else '[~]'} {model_name}: grounded={grounded} ({env_name})")
        time.sleep(1)  # avoid hitting the per-minute limit with a burst of calls

    try:
        with open(_GROUNDING_PROBE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"tested_at": datetime.now().isoformat(timespec="seconds"), "results": results},
                f, ensure_ascii=False, indent=2,
            )
        print(f"   [saved] results written to {_GROUNDING_PROBE_PATH}")
    except Exception as e:
        print(f"   [warn] could not write {_GROUNDING_PROBE_PATH}: {e}")
    return results


# ============================================================
# 2. YouTube Data API v3
# ============================================================
YOUTUBE_DAILY_SEARCH_CAP = 80
_YOUTUBE_USAGE_PATH = os.path.join(_LOG_DIR, "trend_search_youtube_usage.json")


def _load_youtube_usage():
    today = _pt_today_str()
    try:
        with open(_YOUTUBE_USAGE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    if data.get("date") != today:
        data = {"date": today, "search_calls": 0}
    return data


def _bump_youtube_usage():
    data = _load_youtube_usage()
    data["search_calls"] = data.get("search_calls", 0) + 1
    try:
        with open(_YOUTUBE_USAGE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"      [warn] [trend_search] could not write the YouTube usage ledger (the lookup itself is unaffected): {e}")


_YOUTUBE_CACHE_PATH = os.path.join(_LOG_DIR, "trend_search_youtube_cache.json")
YOUTUBE_CACHE_TTL_DAYS = 7        # queries with results: reuse the cache for 7 days, no quota spent
YOUTUBE_CACHE_EMPTY_TTL_DAYS = 2  # queries that really returned 0 items: only remember for 2 days


def _yt_cache_key(query, max_results, relevance_language, region_code, published_after_days):
    return "|".join(str(x) for x in (query.strip().lower(), max_results, relevance_language or "",
                                      region_code or "", published_after_days or 0))


def _yt_cache_load():
    try:
        with open(_YOUTUBE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _yt_cache_get(key):
    """Returns a list (hit; may be an empty list) or None (missing / expired)."""
    e = _yt_cache_load().get(key)
    if not e:
        return None
    ttl = YOUTUBE_CACHE_TTL_DAYS if e.get("videos") else YOUTUBE_CACHE_EMPTY_TTL_DAYS
    if time.time() - e.get("ts", 0) > ttl * 86400:
        return None
    return e.get("videos") or []


def _yt_cache_put(key, videos):
    data = _yt_cache_load()
    now = time.time()
    data = {k: v for k, v in data.items() if now - v.get("ts", 0) < 30 * 86400}  # also drop entries older than 30 days
    data[key] = {"ts": now, "videos": videos}
    try:
        with open(_YOUTUBE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"      [warn] [trend_search] could not write the YouTube cache (the lookup itself is unaffected): {e}")


def youtube_budget_left():
    """How many more search.list calls can be made today; 0 once Google has reported
    that the quota is exhausted."""
    u = _load_youtube_usage()
    if u.get("quota_exhausted"):
        return 0
    return max(0, YOUTUBE_DAILY_SEARCH_CAP - u.get("search_calls", 0))


def youtube_usage_line():
    u = _load_youtube_usage()
    s = f"used {u.get('search_calls', 0)}/{YOUTUBE_DAILY_SEARCH_CAP} search.list calls today (Pacific time)"
    return s + "; Google has reported the quota as exhausted" if u.get("quota_exhausted") else s


def _mark_youtube_quota_exhausted():
    data = _load_youtube_usage()
    data["quota_exhausted"] = True
    try:
        with open(_YOUTUBE_USAGE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def youtube_trend_search(query, max_results=5, relevance_language=None, region_code=None,
                         published_after_days=None):
    """Simple interface: returns only the list of videos. Callers that need to tell "really
    0 results" apart from "quota exhausted / error, so unknown" should use
    youtube_trend_search_ex()."""
    return youtube_trend_search_ex(query, max_results, relevance_language, region_code,
                                   published_after_days)["videos"]


def youtube_trend_search_ex(query, max_results=5, relevance_language=None, region_code=None,
                            published_after_days=None):
    """Returns {"videos": [...], "status": ...}. status is one of:
      "cached" = cache hit (no quota spent), "ok" = live lookup with results,
      "empty"  = live lookup that really returned 0 items,
      "quota"  = today's quota (the local cap or Google's report) is used up,
      "error"  = the API failed, "nokey" = no API key configured.
    quota / error / nokey mean "unknown whether anyone cares", which is not the same as 0
    results, so callers should not discard a topic because of them.
    Results are cached for 7 days (the same query does not spend another 100 units), and
    once Google reports quotaExceeded the rest of the day is skipped instead of calling the
    API for every query."""
    key = _yt_cache_key(query, max_results, relevance_language, region_code, published_after_days)
    hit = _yt_cache_get(key)
    if hit is not None:
        return {"videos": hit, "status": "cached" if hit else "empty"}
    r = _youtube_live_search(query, max_results, relevance_language, region_code, published_after_days)
    if r["status"] in ("ok", "empty"):
        _yt_cache_put(key, r["videos"])
    return r


def _youtube_live_search(query, max_results=5, relevance_language=None, region_code=None,
                         published_after_days=None):
    """Find the most-viewed YouTube videos on a topic. Returns {"videos": list[{title, url,
    view_count, like_count, comment_count}], "status": ...}. Reads YOUTUBE_API_KEY from .env;
    a missing key, an exceeded daily cap or an API error all return an empty list with the
    matching status and never raise.

    published_after_days: only consider videos published in the last N days. With
    order=viewCount and no time range the API returns the all-time most-viewed videos (often
    years old), which is not "what is trending now"; with this it is "most watched recently".
    Default None = no limit.

    relevance_language / region_code: leave empty for no language / region restriction. For
    multi-language channels pass e.g. relevance_language="en", region_code="US" so that
    results in one language do not leak into the other."""
    configure()
    api_key = (os.getenv("YOUTUBE_API_KEY") or "").strip()
    if not api_key:
        print("      [warn] [trend_search] YOUTUBE_API_KEY is not set; skipping the YouTube trend lookup.")
        return {"videos": [], "status": "nokey"}

    usage = _load_youtube_usage()
    if usage.get("quota_exhausted") or usage.get("search_calls", 0) >= YOUTUBE_DAILY_SEARCH_CAP:
        print(f"      [warn] [trend_search] today's YouTube quota is used up ({youtube_usage_line()}); "
              f"skipping this lookup (result unknown, not the same as 0 items): \"{query}\"")
        return {"videos": [], "status": "quota"}

    try:
        from googleapiclient.discovery import build
        youtube = build("youtube", "v3", developerKey=api_key)

        search_kwargs = dict(q=query, part="id", type="video", order="viewCount", maxResults=max_results)
        if relevance_language:
            search_kwargs["relevanceLanguage"] = relevance_language
        if region_code:
            search_kwargs["regionCode"] = region_code
        if published_after_days:
            from datetime import timedelta, timezone
            cutoff = datetime.now(timezone.utc) - timedelta(days=published_after_days)
            search_kwargs["publishedAfter"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        search_resp = youtube.search().list(**search_kwargs).execute()
        _bump_youtube_usage()

        video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])
                     if item.get("id", {}).get("videoId")]
        if not video_ids:
            return {"videos": [], "status": "empty"}

        stats_resp = youtube.videos().list(
            part="snippet,statistics", id=",".join(video_ids),
        ).execute()

        results = []
        for v in stats_resp.get("items", []):
            stats = v.get("statistics", {})
            results.append({
                "title": v.get("snippet", {}).get("title", ""),
                "url": f"https://www.youtube.com/watch?v={v['id']}",
                "view_count": int(stats.get("viewCount", 0) or 0),
                "like_count": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
            })
        results.sort(key=lambda r: r["view_count"], reverse=True)
        return {"videos": results, "status": "ok" if results else "empty"}
    except Exception as e:
        if "quotaExceeded" in str(e) or "quota" in str(e).lower():
            _mark_youtube_quota_exhausted()
            print(f"      [warn] [trend_search] Google reported the YouTube quota as exhausted; no more lookups today (result unknown): \"{query}\"")
            return {"videos": [], "status": "quota"}
        print(f"      [warn] [trend_search] YouTube lookup failed; skipping: {e}")
        return {"videos": [], "status": "error"}


# ============================================================
# Reddit (PRAW, read-only)
# ============================================================
# Requires REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET in .env (create a "script" type app at
# reddit.com/prefs/apps). REDDIT_USER_AGENT may be left empty and the default below is used.
# If nothing is configured the function simply returns [] without affecting the other
# sources, following the same "never raise" rule as the rest of this module.
_DEFAULT_REDDIT_USER_AGENT = "trend-search/0.1 (trend_search.py)"


def reddit_trend_search(query, subreddits=None, limit=5, sort="relevance", time_filter="month"):
    """Find Reddit posts related to query that have a lot of discussion. Returns
    list[{title, url, permalink, score, num_comments, subreddit, selftext}] sorted by score.

    subreddits: list of subreddit names to limit the search (e.g.
                ["personalfinance", "financialindependence"]); empty = search all of Reddit.
                sort / time_filter map directly to the parameters of PRAW's search().
    Returns [] (never raises) when REDDIT_CLIENT_ID/SECRET are missing, the package is not
    installed, or the lookup fails."""
    configure()
    client_id = (os.getenv("REDDIT_CLIENT_ID") or "").strip()
    client_secret = (os.getenv("REDDIT_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        print("      [warn] [trend_search] REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET are not set; skipping the Reddit trend lookup "
              "(create a script-type app at https://www.reddit.com/prefs/apps to get them).")
        return []

    try:
        import praw
    except ImportError:
        print("      [warn] [trend_search] praw is not installed (pip install praw); skipping the Reddit trend lookup.")
        return []

    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=(os.getenv("REDDIT_USER_AGENT") or "").strip() or _DEFAULT_REDDIT_USER_AGENT,
        )
        reddit.read_only = True

        if subreddits:
            scope = reddit.subreddit("+".join(subreddits))
        else:
            scope = reddit.subreddit("all")

        results = []
        for submission in scope.search(query, sort=sort, time_filter=time_filter, limit=limit):
            results.append({
                "title": submission.title or "",
                "url": f"https://www.reddit.com{submission.permalink}",
                "permalink": submission.permalink or "",
                "score": int(submission.score or 0),
                "num_comments": int(submission.num_comments or 0),
                "subreddit": str(submission.subreddit),
                "selftext": (submission.selftext or "")[:500],
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results
    except Exception as e:
        print(f"      [warn] [trend_search] Reddit lookup failed; skipping: {e}")
        return []


# ============================================================
# 3. Google Trends (pytrends, best effort, unmaintained package)
# ============================================================
_GTRENDS_CACHE_PATH = os.path.join(_LOG_DIR, "trend_search_gtrends_cache.json")
_GTRENDS_BATCH = 5  # pytrends accepts at most 5 keywords per request (anchor included)


def _gtrends_cache_load():
    try:
        with open(_GTRENDS_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    return data if data.get("date") == _pt_today_str() else {"date": _pt_today_str(), "entries": {}}


def _gtrends_series_stats(series):
    """One weekly search-interest series -> (status, momentum, mean).
    status uses the classic rule (average of the latest quarter vs the earliest quarter,
    +/-20%). On top of that a "last 4 weeks vs the earlier average" rule is applied:
    evergreen topics stay flat all year, so the classic rule labels almost everything
    stable and the ranking is useless; when the last 4 weeks jump (>= 1.3x) or drop
    (<= 0.7x) that is reflected in status too. momentum = last-4-week average / earlier
    average (None when there is no data)."""
    n = len(series)
    mean = float(series.mean()) if n else 0.0
    # Very low volume (pytrends values are relative, 0-100; a mean below 2 means almost all
    # values are 0 or 1): ratios explode on a tiny base (e.g. 0.2 -> 1.8 is "9x"), which is
    # not real warming up, so treat it as insufficient data = stable.
    if n < 8 or mean < 2:
        return "stable", None, mean
    quarter = n // 4
    recent_q, earlier_q = series.iloc[-quarter:].mean(), series.iloc[:quarter].mean()
    if earlier_q == 0:
        status = "rising" if recent_q > 0 else "stable"
    elif recent_q > earlier_q * 1.2:
        status = "rising"
    elif recent_q < earlier_q * 0.8:
        status = "declining"
    else:
        status = "stable"
    prior = series.iloc[:-4].mean()
    momentum = float(series.iloc[-4:].mean() / prior) if prior > 0 else None
    if status == "stable" and momentum is not None:
        if momentum >= 1.3:
            status = "rising"
        elif momentum <= 0.7:
            status = "declining"
    return status, momentum, mean


def _gtrends_fetch_batch(batch, timeframe, hl, geo):
    """Query one batch (<= 5) of keywords; a transient failure (429) is retried once after
    a short wait. Returns {kw: (status, momentum, mean)}, or {} on failure."""
    from pytrends.request import TrendReq
    for attempt in (1, 2):
        try:
            pytrends = TrendReq(hl=hl, tz=480, timeout=(5, 10))
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
            if df is None or df.empty:
                return {}
            return {kw: _gtrends_series_stats(df[kw]) for kw in batch if kw in df.columns}
        except Exception as e:
            if attempt == 2:
                print(f"      [warn] [trend_search] Google Trends lookup failed for batch {batch} (pytrends is no longer "
                      f"maintained, so occasional failures are expected); skipping: {e}")
                return {}
            time.sleep(8)
    return {}


def google_trends_scores(keywords, timeframe="today 12-m", hl="en-US", geo=""):
    """Full version of google_trends_digest: returns {keyword: {"status": rising/stable/declining,
    "level": 0-1 (relative search volume within this set of keywords; the hottest is 1),
    "momentum": last 4 weeks / earlier average, or None}}.

    - More than 5 keywords are queried in batches, and every batch carries the same anchor
      keyword: pytrends values are relative to the maximum of each batch (= 100), so values
      from different batches cannot be compared directly. The anchor's mean in each batch is
      used as a ruler to convert everything back to the units of the first batch, which puts
      all keywords on one scale (level).
    - Lookups for the same (keyword list, hl, geo) on the same day come from a cache (each
      pytrends call takes 8-16 seconds and repeated calls in a short time easily hit a 429).
    Any failure returns {} (or only the keywords that were found) and never raises."""
    keywords = list(dict.fromkeys(k.strip() for k in (keywords or []) if k and k.strip()))
    if not keywords:
        return {}

    cache = _gtrends_cache_load()
    cache_key = "|".join(["v2", hl, geo, timeframe] + keywords)
    if cache_key in cache["entries"]:
        return cache["entries"][cache_key]

    try:
        # Query the first batch (first 5 keywords) first. The anchor is the keyword with the
        # highest mean search volume in that batch - it cannot simply be the first keyword,
        # because that one may have a volume of 0 and then it cannot serve as a ruler.
        # Every later batch carries the anchor and converts back to first-batch units by the
        # ratio of the anchor's mean in the two batches.
        raw = {}   # kw -> (status, momentum, mean_in_first_batch_units)
        first = _gtrends_fetch_batch(keywords[:_GTRENDS_BATCH], timeframe, hl, geo)
        for kw, (status, momentum, mean) in first.items():
            raw[kw] = (status, momentum, mean)
        rest = keywords[_GTRENDS_BATCH:]
        anchor = max(first, key=lambda k: first[k][2]) if first else None
        anchor_ref = first[anchor][2] if anchor else 0.0
        step = _GTRENDS_BATCH - 1
        for i in range(0, len(rest), step):
            batch = rest[i:i + step]
            if not anchor_ref:   # the first batch was all zeros or failed: no ruler, so later batches only get a status
                batch_kws = batch
            else:
                batch_kws = [anchor] + batch
            time.sleep(2)  # a short pause between batches lowers the chance of a 429
            stats = _gtrends_fetch_batch(batch_kws, timeframe, hl, geo)
            a_mean = stats.get(anchor, (None, None, 0.0))[2] if anchor_ref else 0.0
            scale = (anchor_ref / a_mean) if (a_mean and anchor_ref) else None
            for kw, (status, momentum, mean) in stats.items():
                if kw in raw:
                    continue
                raw[kw] = (status, momentum, mean * scale if scale else None)

        peak = max((m for _s, _mo, m in raw.values() if m), default=0.0)
        result = {}
        for kw, (status, momentum, mean) in raw.items():
            level = round(mean / peak, 3) if (peak and mean is not None) else None
            result[kw] = {"status": status, "level": level,
                          "momentum": round(momentum, 2) if momentum is not None else None}
        if result:
            cache["entries"][cache_key] = result
            try:
                with open(_GTRENDS_CACHE_PATH, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"      [warn] [trend_search] could not write the Google Trends cache (the lookup itself is unaffected): {e}")
        return result
    except Exception as e:
        print(f"      [warn] [trend_search] Google Trends lookup failed (pytrends is no longer maintained, so occasional failures are expected); skipping: {e}")
        return {}


def google_trends_digest(keywords, timeframe="today 12-m", hl="en-US", geo=""):
    """Look up how search interest in keywords has moved over a period. Returns
    {keyword: "rising"/"stable"/"declining"}. pytrends is unmaintained and may break at any
    time when Google changes something (429 / captcha); any failure returns {} and never
    raises. For relative popularity (level) and recent momentum use google_trends_scores();
    this is the simple interface that only returns status.

    hl / geo: interface language and region of the query. Pass e.g. hl="en-US", geo="US" for
    English-language topics and hl="zh-TW", geo="TW" for Chinese-language ones, so that each
    audience is looked up in its own language and region instead of one fixed setting
    (using the same region and language for every query distorts the numbers)."""
    scores = google_trends_scores(keywords, timeframe=timeframe, hl=hl, geo=geo)
    return {kw: v["status"] for kw, v in scores.items()}


if __name__ == "__main__":
    import sys

    if "--probe-models" in sys.argv:
        configure()
        probe_grounding_models()
    else:
        print(__doc__)
        print("Available command: python trend_search.py --probe-models")
