# -*- coding: utf-8 -*-
"""
aliyun_vision_client.py - Alibaba Cloud Model Studio (DashScope) image / video generation client
==================================================================================================
Calls DashScope's native task APIs. Depending on the model, image generation is either a
"create task -> poll for result" flow or a single synchronous call, and video generation is
always "create task -> poll". Both differ from the OpenAI-compatible /chat/completions mode, so
this module cannot share a base_url with qwen_client.py.

This module does two things only - generate images and generate videos:
    aliyun_vision_client.generate_image("A cat playing piano in the rain")
    aliyun_vision_client.generate_video("A cat playing piano in the rain, slow dolly-in")

The interface deliberately mirrors agnes_vision_client.py (same directory): both modules expose
generate_image() / generate_video() with the same parameter shape, and both return an image /
video URL string. Swapping one provider for the other should therefore only require changing
the import (whether it is a fully painless swap still depends on the caller using only the
parameter subset that both providers support).

Model catalog, tiers and quota allowances live in aliyun_vision_models.json in the same
directory. The catalog is independent of Agnes' model list (which is hard-coded in
agnes_vision_client.py); the two are never mixed.

[note] Tiers are derived from naming conventions and model generations; the vendor publishes no
       official tiering (the same caveat applies to MODEL_TIERS in qwen_client.py). Review
       them before relying on them.

[note] Only models in the SAME tier may act as fallbacks for each other:
       get_fallback_model_names() only searches the tier that the primary model belongs to and
       never falls back across tiers (a flagship failure must not silently degrade to a
       low-tier model and produce a quality drop nobody notices). Caller-supplied
       fallback_models are validated the same way: a model from another tier raises
       AliyunCrossTierFallbackError before any request is sent.

[note] Image models are served by three completely different APIs, not by one single
       "text2image / video-generation async task" API. The error "url error, please check
       url!" means "model / endpoint mismatch" (for example calling a multimodal endpoint with
       a text-only model, or the reverse); it is NOT a data URI formatting problem - all three
       APIs accept data URIs. The real problem is the wrong endpoint / payload shape:

   1) Text-to-image (legacy): POST services/aigc/text2image/image-synthesis,
      input = {"prompt": ..., "images": [...]} (the shape built by _build_image_input()),
      task_id / polling mode. Used by the Wan "text-to-image" category (wan2.1/2.2/2.5/2.6-t2i-*)
      and the Z-Image series.

   2) Image editing (legacy payload / new endpoint): POST
      services/aigc/image2image/image-synthesis. The input field is identical to API 1)
      ({"prompt": ..., "images": [...]}) and it is also task_id / polling mode (same
      X-DashScope-Async: enable header, same output.results[].url), only the resource path
      differs (image2image instead of text2image). Currently only wan2.5-i2i-preview uses it,
      see _IMAGE2IMAGE_LEGACY_MODELS. (The images array accepts at most 3 images, up to 10MB
      each, 384~5000px per side.)

   3) New messages format: synchronous POST services/aigc/multimodal-generation/generation,
      input = {"messages": [{"role": "user", "content": [{"image": "..."}, ..., {"text": "..."}]}]},
      the result is in output.choices[].message.content[] (items of type "image"); there is no
      task_id / polling, and no "ratio" parameter (only "size"). Used by the whole Qwen
      qwen-image family (even plain text-to-image goes through it - the vendor unified
      text-to-image and image-to-image / editing into one API) and by Wan's wan2.6-image /
      wan2.7-image / wan2.7-image-pro (note: these share the "image-generation/editing"
      category with wan2.5-i2i-preview but belong to a different API generation, see the
      exclusion logic around _IMAGE2IMAGE_LEGACY_MODELS).

   _uses_multimodal_api(model_name) and _IMAGE2IMAGE_LEGACY_MODELS decide automatically which
   API each model uses, so callers never need to care. That is also why one tier can contain
   models of different API generations (e.g. wan2.5-i2i-preview and wan2.6-image in the
   "image-generation/editing" category); fallback still switches across APIs transparently and
   the generate_image() / generate_image_urls() interface does not change.
   [note] This classification was verified model by model against the vendor docs and by testing
   wan2.5-i2i-preview. When the vendor adds a model, read that model's own API reference first
   (endpoint / field names / sync vs async) before deciding whether it belongs to
   _uses_multimodal_api() or _IMAGE2IMAGE_LEGACY_MODELS; do not assume it shares an API with the
   other models of the same category just because the category name looks alike (that was
   exactly why wan2.5-i2i-preview was misclassified at first).

   Video models (generate_video) have not hit the same problem and keep using the legacy
   video-generation async-task flow; if they ever return the same "url error", the cause is
   almost certainly the wrong API - consult the vendor docs the same way.

Usage 1: generate an image (text-to-image / image-to-image / multi-image composition, switched
         automatically depending on whether image / images are passed)
    import aliyun_vision_client as av
    url = av.generate_image("A floating city at sunset, cinematic")
    url = av.generate_image("Change the background to a cyberpunk night scene", image="https://.../input.png")
    url = av.generate_image("Change the background to a cyberpunk night scene", image=r"C:\\photos\\input.png")  # local image, see below

Usage 2: generate a video (text-to-video / image-to-video / first-last-frame / reference-to-video)
    url = av.generate_video("A cat plays piano in the rain, slow dolly-in")
    url = av.generate_video("The character turns and walks to the window", first_frame="https://.../a.png",
                            last_frame="https://.../b.png")
    url = av.generate_video("The character turns and walks to the window", first_frame=r"C:\\photos\\a.png")  # local images work too

How to send a local image (applies to image / images / first_frame / last_frame / reference_images):
   DashScope image input fields accept only two formats: (1) a publicly reachable URL and
   (2) a data URI ("data:<mime>;base64,<base64 content>"). Multipart file upload is not
   supported and a local path string means nothing to the API. This module detects it
   automatically: any string that does not start with "http://" / "https://" / "data:" is
   treated as a local file path and passed through encode_local_image() below (read file ->
   base64-encode -> build a data URI) before sending, so callers never have to handle it.

   To reuse this in another project (without importing this whole file), copy just the
   encode_local_image() function and its docstring; the logic is only "read file ->
   base64.b64encode -> build a data:<mime>;base64,<...> string" and it depends on nothing else
   in this module. The technique is not Alibaba-specific: the image_url field of multimodal
   models such as qwen-vl, and the image input of many other LLM APIs, accept the same data URI.

Usage 3: pick a model by tier (see IMAGE_MODEL_TIERS / VIDEO_MODEL_TIERS: flagship / high / mid / low)
    model = av.get_tier_default("image", "high")
    url = av.generate_image("...", model=model)

Usage 3b: for image-to-image / image editing (passing image / images), which model to choose?
    av.get_image_edit_capable_models()   # {tier: [model_name, ...]}: models whose image-input
                                          # support was verified against the vendor docs; the
                                          # safest choice with the most predictable quality
    url = av.generate_image("Extract the subject and remove the background", model="wan2.5-i2i-preview", image=r"C:\\a.png")

    [note] This list is NOT "every model that can take an image": models on the legacy API
    (the Wan "text-to-image" category / Z-Image, i.e. _uses_multimodal_api()==False) can also be
    given image / images even if they are not listed here (the legacy API's images field is a
    generic field accepted regardless of model category); whether the model really edits "along
    with" the reference image varies per model and must be tested (for example one text-to-image
    model of the wan2.2 family worked as a background reference while its flash sibling did not;
    both belong to the "text-to-image" category and neither is in
    get_image_edit_capable_models() - that is a quality difference, not an API-level rule). The
    only combination generate_image_urls() actually blocks, by raising AliyunVisionAPIError, is
    "uses the new messages API but is not confirmed to support image input" (for example passing
    an image to qwen-image-2.0), because misusing that API fails the whole request (400
    "url error") instead of merely producing a less accurate image.

Usage 4: find out which model actually generated an image / video (with fallback it is not
         necessarily the model you passed; needed when labelling outputs to compare models)
    url, used_model = av.generate_image_with_model("...")
    url, used_model = av.generate_video_with_model("...")
    av.last_used_model("image")   # or look it up afterwards: the last model that succeeded

Model classification overview - image models actually have three independent classification
axes; do not mix them up:

    1) Quality tier (IMAGE_MODEL_TIERS / flagship / high / mid / low)
       -> decides "how expensive / how good, roughly"; by default fallback only searches the
         same tier (get_fallback_model_names() / AliyunCrossTierFallbackError).
         Applies to every task (text-to-image or image-to-image); pick with
         get_tier_default(output_type, tier).

    2) API generation (_uses_multimodal_api(), which of the three APIs is used)
       -> purely technical: decides which endpoint / payload shape the client uses internally.
         Callers never need to care; the generate_image() / generate_video() interface does not
         change and fallback switches across generations automatically.

    3) Whether image input is supported (image-to-image / image-editing capability,
       _image_input_allowed() / get_image_edit_capable_models())
       -> the only axis that can actually block a request. Rules:
         - Models on the new messages API: only those listed by get_image_edit_capable_models()
           may be given image / images; others are rejected by generate_image_urls() (misuse
           would return 400).
         - Models on the legacy API: all accept image / images (generic field) but the effect
           varies per model and must be tested - this is why two models with the same tier and
           the same API generation can behave completely differently for image-to-image.

    Plain-language decision flow:
      No input image needed (pure text-to-image) -> pick by tier with get_tier_default(); none of
      the three axes needs any attention.
      Input image needed (image-to-image / editing) -> prefer models in
      get_image_edit_capable_models(); if you want to try a legacy-API model that is not listed
      (for instance for a specific visual style), test it on a small sample before production
      use, and never assume that "same tier / same category name" means the same result.

Command line:
    python aliyun_vision_client.py image "prompt"       generate one image, print the URL
    python aliyun_vision_client.py video "prompt"       generate one video, print the URL
    python aliyun_vision_client.py --usage              list cumulative usage / allowance per model (no API call)

.env settings (shares DASHSCOPE_API_KEY with qwen_client.py, but uses a different base_url, see above):
    DASHSCOPE_API_KEY              required (same key as qwen_client.py)
    ALIYUN_VISION_BASE_URL         DashScope native API base URL, default is the international site
                                    https://dashscope-intl.aliyuncs.com/api/v1
    ALIYUN_IMAGE_MODEL             primary image model; unset = first model of DEFAULT_IMAGE_TIER
    ALIYUN_VIDEO_MODEL             primary video model; unset = first model of DEFAULT_VIDEO_TIER
    ALIYUN_IMAGE_MODEL_<TIER>      default model of that tier (FLAGSHIP / HIGH / MID / LOW); unset = first in the list
    ALIYUN_VIDEO_MODEL_<TIER>      same as above, for video
    ALIYUN_VISION_TIMEOUT_SECONDS  polling timeout: maximum seconds to wait after a task is created, default 600 (raise for video)
    ALIYUN_VISION_POLL_INTERVAL    polling interval in seconds, default 5
    ALIYUN_VISION_QUOTA_RESERVE    reserved allowance (buffer for drift between the local ledger and the console), default 0

[note] The usage ledger is stored in _log/aliyun_vision_usage.json next to this module. It only
       counts calls made through this module (generate_image / generate_video); usage consumed
       elsewhere (console playground, manual tests, other programs calling the API directly)
       must be added with set_used_quota(model_name, used), otherwise the local ledger is more
       optimistic than what the console shows.
"""

import base64
import json
import os
import time

import requests
from dotenv import load_dotenv

_CONFIGURED = False
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# The model that actually served the most recent SUCCESSFUL request. With fallback, callers need
# this to know which model really produced an image / video (used when labelling output file
# names / metadata to compare models).
_LAST_USED = {"image": "", "video": ""}


def last_used_model(output_type):
    """The model actually used by the last successful generate_image / generate_video call (may
    be a fallback model, not necessarily the model passed in). Empty string if none succeeded yet."""
    return _LAST_USED[output_type]

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_TIMEOUT = 600.0
DEFAULT_POLL_INTERVAL = 5.0
DEFAULT_QUOTA_RESERVE = 0

TIER_LABELS = {"flagship": "Flagship", "high": "High", "mid": "Mid", "low": "Low"}
TIER_ORDER = ("flagship", "high", "mid", "low")
DEFAULT_IMAGE_TIER = "high"
DEFAULT_VIDEO_TIER = "high"

MODELS_PATH = os.path.join(_MODULE_DIR, "aliyun_vision_models.json")

# Category / series strings used in aliyun_vision_models.json. The code below compares against
# these constants, so the JSON catalog and the logic stay consistent.
CATEGORY_IMAGE_TRANSLATION = "image-translation"
CATEGORY_IMAGE_EDIT = "image-generation/editing"
CATEGORY_TEXT_TO_VIDEO = "text-to-video"
SERIES_QWEN = "Qwen"


class AliyunVisionAPIError(Exception):
    """Raised when the Alibaba Cloud vision API fails (HTTP 4xx/5xx, task FAILED, polling timeout...)."""
    pass


class AliyunVisionBudgetExceededError(Exception):
    """The cumulative usage of this model is about to exceed its allowance (total_quota); the
    request was NOT sent. Deliberately not a subclass of AliyunVisionAPIError: this is not an API
    error, and retrying or switching to another request would be blocked the same way."""
    pass


class AliyunCrossTierFallbackError(Exception):
    """Raised when the caller's fallback_models contain a model from a different tier than the
    primary model. This is a deliberate safety valve: only models of the same tier may back each
    other up, so a request is never silently downgraded (or upgraded, burning extra allowance)
    when the tiers are unclear."""
    pass


def _load_models():
    with open(MODELS_PATH, encoding="utf-8") as f:
        return json.load(f)


_CATALOG = _load_models()


def _build_tiers(output_type):
    """Return {tier: [(name, note), ...]} in the original JSON order (earlier in the list =
    recommended first).

    [note] category == "image-translation" (currently only qwen-mt-image-2.0) is deliberately
    excluded: it translates the text inside an image, which has a completely different API shape
    from text-to-image / image-to-image (not implemented in this module). It must not be mixed
    into the tier pool as a normal image model, otherwise get_fallback_model_names() could pick it
    as a fallback and the whole task would be sent to the wrong API."""
    tiers = {t: [] for t in TIER_ORDER}
    for m in _CATALOG["models"]:
        if m["output_type"] == output_type and m["tier"] in tiers and m["category"] != CATEGORY_IMAGE_TRANSLATION:
            tiers[m["tier"]].append((m["name"], m.get("note") or ""))
    return tiers


IMAGE_MODEL_TIERS = _build_tiers("image")
VIDEO_MODEL_TIERS = _build_tiers("video")
_MODEL_INFO = {m["name"]: m for m in _CATALOG["models"]}


def _tiers_for(output_type):
    return IMAGE_MODEL_TIERS if output_type == "image" else VIDEO_MODEL_TIERS


def get_model_info(model_name):
    """The full record of this model in aliyun_vision_models.json (series / category / quota...), or None if unknown."""
    return _MODEL_INFO.get(model_name)


def get_model_tier(output_type, model_name):
    """The tier of the model ("flagship" / "high" / "mid" / "low"), or None if it is not in the catalog."""
    for tier, models in _tiers_for(output_type).items():
        if any(name == model_name for name, _note in models):
            return tier
    return None


def get_tier_models(output_type, tier):
    """Model names of a tier (recommended ones first). output_type: "image" or "video"."""
    return [name for name, _note in _tiers_for(output_type)[tier]]


def _supports_no_input_call(output_type, model_name):
    """Can this model be called WITHOUT any image / reference parameter? Used when picking default
    models, to avoid defaulting to a model that always needs a reference image / audio (for example
    reference-to-video or image translation), which would fail as soon as the caller passes only a
    prompt. Decided from the "category" field of aliyun_vision_models.json."""
    info = get_model_info(model_name)
    category = (info or {}).get("category") or ""
    if output_type == "video":
        return category == CATEGORY_TEXT_TO_VIDEO
    return category != CATEGORY_IMAGE_TRANSLATION


def get_tier_default(output_type, tier):
    """Default model of a tier: ALIYUN_IMAGE_MODEL_<TIER> / ALIYUN_VIDEO_MODEL_<TIER> from .env;
    if unset, prefer a model that "can be called without a reference image / audio" (see
    _supports_no_input_call), falling back to the first model of the list if the tier has none;
    returns None if the tier has no models at all."""
    prefix = "ALIYUN_IMAGE_MODEL_" if output_type == "image" else "ALIYUN_VIDEO_MODEL_"
    override = (os.getenv(f"{prefix}{tier.upper()}") or "").strip()
    if override:
        return override
    models = get_tier_models(output_type, tier)
    if not models:
        return None
    for name in models:
        if _supports_no_input_call(output_type, name):
            return name
    return models[0]


def get_default_model_name(output_type):
    """Primary model: ALIYUN_IMAGE_MODEL / ALIYUN_VIDEO_MODEL from .env; if unset, the default
    model of DEFAULT_IMAGE_TIER / DEFAULT_VIDEO_TIER for that output_type."""
    env_name = "ALIYUN_IMAGE_MODEL" if output_type == "image" else "ALIYUN_VIDEO_MODEL"
    override = (os.getenv(env_name) or "").strip()
    if override:
        return override
    default_tier = DEFAULT_IMAGE_TIER if output_type == "image" else DEFAULT_VIDEO_TIER
    return get_tier_default(output_type, default_tier)


def get_fallback_model_names(output_type, primary):
    """The other models in the same tier as primary (excluding itself). If primary is not in any
    tier list, returns an empty list (no known tier, so no guessing who to fall back to)."""
    tier = get_model_tier(output_type, primary)
    if tier is None:
        return []
    return [n for n in get_tier_models(output_type, tier) if n != primary]


def _assert_same_tier(output_type, primary, fallback_models):
    """Every model in fallback_models must be in the same tier as primary, otherwise raise AliyunCrossTierFallbackError."""
    primary_tier = get_model_tier(output_type, primary)
    for m in fallback_models:
        m_tier = get_model_tier(output_type, m)
        if primary_tier is None or m_tier != primary_tier:
            raise AliyunCrossTierFallbackError(
                f"[error] Refusing to use '{m}' (tier: {TIER_LABELS.get(m_tier, m_tier or 'unknown')}) as a "
                f"fallback for '{primary}' (tier: {TIER_LABELS.get(primary_tier, primary_tier or 'unknown')}).\n"
                f"   This module only lets models of the same tier back each other up, to avoid "
                f"silent cross-tier downgrades / upgrades."
            )


def configure(env_path=None, force=False):
    """Load .env and make sure the API key exists. Safe to call repeatedly (by default the check
    really runs only once)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
    if not get_api_key():
        raise EnvironmentError(
            "[error] Environment variable DASHSCOPE_API_KEY not found. Set it in .env "
            "(create an API key in the Alibaba Cloud Model Studio console; it is the same key "
            "used by qwen_client.py)."
        )
    _CONFIGURED = True


def get_api_key():
    return (os.getenv("DASHSCOPE_API_KEY") or "").strip()


def get_base_url():
    return (os.getenv("ALIYUN_VISION_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _env_float(name, default):
    raw = os.getenv(name)
    try:
        return float(raw) if raw and str(raw).strip() else default
    except ValueError:
        return default


def get_timeout(explicit=None):
    if explicit is not None:
        return explicit
    return _env_float("ALIYUN_VISION_TIMEOUT_SECONDS", DEFAULT_TIMEOUT)


def get_poll_interval():
    return _env_float("ALIYUN_VISION_POLL_INTERVAL", DEFAULT_POLL_INTERVAL)


def _headers(async_task=True):
    h = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }
    if async_task:
        h["X-DashScope-Async"] = "enable"
    return h


def _validate_prompt(prompt):
    if not prompt or not str(prompt).strip():
        raise ValueError("[error] The prompt about to be sent to the Alibaba Cloud vision API is an empty string! Check the upstream source.")


# ============================================================
# Local image -> data URI (can be copied as-is into other projects; depends on nothing else in this module)
# ============================================================
# Every DashScope "image input" field - input.images for text/image-to-image, img_url /
# first_frame_url / last_frame_url for video, image_url for multimodal models such as qwen-vl -
# accepts only two formats: (1) a publicly reachable URL, (2) a data URI:
# "data:<mime>;base64,<base64 content>". Multipart/form-data upload is not supported and local
# disk path strings are not understood. The only way to send a local image is "read file ->
# base64-encode -> build a data URI", which is what the function below does.
#
# [note] _MAX_LOCAL_IMAGE_BYTES (10MB) is a conservative default inferred from the common numbers
#    in DashScope's image-input limits. The actual limit differs per model, so check it yourself;
#    this only avoids sending a large request that is bound to be rejected, and is not an
#    authoritative value.
_LOCAL_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
}
_MAX_LOCAL_IMAGE_BYTES = 10 * 1024 * 1024


def encode_local_image(path):
    """Convert a local image file to a data URI usable by DashScope (and other APIs that accept the same format).

    [note] Meant for reuse in other projects: this function deliberately depends on nothing else
       in this module (no _CATALOG / _MODEL_INFO / .env ...). Copy just this function plus the
       "Local image -> data URI" comment above into another program and it works unchanged,
       without moving the whole aliyun_vision_client.py file.

    - path: local image file path (.jpg / .jpeg / .png / .webp / .bmp / .gif)
    - returns: a data URI string such as "data:image/png;base64,iVBORw0KG..."
    - raises ValueError (instead of sending a doomed request) if the file does not exist, cannot
      be read, is empty, has an unknown extension, or exceeds _MAX_LOCAL_IMAGE_BYTES once encoded.
    """
    ext = os.path.splitext(path)[1].lower()
    mime = _LOCAL_IMAGE_EXTENSIONS.get(ext)
    if not mime:
        raise ValueError(
            f"[error] Unrecognized image format '{ext}' ({path}). "
            f"Supported: {', '.join(sorted(_LOCAL_IMAGE_EXTENSIONS))}."
        )
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise ValueError(f"[error] Cannot read local image file '{path}': {e}")
    if not raw:
        raise ValueError(f"[error] Local image file is empty: {path}")

    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > _MAX_LOCAL_IMAGE_BYTES:
        raise ValueError(
            f"[error] '{path}' is {len(encoded):,} bytes after base64 encoding, above the "
            f"recommended limit of {_MAX_LOCAL_IMAGE_BYTES:,} bytes (a conservative value inferred "
            f"from DashScope's image-input limits; check the real limit of your model. If needed, "
            f"compress the image first or pass a publicly reachable URL instead)."
        )
    return f"data:{mime};base64,{encoded}"


def _looks_like_local_path(value):
    """Rough check whether a string is a local file path: anything that does not start with http(s):// or data: is treated as one."""
    return isinstance(value, str) and bool(value) and not value.startswith(("http://", "https://", "data:"))


def _resolve_image_ref(value):
    """Shared conversion for single-image parameters such as image / first_frame / last_frame:
    URLs and data URIs pass through unchanged; anything that looks like a local path is turned
    into a data URI by encode_local_image()."""
    if value is None or not _looks_like_local_path(value):
        return value
    return encode_local_image(value)


def _resolve_image_refs(values):
    """Shared conversion for "many at once" parameters such as images / reference_images,
    applying _resolve_image_ref to each element."""
    if values is None:
        return None
    return [_resolve_image_ref(v) for v in values]


# ============================================================
# Usage budget: cumulative usage per model must not exceed the console allowance (total_quota)
# ============================================================
_LOG_DIR = os.path.join(_MODULE_DIR, "_log")
os.makedirs(_LOG_DIR, exist_ok=True)
USAGE_LEDGER_PATH = os.path.join(_LOG_DIR, "aliyun_vision_usage.json")


def get_quota_reserve():
    return int(_env_float("ALIYUN_VISION_QUOTA_RESERVE", DEFAULT_QUOTA_RESERVE))


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


def get_used_quota(model_name):
    return int((_load_ledger().get(model_name) or {}).get("used") or 0)


def set_used_quota(model_name, used):
    """Manually correct the ledger (for example to match the usage shown in the console)."""
    ledger = _load_ledger()
    ledger[model_name] = {"used": int(used), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_ledger(ledger)


def _add_to_ledger(model_name, count):
    if count:
        set_used_quota(model_name, get_used_quota(model_name) + count)


def _check_budget(model_name, worst_case):
    info = get_model_info(model_name)
    limit = int(info["total_quota"]) if info and info.get("total_quota") is not None else 0
    if limit <= 0:
        return   # no allowance information in the catalog = no cap
    used = get_used_quota(model_name)
    allowed = limit - get_quota_reserve()
    if used + worst_case > allowed:
        raise AliyunVisionBudgetExceededError(
            f"[blocked] {model_name} has used {used:,} and this call may use up to {worst_case:,} more, "
            f"which would exceed the budget cap {allowed:,} (catalog allowance {limit:,} - "
            f"reserve {get_quota_reserve():,}). The request was not sent. Switch to a model that "
            f"still has allowance, or verify the usage and adjust the ledger.")


def budget_status(model_name):
    info = get_model_info(model_name)
    limit = int(info["total_quota"]) if info and info.get("total_quota") is not None else 0
    used = get_used_quota(model_name)
    if limit <= 0:
        return f"{model_name}: used {used:,} (no allowance information in the catalog)"
    return f"{model_name}: used {used:,} / {limit:,} ({used / limit:.1%}), {max(limit - used, 0):,} left"


# ============================================================
# DashScope async tasks: create a task -> poll for the result
# ============================================================
_QUOTA_MARKERS = ("arrearage", "insufficientbalance", "quotaexceeded", "freetieronly")


def _create_task(resource_path, payload, max_retries, label):
    """POST to create an async task and return (task_id, True); (None, False) means "give up on
    this model, switch to a fallback"."""
    url = f"{get_base_url()}/{resource_path}"
    delay = 5.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=30)
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Alibaba Cloud task creation connection error ({type(e).__name__}), retrying in {delay:.0f}s... ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code >= 400:
            body_lower = resp.text.lower()
            try:
                code_lower = str(resp.json().get("code") or "").lower()
            except (ValueError, AttributeError):
                code_lower = ""

            if resp.status_code in (400, 403) and any(m in body_lower or m in code_lower for m in _QUOTA_MARKERS):
                print(f"   [skip] Alibaba Cloud model allowance exhausted or account in arrears (HTTP {resp.status_code}); switching to a fallback model.")
                return None, False

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = AliyunVisionAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                print(f"   [wait] Alibaba Cloud returned HTTP {resp.status_code}, retrying in {delay:.0f}s... ({attempt}/{max_retries}) {label or ''}")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code in (404,) or "modelnotfound" in code_lower or "invalidparameter" in code_lower:
                print(f"   [skip] Alibaba Cloud model not found or parameters rejected (HTTP {resp.status_code}: {resp.text[:200]}); switching to a fallback model.")
                return None, False

            if resp.status_code == 401:
                raise AliyunVisionAPIError(
                    f"[error] Alibaba Cloud API 401: invalid key, or the key's region does not match ALIYUN_VISION_BASE_URL ({get_base_url()}). {resp.text[:300]}")

            raise AliyunVisionAPIError(f"[error] Alibaba Cloud task creation failed HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        task_id = (data.get("output") or {}).get("task_id")
        if not task_id:
            last_error = AliyunVisionAPIError(f"Response has no task_id: {str(data)[:300]}")
            print(f"   [warn] Unexpected Alibaba Cloud task creation response, retrying in {delay:.0f}s... ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        return task_id, True

    print(f"   [error] Alibaba Cloud task creation still failing after {max_retries} retries: {last_error}")
    return None, False


def _poll_task(task_id, model_name, timeout, label):
    """Poll the task until SUCCEEDED / FAILED / timeout. Returns (urls, usage_count); empty urls means failure."""
    url = f"{get_base_url()}/tasks/{task_id}"
    interval = get_poll_interval()
    deadline = time.monotonic() + timeout
    next_beat = time.monotonic() + 30.0

    while True:
        try:
            resp = requests.get(url, headers=_headers(async_task=False), timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"   [warn] Alibaba Cloud task polling connection error ({type(e).__name__}), retrying in {interval:.0f}s...")
            time.sleep(interval)
            if time.monotonic() > deadline:
                raise AliyunVisionAPIError(f"[error] {model_name} polling timed out ({timeout:.0f}s) and the service is still unreachable.")
            continue

        output = data.get("output") or {}
        status = output.get("task_status")

        if status == "SUCCEEDED":
            results = output.get("results") or []
            urls = [r.get("url") for r in results if isinstance(r, dict) and r.get("url")]
            if not urls and isinstance(output.get("video_url"), str):
                urls = [output["video_url"]]
            usage = (data.get("usage") or {})
            count = usage.get("image_count") or usage.get("video_count") or len(urls) or 1
            label_note = f" | {label}" if label else ""
            print(f"   [ok] Alibaba Cloud {model_name} task completed{label_note}")
            return urls, count

        if status == "FAILED":
            raise AliyunVisionAPIError(
                f"[error] Alibaba Cloud {model_name} task failed: {output.get('code')} {output.get('message')}")

        now = time.monotonic()
        if now > deadline:
            raise AliyunVisionAPIError(f"[error] {model_name} polling timed out ({timeout:.0f}s); task is still {status}.")
        if now >= next_beat:
            print(f"   [wait] Alibaba Cloud {model_name} task status {status}... waited {timeout - (deadline - now):.0f}s")
            next_beat = now + 30.0
        time.sleep(interval)


# ============================================================
# Image generation
# ============================================================
# APIs 1) and 2) share this payload shape ({"prompt":..., "images":[...]}); they differ only in
# the resource path, see _legacy_task_resource_path() below.
def _build_image_input(prompt, image, images, negative_prompt):
    input_ = {"prompt": prompt}
    refs = images or ([image] if image else None)
    if refs:
        input_["images"] = refs
    if negative_prompt:
        input_["negative_prompt"] = negative_prompt
    return input_


_TEXT2IMAGE_RESOURCE_PATH = "services/aigc/text2image/image-synthesis"
_IMAGE2IMAGE_RESOURCE_PATH = "services/aigc/image2image/image-synthesis"

# API 2): payload / response format identical to API 1), only the resource path differs
# (image2image/image-synthesis instead of text2image/image-synthesis). Currently only
# wan2.5-i2i-preview takes this route (verified by testing and against the vendor's
# wan2-5-image-edit-api-reference). If other Alibaba models use this endpoint in the future,
# add the model name to this set; no other code needs to change.
_IMAGE2IMAGE_LEGACY_MODELS = {"wan2.5-i2i-preview"}


def _legacy_task_resource_path(model_name):
    """Which resource path APIs 1) / 2) (task_id / polling mode) should call."""
    return _IMAGE2IMAGE_RESOURCE_PATH if model_name in _IMAGE2IMAGE_LEGACY_MODELS else _TEXT2IMAGE_RESOURCE_PATH


# ------------------------------------------------------------
# API 3): new multimodal-generation/generation (the whole Qwen qwen-image family, and Wan's
# wan2.6-image / wan2.7-image / wan2.7-image-pro). A synchronous call that returns the result
# directly; no task_id / polling.
# ------------------------------------------------------------
_MULTIMODAL_GENERATION_PATH = "services/aigc/multimodal-generation/generation"


def _uses_multimodal_api(model_name):
    """Should this image model use API 3) (new messages format) or APIs 1) / 2) (task_id / polling)?
    Decided from the series / category in aliyun_vision_models.json, based on a check against the
    vendor docs: the whole Qwen qwen-image family = new; within the Wan "image-generation/editing"
    category, everything except the models listed in _IMAGE2IMAGE_LEGACY_MODELS (currently
    wan2.5-i2i-preview) = new; the Wan "text-to-image" category + the Z-Image series = legacy (API 1).

    [note] category == "image-translation" (qwen-mt-image-2.0) is deliberately excluded: it
    belongs to neither of the two APIs but to a completely different API shape that this module
    does not implement (see the explicit rejection at the start of generate_image_urls())."""
    if model_name in _IMAGE2IMAGE_LEGACY_MODELS:
        return False
    info = get_model_info(model_name)
    if not info or info.get("category") == CATEGORY_IMAGE_TRANSLATION:
        return False
    return info.get("series") == SERIES_QWEN or info.get("category") == CATEGORY_IMAGE_EDIT


# ============================================================
# Which image models are CONFIRMED to support image input (image-to-image / editing), not just which API they use
# ============================================================
# [note] "new vs legacy API" is a question of WHICH ENDPOINT to call; whether image input is
#    supported is a completely separate question, and "uses the new API" must not be used to
#    infer "takes images" - the qwen-image-2.0 / qwen-image-2.0-pro series also use the new API
#    (API 3), but the vendor docs do not confirm image-to-image support for them
#    (qwen-image-generation-and-editing-api-reference lists only 2.1-pro / 2.1-turbo / 3.0 /
#    3.0-pro). Passing an image to them anyway risks the same kind of failure that qwen-image-2.0
#    once showed ("url error"): a wrong usage without a clear error message.
#
# So this module deliberately splits the decision into two layers:
#   _uses_multimodal_api(): decides "which endpoint / payload shape" (purely technical).
#   _supports_image_input(): decides "can this model take an image" (capability; conservative
#     default = when unsure, treat as unsupported, never guess - a model that is not listed
#     below is not "confirmed unsupported", this module merely found no vendor doc confirming it).
#   When generate_image_urls() receives image / images it keeps only models for which
#   _supports_image_input() is True in models_to_try (fallbacks included), so a model that does
#   not actually take images is never used as a fallback just because "its tier is the same".
_QWEN_IMAGE_EDIT_CAPABLE = {"qwen-image-2.1-pro", "qwen-image-2.1-turbo", "qwen-image-3.0", "qwen-image-3.0-pro"}
# Source: the vendor's qwen-image-generation-and-editing-api-reference page (the qwen-image-2.0 /
# 2.0-pro series docs do not mention image-to-image capability, so they are not in this list).


def _supports_image_input(model_name):
    """Is this image model "confirmed" to accept reference images (image-to-image / editing)?
    Conservative default: every model whose category == "image-generation/editing" (the category
    name itself implies support), plus the Qwen versions explicitly verified in
    _QWEN_IMAGE_EDIT_CAPABLE; everything else is treated as unsupported, even if it uses the new
    multimodal-generation API (see the explanation above: they are different dimensions)."""
    if model_name in _QWEN_IMAGE_EDIT_CAPABLE:
        return True
    info = get_model_info(model_name)
    return bool(info and info.get("category") == CATEGORY_IMAGE_EDIT)


def get_image_edit_capable_models():
    """Return the image models currently "confirmed to support image input", grouped by tier:
    {tier: [model_name, ...]}. For image-to-image / editing (for example extracting a subject or
    local edits), picking from here is the safest option with the most predictable quality.

    [note] This list only contains models "verified one by one against the vendor docs"; it is
    not the complete list of "which models may be given an image". Models on the legacy API
    (text2image / image2image-image-synthesis, see _uses_multimodal_api()) can also be given
    image / images even if they are not listed here (the legacy API's images field is a generic
    field accepted regardless of category, and generate_image_urls() does not block it); whether
    they really edit "along with" the reference image is a quality question (results vary per
    model, found by trial and error), not an API-level guarantee, so they are not in this
    "confirmed" list; evaluate the result yourself. What this list really blocks are models on
    the new messages API (_uses_multimodal_api()==True) that are not confirmed to support image
    input - misusing that API genuinely fails with 400 "url error" (that is how qwen-image-2.0
    was caught), which is not a matter of slightly worse quality."""
    result = {t: [] for t in TIER_ORDER}
    for tier in TIER_ORDER:
        for name, _note in IMAGE_MODEL_TIERS[tier]:
            if _supports_image_input(name):
                result[tier].append(name)
    return result


def _image_input_allowed(model_name):
    """When generate_image_urls() receives image / images, may this model stay in the
    models_to_try candidate list (i.e. should it be filtered out)?

    - Legacy API (text2image / image2image-image-synthesis): always allowed. The images field is
      a generic field of both payloads and is accepted regardless of model category; it does not
      fail the whole request for a "model/endpoint mismatch" - this has been verified in
      production (a text-to-image model of the wan2.2 family was used as a background generator
      with reference images in many runs). Whether the result is good varies per model and is a
      quality question, outside what this function blocks.
    - New API (multimodal-generation): only models confirmed by _supports_image_input() are
      allowed, because this API has a tested "model/endpoint mismatch -> 400 url error" risk
      (qwen-image-2.0 failed exactly that way); allowing the rest would make callers step into the
      same pit again."""
    if not _uses_multimodal_api(model_name):
        return True
    return _supports_image_input(model_name)


# ============================================================
# Tested low + mid tier fallback list for subject extraction (image-to-image that must keep the source's style / pose)
# ============================================================
# Test setup: the same watercolor-illustration character image was fed to every "text-to-image"
# category model in the low + mid tiers for the same task (extract the subject, remove the
# background, keep the style). Results:
#   wan2.1-t2i-turbo / wan2.2-t2i-flash / wan2.5-t2i-preview / wan2.2-t2i-plus
#     all turned the style into "photorealistic portrait photography", ignoring the illustration
#     style of the reference image and picking up only loose features (sweater + cross-legged
#     pose) - these models treat the reference image as loose inspiration instead of really
#     "editing" it, so they are unsuitable for subject extraction that must preserve the source
#     style (they may still be fine for looser tasks such as background generation, which is a
#     different matter and must not be applied to subject extraction).
#   wan2.1-t2i-plus: 180-second polling timeout, no result.
#   z-image-turbo: rejects image input even on the legacy API (HTTP 400 url error); unsupported.
#   qwen-image-2.0: new API, image input not confirmed; correctly blocked by _image_input_allowed().
# The only two that passed (style and pose preserved), and also all members of
# get_image_edit_capable_models() for tier = low / mid:
SUBJECT_EXTRACTION_FALLBACK_MODELS = ["wan2.5-i2i-preview", "qwen-image-2.1-turbo"]
# Ordered by tested quality (wan2.5-i2i-preview is clean and complete; qwen-image-2.1-turbo
# occasionally crops / shifts the subject but still keeps the style, so it is second). These two
# models belong to different tiers (low / mid). To use them together as fallbacks, pass
# allow_cross_tier_fallback=True to generate_image() / generate_image_urls(), otherwise
# AliyunCrossTierFallbackError is raised (see that parameter's description).
# The high tier (wan2.6-image / wan2.7-image / wan2.7-image-pro / qwen-image-2.1-pro / 3.0 /
# 3.0-pro) is deliberately NOT in this list: this is a cost-saving combination for routine
# high-volume subject extraction; the high-tier allowance is kept for cases that really need
# higher quality and must not be consumed silently by automatic fallback. To use a high-tier
# model, specify it manually instead of relying on this list.


def _call_multimodal_image(model_name, prompt, image_refs, size, n, timeout, max_retries, label):
    """Synchronous call to multimodal-generation/generation. image_refs may be None (pure
    text-to-image) or a list of URLs / data URIs (image-to-image / multi-image reference; per the
    vendor docs, images come first and the text prompt last).

    [note] This is a synchronous API: generation completes inside a single HTTP request /
    response with no polling, so this timeout is the total time available for the whole
    generation (it is the same number as get_timeout() uses for polling, but means something
    different: here it is the timeout of one HTTP request, not the total polling duration).

    The return convention matches _create_task(): (urls, True) = success; (None, False) = give up
    on this model and switch to a fallback (allowance exhausted / model not found / parameters
    rejected / retries exhausted); truly unretryable errors such as 401 raise AliyunVisionAPIError
    directly without trying other models."""
    content = [{"image": ref} for ref in (image_refs or [])]
    content.append({"text": prompt})
    payload = {"model": model_name, "input": {"messages": [{"role": "user", "content": content}]}}
    parameters = {}
    if size:
        parameters["size"] = size
    if n:
        parameters["n"] = n
    if parameters:
        payload["parameters"] = parameters

    url = f"{get_base_url()}/{_MULTIMODAL_GENERATION_PATH}"
    delay = 5.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=_headers(async_task=False), json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Alibaba Cloud multimodal-generation connection error ({type(e).__name__}), retrying in {delay:.0f}s... ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code >= 400:
            body_lower = resp.text.lower()
            try:
                code_lower = str(resp.json().get("code") or "").lower()
            except (ValueError, AttributeError):
                code_lower = ""

            if resp.status_code in (400, 403) and any(m in body_lower or m in code_lower for m in _QUOTA_MARKERS):
                print(f"   [skip] Alibaba Cloud model allowance exhausted or account in arrears (HTTP {resp.status_code}); switching to a fallback model.")
                return None, False

            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = AliyunVisionAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                print(f"   [wait] Alibaba Cloud returned HTTP {resp.status_code}, retrying in {delay:.0f}s... ({attempt}/{max_retries}) {label or ''}")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue

            if resp.status_code in (404,) or "modelnotfound" in code_lower or "invalidparameter" in code_lower:
                print(f"   [skip] Alibaba Cloud model not found or parameters rejected (HTTP {resp.status_code}: {resp.text[:200]}); switching to a fallback model.")
                return None, False

            if resp.status_code == 401:
                raise AliyunVisionAPIError(
                    f"[error] Alibaba Cloud API 401: invalid key, or the key's region does not match ALIYUN_VISION_BASE_URL ({get_base_url()}). {resp.text[:300]}")

            raise AliyunVisionAPIError(f"[error] Alibaba Cloud multimodal-generation failed HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        output = data.get("output") or {}
        urls = []
        for choice in output.get("choices") or []:
            for item in (choice.get("message") or {}).get("content") or []:
                if isinstance(item, dict) and item.get("image"):
                    urls.append(item["image"])

        if not urls:
            last_error = AliyunVisionAPIError(f"Response has no image: {str(data)[:300]}")
            print(f"   [warn] Alibaba Cloud {model_name} returned no image, retrying in {delay:.0f}s... ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        usage = data.get("usage") or {}
        count = usage.get("image_count") or len(urls)
        label_note = f" | {label}" if label else ""
        print(f"   [ok] Alibaba Cloud {model_name} (multimodal-generation) completed{label_note}")
        return urls, count

    print(f"   [error] Alibaba Cloud {model_name} multimodal-generation still failing after {max_retries} retries: {last_error}")
    return None, False


# ============================================================
# ratio -> size automatic conversion (the new API 3) does not accept a ratio parameter. When the
# caller gives only a ratio, the client converts it to a valid size string so that callers do not
# have to look up the vendor's ratio / pixel table themselves)
# ============================================================
# Basis:
#   - Wan "image-generation/editing" category (wan2.6-image / wan2.7-image / wan2.7-image-pro):
#     the "recommended resolutions for common ratios" table in the vendor's
#     wan-image-generation-api-reference (matching the "1K" tier of image editing mode),
#     transcribed below. The vendor only publishes these 8 entries; other ratios are derived
#     from the same total pixel count (1280*1280). Valid range: total pixels 768*768~2048*2048,
#     aspect ratio 1:4~4:1 (out of range raises ValueError, rather than sending a request that
#     is bound to be rejected).
#     [note] This table was transcribed from the wan2.6-image docs. wan2.7-image / wan2.7-image-pro
#     support higher tiers (2K or more), but the vendor does not publish a table specific to those
#     two models, so the same table is reused as a conservative default - it only guarantees the
#     right ratio, not the highest resolution available to those models; pass size directly to
#     push resolution higher.
#   - Qwen qwen-image family: the vendor's qwen-image-generation-and-editing-api-reference has no
#     ratio table, only range limits (total pixels 512*512~2048*2048, aspect ratio 1:8~8:1) and an
#     example value "1024*1024". With no table to consult, a formula is used: take the docs'
#     example 1024*1024 (about 1.05M pixels) as the base total pixel count, derive width / height
#     from the ratio, round to a multiple of 16 (both docs say output sizes are "multiples of
#     16", a general DashScope image-model convention), then clamp to the valid range.
_WAN_IMAGE_EDIT_RATIO_SIZES = {
    "1:1": "1280*1280", "2:3": "800*1200", "3:2": "1200*800",
    "3:4": "960*1280", "4:3": "1280*960", "9:16": "720*1280",
    "16:9": "1280*720", "21:9": "1344*576",
}
_QWEN_IMAGE_DEFAULT_TOTAL_PIXELS = 1024 * 1024
_QWEN_IMAGE_SIDE_RANGE = (512, 2048)
_QWEN_IMAGE_ASPECT_LIMIT = 8.0
_WAN_IMAGE_EDIT_DEFAULT_TOTAL_PIXELS = 1280 * 1280
_WAN_IMAGE_EDIT_SIDE_RANGE = (768, 2048)
_WAN_IMAGE_EDIT_ASPECT_LIMIT = 4.0


def _parse_ratio(ratio):
    """Parse strings such as "16:9" / "16*9" / "16/9" into (width weight, height weight). A bad
    format raises ValueError immediately - a wrong format from the caller must show a clear error
    right away instead of silently producing an odd size."""
    raw = str(ratio).strip().replace("*", ":").replace("/", ":")
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"[error] Cannot parse ratio '{ratio}'; use the 'width:height' format, for example \"16:9\".")
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError(f"[error] Cannot parse ratio '{ratio}'; use the 'width:height' format, for example \"16:9\".")
    if w <= 0 or h <= 0:
        raise ValueError(f"[error] ratio must be positive: '{ratio}'.")
    return w, h


def _round_to_multiple_of_16(value, min_side, max_side):
    value = max(min_side, min(max_side, value))
    rounded = int(round(value / 16)) * 16
    return max(min_side, min(max_side, rounded))


def _ratio_to_size(ratio, default_total_pixels, side_range, aspect_limit, limit_label):
    w_ratio, h_ratio = _parse_ratio(ratio)
    aspect = w_ratio / h_ratio
    if not (1 / aspect_limit <= aspect <= aspect_limit):
        raise ValueError(f"[error] ratio '{ratio}' is outside the aspect range 1:{aspect_limit:g}~{aspect_limit:g}:1 allowed by {limit_label}.")
    height = (default_total_pixels / aspect) ** 0.5
    width = height * aspect
    min_side, max_side = side_range
    return (f"{_round_to_multiple_of_16(width, min_side, max_side)}*"
            f"{_round_to_multiple_of_16(height, min_side, max_side)}")


def _resolve_size_for_multimodal_model(model_name, size, ratio):
    """API 3) does not accept ratio, only size. size wins; when only a ratio is given, convert it
    to a valid size string according to the model family (Wan image editing vs Qwen qwen-image),
    see the notes above."""
    if size:
        return size
    if not ratio:
        return None
    normalized = str(ratio).strip().replace("*", ":").replace("/", ":")
    info = get_model_info(model_name)
    if info and info.get("category") == CATEGORY_IMAGE_EDIT:
        if normalized in _WAN_IMAGE_EDIT_RATIO_SIZES:
            return _WAN_IMAGE_EDIT_RATIO_SIZES[normalized]
        return _ratio_to_size(ratio, _WAN_IMAGE_EDIT_DEFAULT_TOTAL_PIXELS, _WAN_IMAGE_EDIT_SIDE_RANGE,
                              _WAN_IMAGE_EDIT_ASPECT_LIMIT, "Wan image-editing mode")
    return _ratio_to_size(ratio, _QWEN_IMAGE_DEFAULT_TOTAL_PIXELS, _QWEN_IMAGE_SIDE_RANGE,
                          _QWEN_IMAGE_ASPECT_LIMIT, "Qwen qwen-image ")


def generate_image(prompt, model=None, image=None, images=None, size=None, ratio=None, n=1,
                   negative_prompt=None, timeout=None, max_retries=3, raise_on_failure=True,
                   label=None, allow_model_fallback=True, fallback_models=None,
                   allow_cross_tier_fallback=False):
    """
    Generate an image (text-to-image; passing image or images switches to image-to-image /
    multi-image composition, handled by the same endpoint).

    - prompt: text description
    - image / images: optional reference images - a publicly reachable URL, a data URI, or a
      "local file path" (for example r"C:\\photos\\input.png", automatically converted to a data
      URI by encode_local_image(); see "How to send a local image" at the top of this file).
      images takes several, image takes one (use one of them; if both are given, images wins)
    - model: optional, defaults to get_default_model_name("image")
    - size: optional, for example "1024*1024" (supported sizes differ per model; check the console)
    - ratio: optional, for example "16:9". Models on the legacy API (Wan "text-to-image" category /
      Z-Image) get the ratio passed to the API directly; models on the new API (Qwen qwen-image
      series / Wan "image-generation/editing" category) do not accept a ratio field, so the client
      converts it to a valid size string according to the model's family before sending (see
      _resolve_size_for_multimodal_model()); callers do not need to care about the underlying
      difference. When both size and ratio are given, size wins and ratio is ignored
    - n: how many images to generate this time, default 1
    - returns: an image URL string (with n>1, the first one of the list; use generate_image_urls for all of them)
    - When fallback_models is not given, fallback is automatic according to
      get_fallback_model_names("image", model) (searching only within the same tier); when given,
      every model must be in the same tier as model, otherwise AliyunCrossTierFallbackError is
      raised - unless allow_cross_tier_fallback=True (see that parameter's description).
    - allow_cross_tier_fallback: default False (keeps the original "only same-tier models may back
      each other up" safety valve). Set it to True to bypass the same-tier restriction ONLY when
      fallback_models is passed explicitly AND you have tested that those models can replace each
      other (not merely because their tier names match) - for example subject extraction with
      SUBJECT_EXTRACTION_FALLBACK_MODELS (see that constant) spans the low / mid tiers, because
      testing showed that "tier" does not tell whether an image-to-image model keeps the style, so
      the list must be specified explicitly rather than guessed by automatic fallback. Has no
      effect when fallback_models is not passed (pure automatic fallback).
    """
    urls = generate_image_urls(prompt, model=model, image=image, images=images, size=size,
                               ratio=ratio, n=n, negative_prompt=negative_prompt, timeout=timeout,
                               max_retries=max_retries, raise_on_failure=raise_on_failure, label=label,
                               allow_model_fallback=allow_model_fallback, fallback_models=fallback_models,
                               allow_cross_tier_fallback=allow_cross_tier_fallback)
    return urls[0] if urls else ""


def generate_image_with_model(prompt, **kwargs):
    """Same as generate_image, but returns (url, model actually used). With fallback, the model
    passed in by the caller is not necessarily the model that generated the image; to label output
    images with "which model made this one" for model evaluation, use this function instead of
    generate_image() plus guessing."""
    url = generate_image(prompt, **kwargs)
    return url, last_used_model("image")


def generate_image_urls(prompt, model=None, image=None, images=None, size=None, ratio=None, n=1,
                        negative_prompt=None, timeout=None, max_retries=3, raise_on_failure=True,
                        label=None, allow_model_fallback=True, fallback_models=None,
                        allow_cross_tier_fallback=False):
    """Same as generate_image, but returns the full list of URLs (use when n>1 or when the model returns several images at once)."""
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_model_name("image")).strip()
    if not model_name:
        raise AliyunVisionAPIError(f"[error] No usable image model (is the DEFAULT_IMAGE_TIER={DEFAULT_IMAGE_TIER} list empty?).")

    model_info = get_model_info(model_name)
    if model_info and model_info.get("category") == CATEGORY_IMAGE_TRANSLATION:
        raise AliyunVisionAPIError(
            f"[error] {model_name} is an 'image translation' model (it translates the text inside an image); "
            f"its API shape is completely different from text-to-image / image-to-image and is not "
            f"implemented in this module, so it cannot be called through generate_image().")

    models_to_try = _models_to_try("image", model_name, allow_model_fallback, fallback_models,
                                   allow_cross_tier_fallback)

    image = _resolve_image_ref(image)
    images = _resolve_image_refs(images)
    image_refs = images or ([image] if image else None)

    if image_refs:
        # Filter only when images were passed, and only drop combinations that "would really hit
        # the wrong API and fail the whole request" (new messages API but image input not
        # confirmed). Legacy-API models are untouched: their images field is generic and allowed
        # for every category, see the full explanation in _image_input_allowed().
        capable = [m for m in models_to_try if _image_input_allowed(m)]
        if not capable:
            rest = ", ".join(models_to_try[1:]) or "(none)"
            raise AliyunVisionAPIError(
                f"[error] image / images were passed, but {model_name} and its fallback models {rest} are all "
                f"models that 'use the new messages API without confirmed image-input support'; sending "
                f"the request would most likely fail with 400 'url error' (model/endpoint mismatch). "
                f"Use a model listed by get_image_edit_capable_models(), for example "
                f"wan2.5-i2i-preview / wan2.6-image / wan2.7-image / wan2.7-image-pro, or Qwen "
                f"qwen-image-2.1-pro / qwen-image-2.1-turbo / qwen-image-3.0 / qwen-image-3.0-pro; "
                f"alternatively use a model on the legacy API (Wan 'text-to-image' category / "
                f"Z-Image), whose effect varies per model and should be evaluated yourself."
            )
        models_to_try = capable

    # Payload for the legacy API (text2image/image-synthesis); only used if the model really takes the legacy path
    input_ = _build_image_input(prompt, image, images, negative_prompt)
    parameters = {"n": n}
    if size:
        parameters["size"] = size
    if ratio:
        parameters["ratio"] = ratio

    worst_case = max(n, 1)
    for current_model in models_to_try:
        if current_model != models_to_try[0]:
            print(f"   [fallback] Switching to fallback model: {current_model}")
        _check_budget(current_model, worst_case)

        if _uses_multimodal_api(current_model):
            # New messages-format API (the whole Qwen qwen-image family / Wan "image-generation/
            # editing" category), see the API overview at the top of this file. This API does not
            # accept ratio, so when only a ratio (no size) is given the client converts it to a
            # valid size string (see the vendor basis above _resolve_size_for_multimodal_model());
            # negative_prompt is not a field of this API and has no equivalent, so we can only say
            # once that callers should write the exclusions into the prompt.
            try:
                effective_size = _resolve_size_for_multimodal_model(current_model, size, ratio)
            except ValueError as e:
                # Valid aspect ranges differ per model family (Qwen 1:8~8:1, Wan image editing
                # 1:4~4:1). A ratio that is invalid for this candidate may still be valid for the
                # other fallback candidates, so skip only this model instead of raising and
                # breaking the whole fallback chain.
                print(f"   [warn] Skipping {current_model}: {e}")
                continue
            if ratio and not size:
                print(f"   [size] {current_model} does not support ratio; converted ratio '{ratio}' to "
                      f"size=\"{effective_size}\".")
            if negative_prompt:
                print(f"   [warn] {current_model} uses the new multimodal-generation API, which does not support negative_prompt"
                      f" (ignored; write the exclusions into the prompt itself).")
            urls, count = _call_multimodal_image(current_model, prompt, image_refs, effective_size, n,
                                                 get_timeout(timeout), max_retries, label)
        else:
            payload = {"model": current_model, "input": input_, "parameters": parameters}
            resource_path = _legacy_task_resource_path(current_model)
            task_id, ok = _create_task(resource_path, payload, max_retries, label)
            if not ok:
                continue
            try:
                urls, count = _poll_task(task_id, current_model, get_timeout(timeout), label)
            except AliyunVisionAPIError as e:
                print(f"   [error] {e}")
                continue

        if urls:
            _add_to_ledger(current_model, count)
            _LAST_USED["image"] = current_model
            return urls

    msg = f"[error] Alibaba Cloud tried all {len(models_to_try)} models ({', '.join(models_to_try)}) and every one failed (image)."
    if raise_on_failure:
        raise AliyunVisionAPIError(msg)
    print(msg)
    return []


# ============================================================
# Video generation
# ============================================================
def _build_video_input(prompt, image, first_frame, last_frame, reference_images, reference_audios):
    input_ = {"prompt": prompt}
    if first_frame or last_frame:
        if first_frame:
            input_["first_frame_url"] = first_frame
        if last_frame:
            input_["last_frame_url"] = last_frame
    elif image:
        input_["img_url"] = image
    if reference_images:
        input_["images"] = reference_images
    if reference_audios:
        input_["audios"] = reference_audios
    return input_


def generate_video(prompt, model=None, image=None, first_frame=None, last_frame=None,
                   reference_images=None, reference_audios=None, duration=None, size=None, ratio=None,
                   timeout=None, max_retries=3, raise_on_failure=True, label=None,
                   allow_model_fallback=True, fallback_models=None):
    """
    Generate a video. The mode is decided automatically from the parameters passed:
    - none passed: text-to-video
    - image: image-to-video
    - first_frame / last_frame: first-last-frame video (pass at least one of them)
    - reference_images / reference_audios: reference-to-video

    - image / first_frame / last_frame / reference_images: same as image / images of
      generate_image - a publicly reachable URL, a data URI, or a local file path (converted
      automatically, see the explanation at the top of this file)
    - model: optional, defaults to get_default_model_name("video")
    - duration: optional, video length in seconds (string or number; the supported range and
      maximum differ per model)
    - size / ratio: optional, output resolution / aspect ratio; check the tiers supported by the
      model in the console
    - returns: a video URL string (on failure, raises or returns an empty string depending on raise_on_failure)
    - When fallback_models is not given, fallback is automatic according to
      get_fallback_model_names("video", model) (searching only within the same tier); when given,
      every model must be in the same tier as model, otherwise AliyunCrossTierFallbackError is raised.
    """
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_model_name("video")).strip()
    if not model_name:
        raise AliyunVisionAPIError(f"[error] No usable video model (is the DEFAULT_VIDEO_TIER={DEFAULT_VIDEO_TIER} list empty?).")

    models_to_try = _models_to_try("video", model_name, allow_model_fallback, fallback_models)

    image = _resolve_image_ref(image)
    first_frame = _resolve_image_ref(first_frame)
    last_frame = _resolve_image_ref(last_frame)
    reference_images = _resolve_image_refs(reference_images)
    input_ = _build_video_input(prompt, image, first_frame, last_frame, reference_images, reference_audios)
    parameters = {}
    if duration is not None:
        parameters["duration"] = str(duration)
    if size:
        parameters["size"] = size
    if ratio:
        parameters["ratio"] = ratio

    for current_model in models_to_try:
        if current_model != models_to_try[0]:
            print(f"   [fallback] Switching to fallback model: {current_model}")
        _check_budget(current_model, 1)
        payload = {"model": current_model, "input": input_}
        if parameters:
            payload["parameters"] = parameters
        task_id, ok = _create_task("services/aigc/video-generation/video-synthesis", payload, max_retries, label)
        if not ok:
            continue
        try:
            urls, count = _poll_task(task_id, current_model, get_timeout(timeout), label)
        except AliyunVisionAPIError as e:
            print(f"   [error] {e}")
            continue
        if urls:
            _add_to_ledger(current_model, count)
            _LAST_USED["video"] = current_model
            return urls[0]

    msg = f"[error] Alibaba Cloud tried all {len(models_to_try)} models ({', '.join(models_to_try)}) and every one failed (video)."
    if raise_on_failure:
        raise AliyunVisionAPIError(msg)
    print(msg)
    return ""


def generate_video_with_model(prompt, **kwargs):
    """Same as generate_video, but returns (url, model actually used), for the same reason as
    generate_image_with_model: with fallback, the model passed in by the caller is not necessarily
    the model that generated the video."""
    url = generate_video(prompt, **kwargs)
    return url, last_used_model("video")


def _models_to_try(output_type, model_name, allow_model_fallback, fallback_models,
                   allow_cross_tier_fallback=False):
    models = [model_name]
    if fallback_models is not None:
        cleaned = [m.strip() for m in fallback_models if m.strip()]
        if not allow_cross_tier_fallback:
            _assert_same_tier(output_type, model_name, cleaned)
        for m in cleaned:
            if m not in models:
                models.append(m)
    elif allow_model_fallback:
        for m in get_fallback_model_names(output_type, model_name):
            if m not in models:
                models.append(m)
    return models


if __name__ == "__main__":
    import sys

    # Make sure the console can print non-ASCII text on Windows terminals with legacy code pages
    sys.stdout.reconfigure(encoding="utf-8")

    if sys.argv[1:2] == ["--usage"]:
        load_dotenv()
        for output_type, tiers in (("image", IMAGE_MODEL_TIERS), ("video", VIDEO_MODEL_TIERS)):
            print(f"\n=== {output_type} ===")
            for tier in TIER_ORDER:
                names = [n for n, _note in tiers[tier]]
                if not names:
                    continue
                print(f"  -- {TIER_LABELS[tier]} --")
                for name in names:
                    print(f"   [usage] {budget_status(name)}")
        sys.exit(0)

    if len(sys.argv) >= 3 and sys.argv[1] in ("image", "video"):
        configure()
        kind = sys.argv[1]
        prompt_arg = " ".join(sys.argv[2:])
        default_model = get_default_model_name(kind)
        print(f"   [model] Alibaba Cloud {kind} primary model: {default_model}"
              f" | fallback: {get_fallback_model_names(kind, default_model)} | {get_base_url()}")
        print(f"   [send] {prompt_arg}\n")
        if kind == "image":
            result = generate_image(prompt_arg)
        else:
            result = generate_video(prompt_arg)
        print("\n" + "=" * 78)
        print(result)
        print("=" * 78)
        sys.exit(0)

    print(__doc__)
