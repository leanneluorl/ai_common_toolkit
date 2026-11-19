# -*- coding: utf-8 -*-
"""
agnes_vision_client.py - Client for the Agnes AI image / video generation APIs
==============================================================================
Calls Agnes' image generation endpoint (/v1/images/generations) and video generation
endpoint (/v1/videos: create an asynchronous task, then poll for the result).

This is a general-purpose wrapper: endpoints, payload shapes and polling fields follow
the vendor's public API documentation. It contains no application-specific logic;
callers keep their own business rules (prompt prefixes, duration planning, concurrency
scheduling) and use this module only to talk to the API.

The interface deliberately mirrors aliyun_vision_client.py: both modules expose
generate_image() / generate_video() with the same parameter shapes, so swapping one
provider for the other is, in principle, just a different import.

Cost guard: only models that are currently free of charge are allowed.
Image: agnes-image-2.1-flash (primary) / agnes-image-2.0-flash (fallback).
Video: agnes-video-2.5-flash (the only model confirmed free; no fallback).
This is the same design as FREE_MODELS in agnes_client.py: generate_image() and
generate_video() reject any model that is not on the allow-list without sending an HTTP
request. The advertised "current price $0" is a promotional price and may change at any
time; if a model stops being free, empty or edit the allow-lists below and the functions
will refuse to call anything until you have re-checked the pricing.

The model lists are independent from aliyun_vision_client.py and are never mixed with it
(Agnes has few models and no tier concept, so they are plain constants).

Usage 1 - images (text-to-image; pass image or images for image-to-image / multi-image
composition, handled by the same endpoint):
    import agnes_vision_client as agv
    url = agv.generate_image("A floating city at sunset, cinematic", ratio="16:9")
    url = agv.generate_image("Turn the background into a cyberpunk night scene",
                             image="https://.../input.png")

Usage 2 - video (text-to-video / first-and-last-frame / reference-based). Agnes has no
pure single-image image-to-video mode: when image is passed without first_frame /
last_frame it is treated as a keyframe request with only a first frame, which is close to
image-to-video but not identical to Aliyun's i2v models, so evaluate the output yourself:
    url = agv.generate_video("A cat playing piano in the rain, slow zoom in")
    url = agv.generate_video("The character turns and walks to the window",
                             first_frame="https://.../a.png", last_frame="https://.../b.png")

Usage 3 - find out which model actually produced a result (with fallback it is not
necessarily the model you asked for; the interface matches aliyun_vision_client):
    url, used_model = agv.generate_image_with_model("...")
    url, used_model = agv.generate_video_with_model("...")

Command line:
    python agnes_vision_client.py image "prompt text"     generate one image and print its URL
    python agnes_vision_client.py video "prompt text"     generate one video and print its URL

Environment variables (.env, shared with agnes_client.py):
    AGNES_API_KEY                 required (the same key as agnes_client.py)
    AGNES_BASE_URL                API base URL, default https://apihub.agnes-ai.com/v1
    AGNES_IMAGE_MODEL             primary image model, default agnes-image-2.1-flash
    AGNES_IMAGE_FALLBACK_MODELS   comma-separated image fallbacks, default agnes-image-2.0-flash
    AGNES_VIDEO_MODEL             primary video model, default agnes-video-2.5-flash
    AGNES_VIDEO_FALLBACK_MODELS   comma-separated video fallbacks, default none (only one free model)
    AGNES_VISION_TIMEOUT_SECONDS  per-request HTTP timeout in seconds, default 300
    AGNES_VIDEO_POLL_TIMEOUT      video polling timeout in seconds, default 900 (clips of up to
                                  12 seconds; queueing plus generation can take minutes)
    AGNES_VIDEO_POLL_INTERVAL     video polling interval in seconds, default 10
    AGNES_VIDEO_QUERY_ENDPOINT    full URL of the video query endpoint, default
                                  https://apihub.agnes-ai.com/agnesapi (per the vendor docs
                                  the query is not under /v1, so do not change only the host)
"""

import os
import time

import requests
from dotenv import load_dotenv

_CONFIGURED = False
# The model that actually served the most recent successful request. With fallback the
# caller needs this to know which model produced a given image / video (useful when
# tagging output files or comparing model quality).
_LAST_USED = {"image": "", "video": ""}


def last_used_model(output_type):
    """The model that actually served the most recent successful generate_image /
    generate_video call (may be a fallback model rather than the one requested).
    Empty string if nothing has succeeded yet. Matches aliyun_vision_client."""
    return _LAST_USED[output_type]


# ============================================================
# Free-model allow-lists: generate_image() / generate_video() may only call models
# listed here. Checked against the vendor's documentation: Agnes Image 2.1/2.0 Flash and
# Agnes Video 2.5 Flash are currently priced at $0 (a promotional price; the standard
# list price is not zero, so watch the vendor's announcements).
# ============================================================
FREE_IMAGE_MODELS = ("agnes-image-2.1-flash", "agnes-image-2.0-flash")
FREE_VIDEO_MODELS = ("agnes-video-2.5-flash",)

DEFAULT_IMAGE_MODEL = "agnes-image-2.1-flash"
DEFAULT_IMAGE_FALLBACK_MODELS = ["agnes-image-2.0-flash"]
DEFAULT_VIDEO_MODEL = "agnes-video-2.5-flash"
DEFAULT_VIDEO_FALLBACK_MODELS = []
DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
DEFAULT_TIMEOUT = 300.0
DEFAULT_VIDEO_POLL_TIMEOUT = 900.0
DEFAULT_VIDEO_POLL_INTERVAL = 10.0
DEFAULT_IMAGE_SIZE = "1024x768"
DEFAULT_VIDEO_SIZE = "720P"
DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_VIDEO_SECONDS = "5"


class AgnesVisionAPIError(Exception):
    """Raised when the Agnes image / video API fails (HTTP 4xx/5xx, task failed,
    retries exhausted, malformed reply...)."""
    pass


class AgnesPaidModelBlockedError(Exception):
    """Raised when the caller asks for a model that is not on the FREE_IMAGE_MODELS /
    FREE_VIDEO_MODELS allow-list. A deliberate safety valve so that a paid model can
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


def get_video_query_url():
    """Video queries go to a different path (not under /v1). The structure follows the
    vendor docs; only the full URL is configurable."""
    return (os.getenv("AGNES_VIDEO_QUERY_ENDPOINT") or "https://apihub.agnes-ai.com/agnesapi").rstrip("/")


def get_default_image_model():
    return (os.getenv("AGNES_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL).strip()


def get_default_video_model():
    return (os.getenv("AGNES_VIDEO_MODEL") or DEFAULT_VIDEO_MODEL).strip()


def get_default_model_name(output_type):
    """Matches aliyun_vision_client.get_default_model_name(output_type)."""
    return get_default_image_model() if output_type == "image" else get_default_video_model()


def get_image_fallback_models(primary=None):
    raw = os.getenv("AGNES_IMAGE_FALLBACK_MODELS")
    names = [n.strip() for n in raw.split(",")] if raw is not None else list(DEFAULT_IMAGE_FALLBACK_MODELS)
    primary = (primary or get_default_image_model()).strip()
    return [n for n in names if n and n != primary]


def get_video_fallback_models(primary=None):
    raw = os.getenv("AGNES_VIDEO_FALLBACK_MODELS")
    names = [n.strip() for n in raw.split(",")] if raw is not None else list(DEFAULT_VIDEO_FALLBACK_MODELS)
    primary = (primary or get_default_video_model()).strip()
    return [n for n in names if n and n != primary]


def get_fallback_model_names(output_type, primary=None):
    """Matches aliyun_vision_client.get_fallback_model_names(output_type, primary)."""
    if output_type == "image":
        return get_image_fallback_models(primary)
    return get_video_fallback_models(primary)


def _env_float(name, default):
    raw = os.getenv(name)
    try:
        return float(raw) if raw and str(raw).strip() else default
    except ValueError:
        return default


def get_timeout(explicit=None):
    if explicit is not None:
        return explicit
    return _env_float("AGNES_VISION_TIMEOUT_SECONDS", DEFAULT_TIMEOUT)


def get_video_poll_timeout():
    return _env_float("AGNES_VIDEO_POLL_TIMEOUT", DEFAULT_VIDEO_POLL_TIMEOUT)


def get_video_poll_interval():
    return _env_float("AGNES_VIDEO_POLL_INTERVAL", DEFAULT_VIDEO_POLL_INTERVAL)


def _headers():
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
    }


def _validate_prompt(prompt):
    if not prompt or not str(prompt).strip():
        raise ValueError("Refusing to send an empty prompt to the Agnes API; check the upstream prompt source.")


def _assert_free_model(output_type, model_name):
    whitelist = FREE_IMAGE_MODELS if output_type == "image" else FREE_VIDEO_MODELS
    if model_name not in whitelist:
        raise AgnesPaidModelBlockedError(
            f"Refusing to call '{model_name}': it is not on the free-model allow-list ({output_type}) {whitelist}.\n"
            f"   This module is intentionally locked to free models so a paid model is never hit by accident.\n"
            f"   If you have checked Agnes' current pricing and the model is free, edit "
            f"{'FREE_IMAGE_MODELS' if output_type == 'image' else 'FREE_VIDEO_MODELS'} in this file."
        )


def _models_to_try(output_type, model_name, allow_model_fallback, fallback_models):
    models = [model_name]
    if fallback_models is not None:
        for m in fallback_models:
            m = m.strip()
            _assert_free_model(output_type, m)
            if m and m not in models:
                models.append(m)
    elif allow_model_fallback:
        for m in get_fallback_model_names(output_type, model_name):
            _assert_free_model(output_type, m)
            if m not in models:
                models.append(m)
    return models


# ============================================================
# Images: POST /v1/images/generations (text-to-image, image-to-image and multi-image
# composition all use the same endpoint)
# ============================================================
def _image_from_payload(item):
    """Prefer the url; if there is none, wrap b64_json in a data URI (the same format
    used for input reference images, so callers can treat the result as a string
    whether it is a URL or base64, and save it either way)."""
    if not isinstance(item, dict):
        return None
    if item.get("url"):
        return item["url"]
    if item.get("b64_json"):
        return f"data:image/png;base64,{item['b64_json']}"
    return None


def _generate_image_once(model_name, prompt, image_refs, size, ratio, return_base64,
                         timeout, max_retries, label):
    extra_body = {"response_format": "b64_json" if return_base64 else "url"}
    if image_refs:
        extra_body["image"] = image_refs
    payload = {"model": model_name, "prompt": prompt, "size": size, "extra_body": extra_body}
    if ratio:
        payload["ratio"] = ratio

    url = f"{get_base_url()}/images/generations"
    delay = 5.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=get_timeout(timeout))
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Agnes {model_name} connection error ({type(e).__name__}); retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = AgnesVisionAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
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
            raise AgnesVisionAPIError(f"Agnes image API error HTTP {resp.status_code}: {resp.text[:300]}")

        data = (resp.json() or {}).get("data") or []
        if not data:
            last_error = AgnesVisionAPIError("the response data[] is empty")
            print(f"   [warn] Agnes {model_name} returned no data; retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        image = _image_from_payload(data[0])
        if not image:
            raise AgnesVisionAPIError(f"The Agnes image response has neither url nor b64_json: {str(data[0])[:200]}")

        label_note = f" | {label}" if label else ""
        print(f"   [ok] Agnes {model_name} | generated 1 image | est. cost $0{label_note}")
        return image, True

    print(f"   [fail] Agnes {model_name} still failing after {max_retries} attempts: {last_error}")
    return None, False


def generate_image(prompt, model=None, image=None, images=None, size=None, ratio=None, n=1,
                   return_base64=False, timeout=None, max_retries=3, raise_on_failure=True,
                   label=None, allow_model_fallback=True, fallback_models=None):
    """
    Call the Agnes image generation API (text-to-image; pass image or images for
    image-to-image / multi-image composition).

    - prompt: the text description
    - image / images: optional input reference images (publicly reachable URLs or data
      URIs). images takes several, image takes one (use either; if both are given,
      images wins)
    - model: optional, default agnes-image-2.1-flash; must be on the FREE_IMAGE_MODELS allow-list
    - size: optional, default "1024x768" (the "1K" / "2K" / "3K" / "4K" tiers also work)
    - ratio: optional aspect ratio to combine with a tier-style size, e.g. "16:9"
    - n: how many images to generate, default 1 (Agnes produces one image per call, so
      n>1 repeats the call n times)
    - returns: an image URL string (or a data URI when return_base64=True); with n>1 it
      returns the first one; use generate_image_urls() to get all of them
    - 429 / 5xx are retried max_retries times; on a 404 (model missing or retired) or
      once retries are exhausted, fallback_models are tried (default:
      get_image_fallback_models(), the other model on the Agnes allow-list).
    """
    urls = generate_image_urls(prompt, model=model, image=image, images=images, size=size, ratio=ratio,
                               n=n, return_base64=return_base64, timeout=timeout, max_retries=max_retries,
                               raise_on_failure=raise_on_failure, label=label,
                               allow_model_fallback=allow_model_fallback, fallback_models=fallback_models)
    return urls[0] if urls else ""


def generate_image_with_model(prompt, **kwargs):
    """Same as generate_image but returns (url, model actually used). The interface
    matches aliyun_vision_client, so tagging outputs with "which model made this" works
    the same way with either module."""
    url = generate_image(prompt, **kwargs)
    return url, last_used_model("image")


def generate_image_urls(prompt, model=None, image=None, images=None, size=None, ratio=None, n=1,
                        return_base64=False, timeout=None, max_retries=3, raise_on_failure=True,
                        label=None, allow_model_fallback=True, fallback_models=None):
    """Same as generate_image but returns the full list of URLs / data URIs (with n>1
    the call is repeated n times to collect enough images)."""
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_image_model()).strip()
    _assert_free_model("image", model_name)
    models_to_try = _models_to_try("image", model_name, allow_model_fallback, fallback_models)
    image_refs = images or ([image] if image else None)
    size = size or DEFAULT_IMAGE_SIZE

    results = []
    for _ in range(max(n, 1)):
        for current_model in models_to_try:
            if current_model != models_to_try[0]:
                print(f"   [switch] Using fallback model: {current_model}")
            img, ok = _generate_image_once(current_model, prompt, image_refs, size, ratio,
                                           return_base64, timeout, max_retries, label)
            if ok:
                results.append(img)
                _LAST_USED["image"] = current_model
                break
        else:
            msg = f"Agnes: all {len(models_to_try)} model(s) ({', '.join(models_to_try)}) were tried and failed (image)."
            if raise_on_failure:
                raise AgnesVisionAPIError(msg)
            print(msg)
            return results

    return results


# ============================================================
# Video: POST /v1/videos creates a task, GET /agnesapi?video_id=...&model_name=... polls it
# ============================================================
def _extract_video_url(data):
    """The finished video URL is in metadata.url; different API versions shape the
    response differently, so look in several places (in the order verified in practice)."""
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("url"):
        return metadata["url"]
    for key in ("url", "video_url", "output_url"):
        if data.get(key):
            return data[key]
    for key in ("data", "output", "result"):
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            found = _extract_video_url(value[0])
            if found:
                return found
        if isinstance(value, dict):
            found = _extract_video_url(value)
            if found:
                return found
    return None


def _create_video_task(model_name, payload, max_retries):
    url = f"{get_base_url()}/videos"
    delay = 5.0
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=_headers(), json=payload, timeout=get_timeout(None))
        except requests.RequestException as e:
            last_error = e
            print(f"   [warn] Agnes {model_name} connection error while creating the video task ({type(e).__name__}); retrying in {delay:.0f}s ({attempt}/{max_retries})")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_error = AgnesVisionAPIError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            retry_after = resp.headers.get("retry-after")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            print(f"   [wait] Agnes {model_name} returned HTTP {resp.status_code} (possibly the 1-task-per-minute rate limit); "
                  f"retrying in {wait:.0f}s ({attempt}/{max_retries})")
            time.sleep(wait)
            delay = min(delay * 2, 60)
            continue

        if resp.status_code == 404:
            print(f"   [skip] Agnes {model_name} returned 404 (model missing or retired); switching to the fallback model.")
            return None, False

        if resp.status_code >= 400:
            raise AgnesVisionAPIError(f"Agnes video API error HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json() or {}
        video_id = data.get("video_id") or data.get("id") or data.get("task_id")
        if not video_id:
            raise AgnesVisionAPIError(f"The create-video-task response has no video_id: {str(data)[:200]}")
        return video_id, True

    print(f"   [fail] Agnes {model_name} still failing to create a video task after {max_retries} attempts: {last_error}")
    return None, False


def _poll_video(video_id, model_name, label):
    url = get_video_query_url()
    deadline = time.monotonic() + get_video_poll_timeout()
    interval = get_video_poll_interval()
    last_status = ""

    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, headers={"Authorization": f"Bearer {get_api_key()}"},
                                params={"video_id": video_id, "model_name": model_name}, timeout=30)
            if resp.status_code == 429:
                time.sleep(interval)
                continue
            if resp.status_code >= 400:
                raise AgnesVisionAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json() or {}
        except requests.RequestException as e:
            print(f"   [warn] Agnes {model_name} polling connection error ({type(e).__name__}); retrying in {interval:.0f}s")
            time.sleep(interval)
            continue

        status = str(data.get("status") or "").lower()
        if status != last_status:
            label_note = f" | {label}" if label else ""
            print(f"   [wait] Agnes {model_name} video status: {status or 'unknown'} ({data.get('progress', 0)}%){label_note}")
            last_status = status

        if status == "completed":
            video_url = _extract_video_url(data)
            if video_url:
                print(f"   [ok] Agnes {model_name} | video finished | est. cost $0")
                return video_url
            raise AgnesVisionAPIError(f"The task completed but no video URL was found: {str(data)[:200]}")
        if status == "failed":
            detail = str(data.get("error") or data.get("message") or data.get("detail") or data)[:300]
            raise AgnesVisionAPIError(f"Agnes {model_name} video generation failed: {detail}")

        time.sleep(interval)

    raise AgnesVisionAPIError(f"Agnes {model_name} video polling timed out ({get_video_poll_timeout():.0f}s).")


def generate_video(prompt, model=None, image=None, first_frame=None, last_frame=None,
                   reference_images=None, reference_audios=None, duration=None, size=None, ratio=None,
                   seed=None, timeout=None, max_retries=3, raise_on_failure=True, label=None,
                   allow_model_fallback=True, fallback_models=None):
    """
    Call the Agnes video generation API. The mode is chosen from the arguments given:
    - none of them: text-to-video (mode="text")
    - first_frame / last_frame: first-and-last-frame video (mode="keyframe"; pass at least one)
    - image without first_frame / last_frame: treated as a keyframe request with only a
      first frame (close to image-to-video; Agnes has no separate i2v mode, so the result
      is not identical to Aliyun's i2v models - evaluate it yourself)
    - reference_images / reference_audios: reference-based video (mode="reference")

    - model: optional, default agnes-video-2.5-flash; must be on the FREE_VIDEO_MODELS allow-list
    - duration: optional, clip length "4"~"12" seconds (string or number), default "5"
    - size: optional; Flash models are fixed at "720P" and other values are rejected by the API
    - ratio: optional output aspect ratio, default "16:9" (see the "video size and aspect
      ratio" table in the Agnes Video docs)
    - returns: the video URL string
    - 429 (possibly the 1-task-per-minute rate limit) / 5xx are retried max_retries times;
      on a 404 or once retries are exhausted, fallback_models are tried (default:
      get_video_fallback_models(); there is only one free model, so no fallback by default).
    """
    _validate_prompt(prompt)
    configure()
    model_name = (model or get_default_video_model()).strip()
    _assert_free_model("video", model_name)
    models_to_try = _models_to_try("video", model_name, allow_model_fallback, fallback_models)

    if first_frame or last_frame:
        mode = "keyframe"
    elif reference_images or reference_audios:
        mode = "reference"
    elif image:
        mode = "keyframe"
        first_frame = image
    else:
        mode = "text"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "seconds": str(duration) if duration is not None else DEFAULT_VIDEO_SECONDS,
        "mode": mode,
        "size": size or DEFAULT_VIDEO_SIZE,
        "aspect_ratio": ratio or DEFAULT_VIDEO_ASPECT_RATIO,
    }
    if seed is not None:
        payload["seed"] = seed
    if mode == "keyframe":
        if first_frame:
            payload["first_frame"] = first_frame
        if last_frame:
            payload["last_frame"] = last_frame
    elif mode == "reference":
        if reference_images:
            payload["images"] = reference_images
        if reference_audios:
            payload["audios"] = reference_audios

    for current_model in models_to_try:
        if current_model != models_to_try[0]:
            print(f"   [switch] Using fallback model: {current_model}")
        payload["model"] = current_model
        video_id, ok = _create_video_task(current_model, payload, max_retries)
        if not ok:
            continue
        try:
            video_url = _poll_video(video_id, current_model, label)
        except AgnesVisionAPIError as e:
            print(f"   [fail] {e}")
            continue
        _LAST_USED["video"] = current_model
        return video_url

    msg = f"Agnes: all {len(models_to_try)} model(s) ({', '.join(models_to_try)}) were tried and failed (video)."
    if raise_on_failure:
        raise AgnesVisionAPIError(msg)
    print(msg)
    return ""


def generate_video_with_model(prompt, **kwargs):
    """Same as generate_video but returns (url, model actually used). The interface
    matches aliyun_vision_client."""
    url = generate_video(prompt, **kwargs)
    return url, last_used_model("video")


if __name__ == "__main__":
    import sys

    # Some Windows consoles default to a legacy code page that cannot print every character.
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) >= 3 and sys.argv[1] in ("image", "video"):
        configure()
        kind = sys.argv[1]
        prompt_arg = " ".join(sys.argv[2:])
        default_model = get_default_model_name(kind)
        print(f"   Agnes {kind} primary model: {default_model}"
              f" | fallback: {get_fallback_model_names(kind, default_model)} | {get_base_url()}")
        print(f"   Sending: {prompt_arg}\n")
        result = generate_image(prompt_arg) if kind == "image" else generate_video(prompt_arg)
        print("\n" + "=" * 78)
        print(result)
        print("=" * 78)
        sys.exit(0)

    print(__doc__)
