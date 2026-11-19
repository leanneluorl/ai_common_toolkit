# -*- coding: utf-8 -*-
"""
cogvideox_client.py - Video generation client for a self-hosted CogVideoX server
================================================================================

Calls a CogVideoX inference server that YOU start on a cloud notebook: either
Google Colab (notebooks/cogvideox_server_colab.ipynb; text-to-video and
image-to-video are switched with SERVER_MODE) or Kaggle Notebooks
(notebooks/cogvideox_server_kaggle.ipynb; image + text together, which has a more
predictable free quota than Colab). Both expose the same HTTP API shape, so this
client does not change when you switch platform or mode - only COGVIDEOX_BASE_URL
in .env has to point at the new URL.

The interface mirrors the usual hosted video-API flow (create task -> poll status
-> download mp4; the response-field parsing is deliberately similar), so swapping
between a hosted video API and this self-hosted server touches as little code as
possible.

Things to know up front:
    - The Colab notebook runs one mode at a time (free Colab does not have enough
      memory to load both models), and the mode must match what the server is
      running, otherwise the task fails cleanly (it is not a connection error):
        * server SERVER_MODE="text"  -> call generate_video(prompt, ...) without first_frame.
        * server SERVER_MODE="image" -> call generate_video(prompt, first_frame=..., ...).
    - Image mode (CogVideoX-5b-I2V) only takes a first frame; there is no last_frame
      control (the open-source release does not provide it).
    - first_frame can be a local image path (the client reads the file and sends it
      as a data URI), unlike APIs that only accept a public URL.
    - Every time a Colab runtime restarts, ngrok hands out a new URL, so
      COGVIDEOX_BASE_URL has to be updated by hand.
    - Free GPU quota is not guaranteed and idle sessions get disconnected, so this
      is not suited to long unattended batch jobs; it works best when you are at the
      keyboard with the notebook open.
    - It costs nothing (free GPU + open-source weights), but the quality is not in
      the same class as commercial hosted video models.

Usage:
    import cogvideox_client

    # text-to-video (server must run with SERVER_MODE="text")
    path = cogvideox_client.generate_video("A street after the rain, a car drives slowly past, cinematic camera", seconds=5)

    # image-to-video (server must run with SERVER_MODE="image"); first_frame may be a local path
    path = cogvideox_client.generate_video(
        "Slow push-in, the person turns naturally and walks to the window", seconds=5,
        first_frame=r"C:\\path\\to\\first_frame.png",
    )
    # path is the downloaded local mp4 file (pathlib.Path)

Command line:
    python cogvideox_client.py "prompt text" [--seconds=5] [--out=clip.mp4] [--image=first_frame.png]
    Sends one request, waits for it to finish and saves the file; handy for checking
    that the server URL / key work.

Environment variables (.env; the last notebook cell prints the first two values):
    COGVIDEOX_BASE_URL         required, the ngrok URL printed by the notebook
                               (e.g. https://xxxx.ngrok-free.app, no trailing slash)
    COGVIDEOX_API_KEY          required, must match SERVER_API_KEY set in the notebook
    COGVIDEOX_TIMEOUT_SECONDS  per-HTTP-request timeout in seconds, default 60
    COGVIDEOX_POLL_INTERVAL    polling interval in seconds, default 5
    COGVIDEOX_POLL_TIMEOUT     total time to wait for generation, default 3600 (60 min)
                               (measured in image mode on one GPU at 4-bit: denoising
                               alone takes about 42 minutes for a 5-second clip)
"""

import base64
import mimetypes
import os
import tempfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

_CONFIGURED = False

DEFAULT_TIMEOUT = 60.0
DEFAULT_POLL_INTERVAL = 5.0
# Measured: in image mode (CogVideoX-5b-I2V, one GPU, 4-bit) a 5-second clip needs about
# 42 minutes for the 50 denoising steps alone (50.9 s/step), far slower than the 2B model
# in text mode. 60 minutes leaves some headroom.
DEFAULT_POLL_TIMEOUT = 3600.0
DEFAULT_SECONDS = "5"
DEFAULT_ASPECT_RATIO = "16:9"


class CogVideoXAPIError(Exception):
    """Raised when the CogVideoX server fails (unreachable, HTTP error, timeout, malformed reply...)."""


def configure(env_path=None, force=False):
    """Load .env and make sure the URL and key exist. Safe to call repeatedly
    (the real check only runs once unless force=True)."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if env_path:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
    if not get_base_url():
        raise EnvironmentError(
            "Environment variable COGVIDEOX_BASE_URL not found. Run "
            "notebooks/cogvideox_server_colab.ipynb in Google Colab first and put the ngrok "
            "URL it prints at the end into your .env."
        )
    if not get_api_key():
        raise EnvironmentError(
            "Environment variable COGVIDEOX_API_KEY not found (it must match SERVER_API_KEY set in the notebook server)."
        )
    _CONFIGURED = True


def get_base_url():
    return (os.getenv("COGVIDEOX_BASE_URL") or "").rstrip("/")


def get_api_key():
    return (os.getenv("COGVIDEOX_API_KEY") or "").strip()


def _float_env(name, default):
    raw = os.getenv(name)
    try:
        return float(raw) if raw and raw.strip() else default
    except ValueError:
        return default


def get_timeout(explicit=None):
    return explicit if explicit is not None else _float_env("COGVIDEOX_TIMEOUT_SECONDS", DEFAULT_TIMEOUT)


def get_poll_interval():
    return _float_env("COGVIDEOX_POLL_INTERVAL", DEFAULT_POLL_INTERVAL)


def get_poll_timeout():
    return _float_env("COGVIDEOX_POLL_TIMEOUT", DEFAULT_POLL_TIMEOUT)


def _headers():
    return {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json",
        # ngrok free domains insert an interstitial warning page for requests that lack
        # this header; without it requests would receive that HTML instead of the API's JSON.
        "ngrok-skip-browser-warning": "true",
    }


def _validate_prompt(prompt):
    if not prompt or not str(prompt).strip():
        raise ValueError("Refusing to send an empty prompt to the CogVideoX server; check the upstream source.")


def _as_first_frame_value(first_frame):
    """Turn first_frame into the string the server expects: an existing local file becomes a
    data URI, anything else is sent unchanged (treated as an http(s) URL or a data URI)."""
    if os.path.exists(str(first_frame)):
        mime = mimetypes.guess_type(str(first_frame))[0] or "image/png"
        with open(first_frame, "rb") as f:
            b64data = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64data}"
    return str(first_frame)


def create_video_task(prompt, seconds=None, aspect_ratio=None, seed=None, first_frame=None, timeout=None):
    """Create a video task and return its video_id.

    Without first_frame this is plain text-to-video (mode="text"); with one it is
    image-to-video (mode="image"), which only succeeds when the server runs with
    SERVER_MODE="image" - otherwise the task fails cleanly (not a connection error;
    the server reports the mode mismatch itself).
    """
    _validate_prompt(prompt)
    configure()
    payload = {
        "model": "cogvideox-5b-i2v" if first_frame else "cogvideox-2b",
        "prompt": prompt,
        "mode": "image" if first_frame else "text",
        "seconds": str(seconds or DEFAULT_SECONDS),
        "aspect_ratio": aspect_ratio or DEFAULT_ASPECT_RATIO,
    }
    if first_frame:
        payload["first_frame"] = _as_first_frame_value(first_frame)
    if seed is not None:
        payload["seed"] = int(seed)

    url = f"{get_base_url()}/v1/videos"
    try:
        resp = requests.post(url, headers=_headers(), json=payload, timeout=get_timeout(timeout))
    except requests.RequestException as e:
        raise CogVideoXAPIError(f"Cannot reach the CogVideoX server ({url}): {e}") from e

    if resp.status_code == 401:
        raise CogVideoXAPIError("Authentication failed (HTTP 401): check that COGVIDEOX_API_KEY matches the one set on the server.")
    if resp.status_code >= 400:
        raise CogVideoXAPIError(f"Creating the task failed with HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json() or {}
    video_id = data.get("video_id") or data.get("id") or data.get("task_id")
    if not video_id:
        raise CogVideoXAPIError(f"The create-task response has no video_id: {str(data)[:200]}")
    return video_id


def _extract_video_url(data):
    """Look for the finished video URL in metadata.url first (the shape hosted video APIs
    use), then fall back to a few common field names."""
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("url"):
        return metadata["url"]
    for key in ("url", "video_url", "output_url"):
        if data.get(key):
            return data[key]
    return None


def poll_video(video_id, poll_interval=None, poll_timeout=None, timeout=None, label=""):
    """Poll until completed / failed / timeout. Returns (video_url, detail); the URL is None when unsuccessful."""
    configure()
    interval = poll_interval or get_poll_interval()
    deadline = time.time() + (poll_timeout or get_poll_timeout())
    url = f"{get_base_url()}/v1/videos/{video_id}"
    last_status = ""
    tag = label or video_id

    while time.time() < deadline:
        try:
            resp = requests.get(url, headers=_headers(), timeout=get_timeout(timeout))
            if resp.status_code == 404:
                raise CogVideoXAPIError(f"video_id={video_id} not found (the server may have restarted and lost its task records)")
            if resp.status_code >= 400:
                raise CogVideoXAPIError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json() or {}
            status = str(data.get("status") or "").lower()
            if status != last_status:
                print(f"   [video] {tag} status: {status or 'unknown'} ({data.get('progress', 0)}%)")
                last_status = status

            if status == "completed":
                video_url = _extract_video_url(data)
                if video_url:
                    return video_url, "completed"
                raise CogVideoXAPIError(f"Task completed but no video URL was found: {str(data)[:200]}")
            if status == "failed":
                detail = str(data.get("error") or data.get("message") or "")[:300] or str(data)[:200]
                print(f"   [fail] {tag} generation failed: {detail}")
                return None, f"task_failed: {detail}"
        except CogVideoXAPIError as e:
            print(f"   [warn] {tag} status query failed (will keep waiting): {str(e)[:160]}")
        except requests.RequestException as e:
            print(f"   [warn] {tag} status query connection failed (will keep waiting): {str(e)[:160]}")

        time.sleep(interval)

    return None, f"poll_timeout after {get_poll_timeout():.0f}s (last_status={last_status or 'unknown'})"


def download(url, path):
    """Download the video URL to a local path. Returns True/False; never leaves a partial file behind."""
    path = Path(path)
    try:
        resp = requests.get(url, headers=_headers(), timeout=get_timeout(None), stream=True)
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
        if tmp.stat().st_size < 10_000:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
        return True
    except Exception as e:
        print(f"   [warn] Download failed: {str(e)[:160]}")
        return False


def generate_video(prompt, seconds=None, aspect_ratio=None, seed=None, first_frame=None,
                    out_path=None, label=""):
    """Create the task, poll and download in one go. Returns the downloaded local file path (Path).

    - Without first_frame this is text-to-video; with a local image path (or URL) it is
      image-to-video - the server must be running the matching SERVER_MODE (see the
      module header).
    - out_path defaults to the system temp directory, with the video_id as the file name.
    - A failed or timed-out generation raises CogVideoXAPIError; an incomplete file is never returned.
    """
    video_id = create_video_task(prompt, seconds=seconds, aspect_ratio=aspect_ratio,
                                  seed=seed, first_frame=first_frame)
    tag = label or video_id
    print(f"   [video] {tag} task submitted (video_id={video_id}); waiting for generation...")

    video_url, detail = poll_video(video_id, label=label)
    if not video_url:
        raise CogVideoXAPIError(f"Video generation failed (video_id={video_id}): {detail}")

    out_path = Path(out_path) if out_path else Path(tempfile.gettempdir()) / f"cogvideox_{video_id}.mp4"
    if not download(video_url, out_path):
        raise CogVideoXAPIError(f"The video was generated but the download failed (video_id={video_id}).")

    print(f"   [ok] {tag} done: {out_path}")
    return out_path


if __name__ == "__main__":
    import sys

    # Some Windows consoles default to a legacy code page that cannot print every character.
    sys.stdout.reconfigure(encoding="utf-8")
    configure()

    seconds_arg = DEFAULT_SECONDS
    out_arg = None
    image_arg = None
    prompt_parts = []
    for arg in sys.argv[1:]:
        if arg.startswith("--seconds="):
            seconds_arg = arg.split("=", 1)[1]
        elif arg.startswith("--out="):
            out_arg = arg.split("=", 1)[1]
        elif arg.startswith("--image="):
            image_arg = arg.split("=", 1)[1]
        else:
            prompt_parts.append(arg)

    prompt_arg = " ".join(prompt_parts) or "A cat running across a sunlit meadow"
    print(f"   CogVideoX server: {get_base_url()}")
    mode_note = f" | first frame: {image_arg}" if image_arg else " | text only"
    print(f"   Sending: {prompt_arg} ({seconds_arg}s{mode_note})\n")
    result_path = generate_video(prompt_arg, seconds=seconds_arg, out_path=out_arg,
                                  first_frame=image_arg)
    print(f"\nDone: {result_path}")
