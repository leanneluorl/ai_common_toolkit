# ai_common_toolkit

A set of small, dependency-light Python clients for working with several AI APIs through one consistent style: text, vision, image and video generation, plus a quota-aware scheduler for the Gemini free tier.

It is a portfolio / code-review sample. The goal is to show how I integrate third-party AI services in a way that survives real-world conditions: rate limits, daily quotas, retired models, flaky networks and slow asynchronous jobs.

> License: [MIT](LICENSE).

## What is inside

| Module | Provider | Highlights |
| --- | --- | --- |
| `ai_client.py` | Google Gemini | Quota-aware scheduling per (quota pool x model): RPM / TPM / RPD buckets, limits learned from 429 responses, load distribution across keys from different Cloud projects, model fallback chain, Pacific-time daily reset, 403 deny-streak handling, 503 requeue, heartbeat for long calls, usage log (JSON + CSV), `--diagnose` command, model registry |
| `groq_client.py` | Groq | OpenAI-compatible chat + vision, `Retry-After` aware, skips straight to a fallback model when the wait is too long, distinguishes "quota" from "request too large" |
| `agnes_client.py` | Agnes | Chat client with a hard allow-list so a paid model can never be called by accident |
| `agnes_vision_client.py` | Agnes | Image and video generation with asynchronous polling and per-type fallback models |
| `aliyun_vision_client.py` | Alibaba Cloud Model Studio | Three image API generations behind one function, async video tasks, local per-model quota ledger, model catalogue in `aliyun_vision_models.json` |
| `qwen_client.py` | Qwen / DashScope | Streaming, optional "thinking" mode with token budget, tiered model catalogue, local token ledger |
| `cogvideox_client.py` + `notebooks/` | Self-hosted CogVideoX | Client for an open-source video model that you serve yourself from a free Colab or Kaggle GPU through ngrok; the server notebooks mimic the Agnes response shape |
| `ai_router.py` | Gemini or Groq | One call, one exception hierarchy, provider chosen by env var or at runtime |
| `trend_search.py` | Google, YouTube, Reddit | Search-grounded generation, YouTube Data API with a daily budget, Reddit and Google Trends digests |

## Design notes

**Quota-aware, not retry-and-pray.** Gemini quota is counted per Google Cloud project and model, not per API key. `ai_client.py` therefore models every quota pool explicitly and sends each request to the pool that can serve it soonest, rather than blindly rotating keys. If a freshly selected key is rate limited immediately, it concludes the keys share one allocation and switches to a shared-pool mode.

**Errors that mean different things are different exceptions.** "Today's quota is gone" (stop), "cannot squeeze in right now" (a single request is too large), and "this one item failed" (carry on with the rest) call for different reactions. `ai_router.py` maps both providers onto `AIQuotaExhaustedError`, `AIRequestTooLargeError` and `AIRequestFailedError`.

**The SDK's hidden retries are switched off.** The Gemini SDK retries internally and its timeout covers the entire retry loop, so a single call can silently hang for minutes. The client disables that and applies a 180 s timeout, so that failures surface immediately and the scheduler decides what to do next.

**Safe defaults.** Empty prompts never reach an API. Paid models are never part of an automatic fallback chain. Usage and ledger files live under `_log/` and are git-ignored.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then fill in the keys you have
python examples/quickstart.py
```

```python
import ai_router

text, used_model = ai_router.generate_json_text("Return {\"ok\": true}", label="demo")
```

```python
import ai_client

chat = ai_client.start_chat()
print(ai_client.send_chat_message(chat, "First question"))
print(ai_client.send_chat_message(chat, "Follow-up, with memory"))
```

Command line helpers for the Gemini client:

```bash
python ai_client.py              # usage statistics and model list (no API call)
python ai_client.py --diagnose   # empirically test whether several keys share one quota
```

## Tests

The tests are fully offline: HTTP and SDK calls are replaced with mocks.

```bash
pytest
```

## Repository layout

```
ai_client.py  groq_client.py  agnes_client.py  agnes_vision_client.py
aliyun_vision_client.py  qwen_client.py  cogvideox_client.py
ai_router.py  trend_search.py
notebooks/    CogVideoX inference servers for Colab and Kaggle
examples/     runnable usage examples
tests/        pytest suite (mock based)
```

## Notes

- Model names, limits and free-tier policies change often. Treat the defaults in the code as a snapshot, and override them through environment variables (see `.env.example`).
- API keys are read from the environment / `.env` only. Nothing is hard-coded.
