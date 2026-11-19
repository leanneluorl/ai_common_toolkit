"""Minimal tour of the toolkit. Copy .env.example to .env and fill in the keys you have.

    python examples/quickstart.py            # uses AI_PROVIDER (default: gemini)
    AI_PROVIDER=groq python examples/quickstart.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ai_router  # noqa: E402


def main():
    provider = ai_router.get_provider()
    print(f"provider: {provider}")
    try:
        text, used_model = ai_router.generate_json_text(
            'Return a JSON object like {"greeting": "..."} with a short greeting.',
            label="quickstart",
        )
    except ai_router.AIQuotaExhaustedError as exc:
        # Gemini and Groq both surface "quota exhausted / rate limited" as this one type.
        print(f"quota exhausted: {exc}")
        return 1
    except ai_router.AIRequestTooLargeError as exc:
        print(f"request too large: {exc}")
        return 1
    print(f"model used: {used_model}")
    print(json.loads(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
