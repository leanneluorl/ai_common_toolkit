"""Image and video generation through the Agnes and Aliyun clients (requires real keys).

    python examples/generate_media.py agnes  "a lighthouse at dusk"
    python examples/generate_media.py aliyun "a lighthouse at dusk"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("agnes", "aliyun"):
        print(__doc__)
        return 2
    provider, prompt = sys.argv[1], " ".join(sys.argv[2:])
    if provider == "agnes":
        import agnes_vision_client as client
    else:
        import aliyun_vision_client as client

    urls = client.generate_image_urls(prompt)
    print("image urls:", urls)
    print("model used:", client.last_used_model("image"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
