#!/usr/bin/env python3
"""Perform read-only model discovery and a short deterministic completion."""

import argparse
import json
import urllib.request


def fetch_json(request, timeout_seconds):
    """Execute an HTTP request and decode its JSON response."""
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def main():
    """Check the models endpoint and submit one short completion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5057")
    parser.add_argument("--model")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    models = fetch_json(f"{base_url}/v1/models", args.timeout_seconds)
    model_ids = [item["id"] for item in models.get("data", [])]
    if not model_ids:
        raise SystemExit("no models returned by /v1/models")
    model = args.model or model_ids[0]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: READY"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    response = fetch_json(request, args.timeout_seconds)
    if not response.get("choices"):
        raise SystemExit("completion response contained no choices")
    print(json.dumps({"status": "SUCCESS", "model": model, "models": model_ids}))


if __name__ == "__main__":
    main()

