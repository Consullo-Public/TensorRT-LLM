#!/usr/bin/env python3
"""Measure aggregate completion tokens/second for fixed-length decode batches."""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


FORMAT_VERSION = 1
PROMPT = (
    "Write an indefinitely continuing numbered technical analysis of paged KV caches, "
    "speculative decoding, batching, and GPU memory movement. Each item must introduce "
    "a new implementation detail. Do not conclude or summarize."
)
REPETITIVE_PROMPT = (
    "Continue the following exact repeated data pattern without commentary:\n"
    + ("alpha beta gamma delta epsilon zeta eta theta\n" * 128)
)


def request_once(base_url, model, tag, output_tokens, prompt, timeout_seconds):
    """Submit one deterministic fixed-output request and return its measurements."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"Run {tag}. {prompt}"}],
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    usage = payload.get("usage") or {}
    return {
        "seconds": time.perf_counter() - started,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def run_batch(args, batch_size, trial, prompt):
    """Run a synchronized request batch and retain failures as evidence."""
    started = time.perf_counter()
    results = []
    error = None
    try:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [
                executor.submit(
                    request_once,
                    args.base_url,
                    args.model,
                    f"batch-{batch_size}-trial-{trial}-request-{index}",
                    args.output_tokens,
                    prompt,
                    args.timeout_seconds,
                )
                for index in range(batch_size)
            ]
            results = [future.result() for future in futures]
    except (OSError, ValueError, urllib.error.URLError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    wall_seconds = time.perf_counter() - started
    total_tokens = sum(result["completion_tokens"] for result in results)
    return {
        "batch_size": batch_size,
        "trial": trial,
        "status": "FAILED" if error else "SUCCESS",
        "error": error,
        "wall_seconds": wall_seconds,
        "completion_tokens": total_tokens,
        "aggregate_tokens_per_second": total_tokens / wall_seconds if not error else 0.0,
        "requests": results,
    }


def summarize(batches, runs):
    """Summarize successful trials by batch size."""
    summary = {}
    for batch_size in batches:
        values = [
            run["aggregate_tokens_per_second"]
            for run in runs
            if run["batch_size"] == batch_size and run["status"] == "SUCCESS"
        ]
        summary[str(batch_size)] = {
            "successful_trials": len(values),
            "failed_trials": sum(
                run["batch_size"] == batch_size and run["status"] != "SUCCESS"
                for run in runs
            ),
            "median_tokens_per_second": statistics.median(values) if values else None,
            "mean_tokens_per_second": statistics.mean(values) if values else None,
            "min_tokens_per_second": min(values) if values else None,
            "max_tokens_per_second": max(values) if values else None,
        }
    return summary


def parse_batches(value):
    """Parse and validate a comma-separated concurrency list."""
    batches = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not batches or any(item <= 0 for item in batches):
        raise argparse.ArgumentTypeError("batches must contain positive integers")
    return batches


def main():
    """Run warmup and benchmark trials, then write a versioned JSON artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5057")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--batches", type=parse_batches, default=parse_batches("1,2,4,8"))
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--workload", choices=("general", "repetitive"), default="general")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output_tokens <= 0 or args.trials <= 0 or args.timeout_seconds <= 0:
        parser.error("output tokens, trials, and timeout must be positive")

    prompt = REPETITIVE_PROMPT if args.workload == "repetitive" else PROMPT
    warmup = request_once(
        args.base_url, args.model, f"{args.label}-warmup", 64, prompt, args.timeout_seconds
    )
    runs = []
    for batch_size in args.batches:
        for trial in range(1, args.trials + 1):
            result = run_batch(args, batch_size, trial, prompt)
            runs.append(result)
            print(
                f"{args.label} batch={batch_size} trial={trial} "
                f"status={result['status']}: "
                f"{result['aggregate_tokens_per_second']:.2f} tok/s",
                flush=True,
            )
            if result["status"] != "SUCCESS":
                break

    artifact = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "output_tokens_per_request": args.output_tokens,
        "workload": args.workload,
        "warmup": warmup,
        "runs": runs,
        "summary": summarize(args.batches, runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

