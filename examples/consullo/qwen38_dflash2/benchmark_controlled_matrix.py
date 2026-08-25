#!/usr/bin/env python3
"""Benchmark synchronized streaming batches with latency and DFlash metrics."""

import argparse
import json
import math
import statistics
import threading
import time
import urllib.parse
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
METRICS = (
    "trtllm_generation_tokens_total",
    "trtllm_num_requests_completed_total",
    "trtllm_request_success_total",
    "trtllm_spec_decode_num_draft_tokens_total",
    "trtllm_spec_decode_num_accepted_tokens_total",
    "trtllm_spec_decode_drafted_tokens_total",
    "trtllm_spec_decode_accepted_tokens_total",
)


def percentile(values, fraction):
    """Return a linearly interpolated percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_values(values):
    """Return stable descriptive statistics for a numeric sequence."""
    return {
        "median": statistics.median(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "p95": percentile(values, 0.95),
    }


def parse_metric_line(line):
    """Parse one Prometheus sample into its metric, labels, and value."""
    sample, value_text = line.rsplit(None, 1)
    if "{" not in sample:
        return sample, {}, float(value_text)
    metric, raw_labels = sample.split("{", 1)
    labels = {}
    for item in raw_labels[:-1].split(","):
        key, value = item.split("=", 1)
        labels[key] = value.strip('"')
    return metric, labels, float(value_text)


def metric_key(metric, labels):
    """Create a deterministic key while retaining token-position labels."""
    if "token_position" in labels:
        return f"{metric}[{labels['token_position']}]"
    if metric == "trtllm_request_success_total":
        return f"{metric}[{labels.get('finished_reason', 'unknown')}]"
    return metric


def scrape_metrics(base_url, timeout_seconds):
    """Read selected cumulative server metrics."""
    url = f"{base_url.rstrip('/')}/prometheus/metrics"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        text = response.read().decode("utf-8")
    snapshot = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(None, 1)[0]
        if metric in METRICS:
            name, labels, value = parse_metric_line(line)
            key = metric_key(name, labels)
            snapshot[key] = snapshot.get(key, 0.0) + value
    return snapshot


def metric_delta(before, after):
    """Subtract cumulative metric snapshots."""
    keys = set(before) | set(after)
    return {key: after.get(key, 0.0) - before.get(key, 0.0) for key in sorted(keys)}


def speculation_summary(delta):
    """Derive token-weighted DFlash acceptance statistics from counter deltas."""
    drafted = [delta.get(f"trtllm_spec_decode_drafted_tokens_total[{index}]", 0.0)
               for index in range(7)]
    accepted = [delta.get(f"trtllm_spec_decode_accepted_tokens_total[{index}]", 0.0)
                for index in range(7)]
    rounds = drafted[0]
    total_drafted = sum(drafted)
    total_accepted = sum(accepted)
    if not rounds:
        return None
    return {
        "draft_rounds": rounds,
        "drafted_tokens_by_position": drafted,
        "accepted_tokens_by_position": accepted,
        "acceptance_rate": total_accepted / total_drafted if total_drafted else None,
        "acceptance_length": 1.0 + total_accepted / rounds,
        "position_survival_rate": [
            accepted[index] / drafted[index] if drafted[index] else None
            for index in range(7)
        ],
        "scalar_draft_tokens": delta.get("trtllm_spec_decode_num_draft_tokens_total", 0.0),
        "scalar_accepted_tokens": delta.get(
            "trtllm_spec_decode_num_accepted_tokens_total", 0.0
        ),
    }


def request_once(base_url, model, output_tokens, timeout_seconds, barrier):
    """Submit one deterministic streaming request and measure user-visible latency."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    started = time.perf_counter()
    first_token_at = None
    usage = {}
    finish_reason = None
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            usage = payload.get("usage") or usage
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                visible = (delta.get("content") or "") + (
                    delta.get("reasoning_content") or ""
                )
                if visible and first_token_at is None:
                    first_token_at = time.perf_counter()
                finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.perf_counter()
    completion_tokens = usage.get("completion_tokens", 0)
    usage_source = "server"
    if not completion_tokens and finish_reason == "length":
        completion_tokens = output_tokens
        usage_source = "forced_length_contract"
    if completion_tokens != output_tokens:
        raise RuntimeError(
            f"expected {output_tokens} completion tokens, received {completion_tokens}"
        )
    ttft = first_token_at - started if first_token_at is not None else None
    elapsed = ended - started
    tpot = (
        (elapsed - ttft) / (completion_tokens - 1)
        if ttft is not None and completion_tokens > 1 else None
    )
    return {
        "status": "SUCCESS",
        "error": None,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": completion_tokens,
        "usage_source": usage_source,
        "finish_reason": finish_reason,
        "ttft_seconds": ttft,
        "end_to_end_seconds": elapsed,
        "tpot_seconds": tpot,
        "client_tokens_per_second": completion_tokens / elapsed,
    }


def safe_request(*args):
    """Retain request failures in the artifact instead of aborting the matrix."""
    try:
        return request_once(*args)
    except Exception as exception:  # The error type and text are benchmark evidence.
        return {
            "status": "FAILED",
            "error": f"{type(exception).__name__}: {exception}",
            "completion_tokens": 0,
        }


def run_batch(args, batch_size, trial, output_tokens, collect_metrics=True):
    """Run one synchronized batch and calculate client and server measurements."""
    before = scrape_metrics(args.base_url, args.timeout_seconds) if collect_metrics else {}
    barrier = threading.Barrier(batch_size + 1)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [
            executor.submit(
                safe_request, args.base_url, args.model, output_tokens,
                args.timeout_seconds, barrier
            )
            for _ in range(batch_size)
        ]
        barrier.wait()
        started = time.perf_counter()
        requests = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    if collect_metrics:
        time.sleep(0.25)
    after = scrape_metrics(args.base_url, args.timeout_seconds) if collect_metrics else {}
    delta = metric_delta(before, after)
    successful = [item for item in requests if item["status"] == "SUCCESS"]
    total_tokens = sum(item["completion_tokens"] for item in successful)
    result = {
        "batch_size": batch_size,
        "trial": trial,
        "status": "SUCCESS" if len(successful) == batch_size else "FAILED",
        "wall_seconds": wall_seconds,
        "completion_tokens": total_tokens,
        "aggregate_tokens_per_second": total_tokens / wall_seconds,
        "request_summary": {},
        "requests": requests,
        "metric_delta": delta,
        "speculation": speculation_summary(delta),
    }
    for field in ("ttft_seconds", "end_to_end_seconds", "tpot_seconds",
                  "client_tokens_per_second"):
        result["request_summary"][field] = summarize_values(
            [item[field] for item in successful if item.get(field) is not None]
        )
    return result


def summarize(batches, runs):
    """Aggregate successful timed trials for each request shape."""
    summary = {}
    for batch_size in batches:
        selected = [run for run in runs if run["batch_size"] == batch_size
                    and run["status"] == "SUCCESS"]
        item = {
            "successful_trials": len(selected),
            "failed_trials": sum(run["batch_size"] == batch_size and
                                  run["status"] != "SUCCESS" for run in runs),
            "aggregate_tokens_per_second": summarize_values(
                [run["aggregate_tokens_per_second"] for run in selected]
            ),
        }
        for field in ("ttft_seconds", "end_to_end_seconds", "tpot_seconds",
                      "client_tokens_per_second"):
            item[field] = summarize_values([
                request[field] for run in selected for request in run["requests"]
                if request.get(field) is not None
            ])
        speculations = [run["speculation"] for run in selected if run["speculation"]]
        if speculations:
            item["acceptance_rate"] = summarize_values(
                [spec["acceptance_rate"] for spec in speculations]
            )
            item["acceptance_length"] = summarize_values(
                [spec["acceptance_length"] for spec in speculations]
            )
            item["position_survival_rate"] = [
                statistics.mean(spec["position_survival_rate"][index]
                                for spec in speculations)
                for index in range(7)
            ]
        summary[str(batch_size)] = item
    return summary


def parse_batches(value):
    """Parse and validate a comma-separated concurrency list."""
    batches = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not batches or any(item <= 0 for item in batches):
        raise argparse.ArgumentTypeError("batches must contain positive integers")
    return batches


def public_endpoint(value):
    """Remove credentials, query parameters, and fragments from a recorded URL."""
    parsed = urllib.parse.urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def write_artifact(args, warmups, runs):
    """Checkpoint all completed work so a server failure cannot erase evidence."""
    artifact = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "base_url": public_endpoint(args.base_url),
        "model": args.model,
        "prompt": PROMPT,
        "output_tokens_per_request": args.output_tokens,
        "warmup_tokens_per_request": args.warmup_tokens,
        "trials": args.trials,
        "warmups": warmups,
        "runs": runs,
        "summary": summarize(args.batches, runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main():
    """Run shape-specific warmups and controlled timed trials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:5057")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--batches", type=parse_batches,
                        default=parse_batches("1,8,12,15"))
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.output_tokens, args.warmup_tokens, args.trials,
           args.timeout_seconds) <= 0:
        parser.error("token counts, trials, and timeout must be positive")

    warmups = []
    runs = []
    for batch_size in args.batches:
        warmup = run_batch(args, batch_size, 0, args.warmup_tokens, False)
        warmups.append(warmup)
        write_artifact(args, warmups, runs)
        print(f"{args.label} batch={batch_size} warmup={warmup['status']}", flush=True)
        if warmup["status"] != "SUCCESS":
            continue
        for trial in range(1, args.trials + 1):
            result = run_batch(args, batch_size, trial, args.output_tokens)
            runs.append(result)
            artifact = write_artifact(args, warmups, runs)
            median_ttft = result["request_summary"]["ttft_seconds"]["median"]
            ttft_text = f"{median_ttft:.3f}s" if median_ttft is not None else "n/a"
            print(
                f"{args.label} batch={batch_size} trial={trial} "
                f"status={result['status']} "
                f"aggregate={result['aggregate_tokens_per_second']:.2f} tok/s "
                f"ttft={ttft_text}",
                flush=True,
            )

    artifact = write_artifact(args, warmups, runs)
    print(json.dumps(artifact["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
