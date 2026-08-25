#!/usr/bin/env python3
"""Compare median aggregate throughput in two benchmark artifacts."""

import argparse
import json
from pathlib import Path


def load_summary(path):
    """Load and validate a benchmark summary."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("format_version") != 1 or not isinstance(artifact.get("summary"), dict):
        raise ValueError(f"unsupported benchmark artifact: {path}")
    return artifact


def main():
    """Print a Markdown comparison table for shared batch sizes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = load_summary(args.baseline)
    candidate = load_summary(args.candidate)
    shared = sorted(set(baseline["summary"]) & set(candidate["summary"]), key=int)
    print("| Batch | Baseline tok/s | Candidate tok/s | Change |")
    print("|---:|---:|---:|---:|")
    for batch in shared:
        before = baseline["summary"][batch]["median_tokens_per_second"]
        after = candidate["summary"][batch]["median_tokens_per_second"]
        if before is None or after is None:
            print(f"| {batch} | n/a | n/a | n/a |")
            continue
        change = ((after / before) - 1.0) * 100.0 if before else float("inf")
        print(f"| {batch} | {before:.2f} | {after:.2f} | {change:+.2f}% |")


if __name__ == "__main__":
    main()

