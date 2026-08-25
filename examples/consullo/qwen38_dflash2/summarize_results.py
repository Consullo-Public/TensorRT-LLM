#!/usr/bin/env python3
"""Render a benchmark artifact as a compact Markdown table."""

import argparse
import json
from pathlib import Path


def main():
    """Read a result artifact and print its successful-trial summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    print(f"# {artifact['label']}")
    print("\n| Batch | Trials | Failed | Median tok/s | Mean | Min | Max |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for batch, values in sorted(artifact["summary"].items(), key=lambda item: int(item[0])):
        metrics = [
            "n/a" if values[key] is None else f"{values[key]:.2f}"
            for key in (
                "median_tokens_per_second",
                "mean_tokens_per_second",
                "min_tokens_per_second",
                "max_tokens_per_second",
            )
        ]
        print(
            f"| {batch} | {values['successful_trials']} | {values['failed_trials']} | "
            + " | ".join(metrics)
            + " |"
        )


if __name__ == "__main__":
    main()

