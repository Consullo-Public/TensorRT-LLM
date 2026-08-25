#!/usr/bin/env python3
"""Learn conservative DFlash policies from repeated controlled artifacts."""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def load_candidates(paths):
    """Load qualifying batch summaries while retaining their provenance."""
    candidates = []
    for path in paths:
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "summary" not in artifact or "expected_prompt_tokens" not in artifact:
            continue
        policy = artifact.get("policy") or {}
        for batch_text, summary in artifact["summary"].items():
            candidates.append({
                "artifact": str(path),
                "label": artifact.get("label"),
                "prompt_tokens": artifact.get("expected_prompt_tokens"),
                "output_tokens": artifact.get("output_tokens_per_request"),
                "batch_size": int(batch_text),
                "draft_length": policy.get("draft_length"),
                "selector_batches": policy.get("selector_batches") or [],
                "successful_trials": summary.get("successful_trials", 0),
                "failed_trials": summary.get("failed_trials", 0),
                "aggregate_tokens_per_second": (
                    summary.get("aggregate_tokens_per_second") or {}
                ).get("median"),
                "p95_ttft_seconds": (summary.get("ttft_seconds") or {}).get("p95"),
                "prefill_gpu_forward_ms": (
                    (summary.get("server_timing") or {})
                    .get("prefill_gpu_forward_ms") or {}
                ).get("median"),
                "acceptance_length": (
                    summary.get("acceptance_length") or {}
                ).get("median"),
                "acceptance_rate": (
                    summary.get("acceptance_rate") or {}
                ).get("median"),
                "position_survival_rate": summary.get("position_survival_rate"),
            })
    return [item for item in candidates
            if item["prompt_tokens"] is not None
            and item["output_tokens"] is not None
            and item["aggregate_tokens_per_second"] is not None]


def effect(candidate, baseline):
    """Return candidate throughput change relative to a baseline."""
    return (candidate["aggregate_tokens_per_second"] /
            baseline["aggregate_tokens_per_second"] - 1.0)


def eligible(candidate, minimum_trials):
    """Require repeated, failure-free trials before changing policy."""
    return (candidate["successful_trials"] >= minimum_trials
            and candidate["failed_trials"] == 0)


def compact(candidate):
    """Return the policy and measured fields needed by a router."""
    if candidate is None:
        return None
    return {key: candidate[key] for key in (
        "label", "artifact", "batch_size", "draft_length", "selector_batches",
        "successful_trials", "failed_trials", "aggregate_tokens_per_second",
        "p95_ttft_seconds", "prefill_gpu_forward_ms", "acceptance_length",
        "acceptance_rate", "position_survival_rate"
    )}


def shape_decisions(candidates, minimum_trials, threshold):
    """Choose a draft length only after it clears an identical-shape baseline."""
    groups = defaultdict(list)
    for item in candidates:
        groups[(item["prompt_tokens"], item["output_tokens"],
                item["batch_size"])].append(item)
    decisions = []
    for shape, items in sorted(groups.items()):
        qualified = [item for item in items if eligible(item, minimum_trials)]
        target = max((item for item in qualified if item["draft_length"] is None),
                     key=lambda item: item["aggregate_tokens_per_second"], default=None)
        draft = max((item for item in qualified if item["draft_length"] is not None),
                    key=lambda item: item["aggregate_tokens_per_second"], default=None)
        decision = "INSUFFICIENT_EVIDENCE"
        relative_effect = None
        if target and draft:
            relative_effect = effect(draft, target)
            if relative_effect >= threshold:
                decision = "ENABLE_DFLASH"
            elif relative_effect <= -threshold:
                decision = "DISABLE_DFLASH"
            else:
                decision = "KEEP_BASELINE"
        elif len([item for item in qualified if item["draft_length"] is not None]) >= 2:
            ordered = sorted(
                (item for item in qualified if item["draft_length"] is not None),
                key=lambda item: item["aggregate_tokens_per_second"], reverse=True)
            relative_effect = effect(ordered[0], ordered[1])
            decision = ("USE_DRAFT_LENGTH" if relative_effect >= threshold
                        else "KEEP_BASELINE")
            draft = ordered[0]
        decisions.append({
            "prompt_tokens": shape[0],
            "output_tokens": shape[1],
            "batch_size": shape[2],
            "decision": decision,
            "relative_throughput_effect": relative_effect,
            "target_only": compact(target),
            "best_dflash": compact(draft),
            "observed_candidates": len(items),
        })
    return decisions


def envelope_decisions(candidates, minimum_trials, threshold):
    """Compare each policy at its best safe concurrency for a request shape."""
    groups = defaultdict(list)
    for item in candidates:
        groups[(item["prompt_tokens"], item["output_tokens"])].append(item)
    decisions = []
    for shape, items in sorted(groups.items()):
        qualified = [item for item in items if eligible(item, minimum_trials)]
        target = max((item for item in qualified if item["draft_length"] is None),
                     key=lambda item: item["aggregate_tokens_per_second"], default=None)
        draft = max((item for item in qualified if item["draft_length"] is not None),
                    key=lambda item: item["aggregate_tokens_per_second"], default=None)
        decision = "INSUFFICIENT_EVIDENCE"
        relative_effect = None
        if target and draft:
            relative_effect = effect(draft, target)
            if relative_effect >= threshold:
                decision = "ENABLE_DFLASH"
            elif relative_effect <= -threshold:
                decision = "DISABLE_DFLASH"
            else:
                decision = "KEEP_BASELINE"
        decisions.append({
            "prompt_tokens": shape[0],
            "output_tokens": shape[1],
            "objective": "MAXIMUM_AGGREGATE_DECODE_TOKENS_PER_SECOND",
            "decision": decision,
            "relative_throughput_effect": relative_effect,
            "target_only": compact(target),
            "best_dflash": compact(draft),
        })
    return decisions


def main():
    """Learn and write effect-thresholded speculative decoding decisions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--minimum-trials", type=int, default=5)
    parser.add_argument("--minimum-relative-effect", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for item in args.inputs:
        paths.extend(sorted(item.glob("*.json")) if item.is_dir() else [item])
    candidates = load_candidates(paths)
    result = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy_change_gate": {
            "minimum_successful_trials": args.minimum_trials,
            "maximum_failed_trials": 0,
            "minimum_absolute_relative_throughput_effect": (
                args.minimum_relative_effect
            ),
        },
        "shape_decisions": shape_decisions(
            candidates, args.minimum_trials, args.minimum_relative_effect),
        "system_envelope_decisions": envelope_decisions(
            candidates, args.minimum_trials, args.minimum_relative_effect),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
