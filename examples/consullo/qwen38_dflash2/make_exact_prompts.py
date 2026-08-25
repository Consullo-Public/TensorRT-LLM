#!/usr/bin/env python3
"""Generate semantically consistent prompts with exact chat-template token counts."""

import argparse
import json
import math
from pathlib import Path

from transformers import AutoTokenizer


SEED = (
    "Write an indefinitely continuing numbered technical analysis of paged KV caches, "
    "speculative decoding, batching, and GPU memory movement. Each item must introduce "
    "a new implementation detail. Do not conclude or summarize. "
)


def parse_targets(value):
    """Parse positive comma-separated token targets."""
    targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not targets or any(item <= 0 for item in targets):
        raise argparse.ArgumentTypeError("targets must contain positive integers")
    return targets


def chat_token_count(tokenizer, content):
    """Count the exact tokens sent to generation after chat templating."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    return len(input_ids)


def exact_prompt(tokenizer, target):
    """Find a repeated-seed prefix whose chat-template length equals target."""
    seed_ids = tokenizer.encode(SEED, add_special_tokens=False)
    repeats = math.ceil((target * 2 + 512) / len(seed_ids))
    pool = seed_ids * repeats
    low = 1
    high = len(pool)
    while low <= high:
        middle = (low + high) // 2
        content = tokenizer.decode(pool[:middle], skip_special_tokens=True)
        if chat_token_count(tokenizer, content) < target:
            low = middle + 1
        else:
            high = middle - 1
    for length in range(max(1, high - 16), min(len(pool), low + 16) + 1):
        content = tokenizer.decode(pool[:length], skip_special_tokens=True)
        if chat_token_count(tokenizer, content) == target:
            return content
    raise RuntimeError(f"could not construct an exact {target}-token prompt")


def main():
    """Write exact prompt files and a machine-readable manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-label", default="Qwen3.8-27B-NVFP4")
    parser.add_argument("--targets", type=parse_targets, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    manifest = {"tokenizer": args.tokenizer_label, "prompts": []}
    for target in args.targets:
        content = exact_prompt(tokenizer, target)
        path = args.output_dir / f"prompt-{target}.txt"
        path.write_text(content, encoding="utf-8")
        manifest["prompts"].append({
            "path": path.name,
            "chat_template_tokens": chat_token_count(tokenizer, content),
        })
    manifest_path = args.output_dir / "prompt-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
