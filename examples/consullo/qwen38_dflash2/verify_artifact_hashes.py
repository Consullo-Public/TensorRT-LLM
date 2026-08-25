#!/usr/bin/env python3
"""Verify local files against a JSON SHA-256 manifest."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    """Calculate a file SHA-256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    """Validate every manifest entry and fail closed on missing files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest.get("artifacts", []):
        path = (args.root / entry["path"]).resolve()
        expected = entry["sha256"].lower()
        if not path.is_file():
            failures.append(f"missing: {path}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch: {path}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {len(manifest.get('artifacts', []))} artifacts")


if __name__ == "__main__":
    main()

