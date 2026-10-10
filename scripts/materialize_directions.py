#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Acquire the public direction Release asset or verify an existing copy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPOSITORY = "Blackfrost-AI/GLM-5.3-DWM32-A2.2-Recipe"
TAG = "directions-v1"
NAME = "glm53-nvfp4-raw32.directions.safetensors"
BYTES = 193_612_804
SHA256 = "e12d8fee3e2f75dc7e269bc09c810f079eeb1abce258376ce65f811192db27f8"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: pathlib.Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"asset not found: {path}")
    if path.stat().st_size != BYTES:
        raise RuntimeError(f"asset size mismatch: {path.stat().st_size} != {BYTES}")
    actual = sha256(path)
    if actual != SHA256:
        raise RuntimeError(f"asset SHA-256 mismatch: {actual}")


def copy_atomic(source: pathlib.Path, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output.parent)
    if usage.free < BYTES + 100_000_000:
        raise RuntimeError("insufficient free space for direction asset plus 100 MB reserve")
    temporary = output.with_name(f".{output.name}.partial")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    verify(temporary)
    os.replace(temporary, output)


def download(output: pathlib.Path) -> None:
    if shutil.which("gh") is None:
        raise RuntimeError("GitHub CLI (gh) is required to download the public Release asset")
    output.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output.parent)
    if usage.free < BYTES + 100_000_000:
        raise RuntimeError("insufficient free space for direction asset plus 100 MB reserve")
    with tempfile.TemporaryDirectory(prefix="glm53-a2p2-", dir=output.parent) as temp_dir:
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                TAG,
                "--repo",
                REPOSITORY,
                "--pattern",
                NAME,
                "--dir",
                temp_dir,
            ],
            check=True,
        )
        source = pathlib.Path(temp_dir) / NAME
        verify(source)
        os.replace(source, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("artifacts") / NAME,
    )
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        help="Verify and copy an existing exact asset instead of using GitHub",
    )
    parser.add_argument(
        "--verify-only",
        type=pathlib.Path,
        help="Verify an existing file without copying it",
    )
    args = parser.parse_args()
    if args.source and args.verify_only:
        parser.error("choose --source or --verify-only")
    if args.verify_only:
        target = args.verify_only.resolve()
        verify(target)
    else:
        target = args.output.resolve()
        if target.exists():
            verify(target)
        elif args.source:
            source = args.source.resolve()
            verify(source)
            copy_atomic(source, target)
        else:
            download(target)
            verify(target)
    print(
        json.dumps(
            {"verified": True, "path": str(target), "bytes": BYTES, "sha256": SHA256},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"verified": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
