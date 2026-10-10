#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Hash every pinned BF16 shard and write a bound local verification receipt."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "evidence/source-lfs-manifest.json"
EXPECTED_CANONICAL_SHA256 = "d7e8f6f8f6d6b03413e7fe5bdefa65c1a4b8d6dd19f30dd963498595ea8c84c2"
EXPECTED_SHARDS = 282
EXPECTED_BYTES = 1_506_667_387_408


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> tuple[dict, dict[str, dict]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    require(manifest["repository"] == "zai-org/GLM-5.3-BF16", "source repository mismatch")
    require(manifest["revision"] == "9d2398f478cab2de883137db3a36ad2c96205e24", "source revision mismatch")
    require(manifest["shard_count"] == EXPECTED_SHARDS, "source shard count mismatch")
    require(manifest["total_bytes"] == EXPECTED_BYTES, "source byte total mismatch")
    entries = manifest["entries"]
    require(len(entries) == EXPECTED_SHARDS, "source manifest entry count mismatch")
    canonical = bytearray()
    by_name = {}
    for number, entry in enumerate(entries, 1):
        name = entry["path"]
        require(name == f"model-{number:05d}-of-{EXPECTED_SHARDS:05d}.safetensors", f"unexpected shard sequence: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None, f"invalid shard hash: {name}")
        require(entry["lfs_oid"] == f"sha256:{entry['sha256']}", f"LFS OID mismatch: {name}")
        canonical.extend(f"{name}\t{entry['bytes']}\t{entry['sha256']}\n".encode("utf-8"))
        by_name[name] = entry
    actual = hashlib.sha256(canonical).hexdigest()
    require(actual == EXPECTED_CANONICAL_SHA256, "source manifest canonical hash mismatch")
    require(manifest["canonicalization"]["sha256"] == actual, "source manifest receipt mismatch")
    return manifest, by_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require(1 <= args.workers <= 8, "workers must be between 1 and 8")
    source = args.source.resolve()
    receipt = args.receipt.resolve()
    require(source.is_dir(), f"source directory not found: {source}")
    manifest, entries = load_manifest()
    actual_names = {path.name for path in source.glob("*.safetensors")}
    require(actual_names == set(entries), "source shard filename set mismatch")

    def verify_one(name: str) -> tuple[str, dict]:
        path = source / name
        expected = entries[name]
        require(path.is_file() and not path.is_symlink(), f"invalid source shard: {name}")
        before = path.stat()
        require(before.st_size == expected["bytes"], f"source shard size mismatch: {name}")
        actual_hash = sha256(path)
        after = path.stat()
        require(
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"source shard changed while hashing: {name}",
        )
        require(actual_hash == expected["sha256"], f"source shard SHA-256 mismatch: {name}")
        return name, {
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": actual_hash,
        }

    files = {}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, result in pool.map(verify_one, sorted(entries)):
            files[name] = result
            print(json.dumps({"verified": name, "bytes": result["bytes"]}), flush=True)

    document = {
        "schema": "blackfrost.verified-local-source.v1",
        "complete": True,
        "repository": manifest["repository"],
        "revision": manifest["revision"],
        "source_root": str(source),
        "manifest_canonical_sha256": EXPECTED_CANONICAL_SHA256,
        "shard_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "files": files,
    }
    require(document["total_bytes"] == EXPECTED_BYTES, "verified byte total mismatch")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_name(f".{receipt.name}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, receipt)
    print(json.dumps({
        "passed": True,
        "receipt": str(receipt),
        "shards": len(files),
        "bytes": document["total_bytes"],
        "manifest_canonical_sha256": EXPECTED_CANONICAL_SHA256,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
