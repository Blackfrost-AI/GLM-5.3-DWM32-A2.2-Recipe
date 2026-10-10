#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Fetch pinned NVIDIA layout headers and calibration scalars without shards."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import struct
import sys
import time
import urllib.request

REPO = "nvidia/GLM-5.3-NVFP4"
REVISION = "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf"
LOCKED_FILES = {
    "config.json": "b0b6b9ab922a82e3688bc39a4117bc443e106fcf88969ef48b3627778f4bdf7b",
    "hf_quant_config.json": "13799d4fed353c20dde2264fbdd7f68a780b514782f8d80bba96ac5e44aa9bba",
    "model.safetensors.index.json": "157752617100cefaf25a468812e14abefca24c356e6e2f7ae55bce8c3b511863",
}
EXPECTED_SHARDS = 50
EXPECTED_TENSORS = 232_403
EXPECTED_DATA_BYTES = 464_115_789_896
EXPECTED_SCALARS = 115_218
EXPECTED_LFS_BYTES = 464_143_549_144
EXPECTED_LFS_CANONICAL_SHA256 = "b689de6aaa32aa8ce801be0bbf2a46b6ac0fb66afc5d14daecfacfa9f213b66a"
ROOT = pathlib.Path(__file__).resolve().parents[1]
REFERENCE: pathlib.Path
INDEX: dict
SHARD_SIZES: dict[str, int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: pathlib.Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def url_for(name: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true"


def fetch_locked_file(name: str, wanted: str) -> None:
    path = REFERENCE / name
    if path.exists():
        require(path.is_file() and sha256_file(path) == wanted, f"hash mismatch: {name}")
        return
    with urllib.request.urlopen(url_for(name), timeout=90) as response:
        value = response.read()
    require(sha256_bytes(value) == wanted, f"download hash mismatch: {name}")
    atomic_write(path, value)


def ranged(url: str, start: int, end: int, total: int) -> bytes:
    require(start >= 0 and end >= start, "invalid byte range")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                require(response.status == 206, f"range request returned {response.status}")
                content_range = response.headers.get("Content-Range", "")
                require(content_range == f"bytes {start}-{end}/{total}", f"unexpected Content-Range: {content_range}")
                data = response.read(end - start + 2)
            require(len(data) == end - start + 1, "range length mismatch")
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))
    raise RuntimeError("unreachable")


def fetch_shard(shard: str) -> tuple[str, dict]:
    scalars_path = REFERENCE / f"{shard}.scalars.json"
    header_path = REFERENCE / f"{shard}.header.json"
    # Always re-read the pinned remote bytes. Historical per-shard cache hashes
    # were not preserved, so accepting an existing cache would make a stale or
    # modified file indistinguishable from the exact reference.
    url = url_for(shard)
    expected_file_bytes = SHARD_SIZES[shard]
    header_size = struct.unpack("<Q", ranged(url, 0, 7, expected_file_bytes))[0]
    require(header_size <= 10_000_000, f"unexpected header size for {shard}: {header_size}")
    header = json.loads(ranged(url, 8, 7 + header_size, expected_file_bytes))
    data_bytes = max(
        (entry["data_offsets"][1] for key, entry in header.items() if key != "__metadata__"),
        default=0,
    )
    require(8 + header_size + data_bytes == expected_file_bytes, f"reference shard byte layout mismatch: {shard}")
    scalar_entries = {
        key: entry
        for key, entry in header.items()
        if key != "__metadata__" and entry["dtype"] == "F32" and entry["shape"] in ([], [1])
    }
    largest = max((entry["data_offsets"][1] for entry in scalar_entries.values()), default=0)
    require(largest <= 1_000_000, f"scalar prefix too large for {shard}: {largest}")
    prefix = ranged(url, 8 + header_size, 7 + header_size + largest, expected_file_bytes) if largest else b""
    scalars = {}
    for key, entry in scalar_entries.items():
        start, end = entry["data_offsets"]
        require(end - start == 4, f"non-scalar F32 entry: {key}")
        value = struct.unpack("<f", prefix[start:end])[0]
        require(value > 0 and value < float("inf"), f"invalid scalar value: {key}")
        scalars[key] = value

    atomic_write(
        header_path,
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    )
    atomic_write(
        scalars_path,
        json.dumps(scalars, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    )
    return shard, scalars


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True, type=pathlib.Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    require(1 <= args.workers <= 16, "workers must be between 1 and 16")

    global REFERENCE, INDEX, SHARD_SIZES
    REFERENCE = args.reference_dir.resolve()
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for name, wanted in LOCKED_FILES.items():
        fetch_locked_file(name, wanted)

    INDEX = json.loads((REFERENCE / "model.safetensors.index.json").read_text(encoding="utf-8"))
    require(len(INDEX["weight_map"]) == EXPECTED_TENSORS, "reference tensor count mismatch")
    require(INDEX["metadata"]["total_size"] == EXPECTED_DATA_BYTES, "reference data size mismatch")
    shards = sorted(set(INDEX["weight_map"].values()))
    require(len(shards) == EXPECTED_SHARDS, "reference shard count mismatch")
    lfs_manifest = json.loads(
        (ROOT / "evidence/reference-lfs-manifest.json").read_text(encoding="utf-8")
    )
    require(lfs_manifest["repository"] == REPO, "reference LFS repository mismatch")
    require(lfs_manifest["revision"] == REVISION, "reference LFS revision mismatch")
    require(lfs_manifest["shard_count"] == EXPECTED_SHARDS, "reference LFS shard count mismatch")
    require(lfs_manifest["total_bytes"] == EXPECTED_LFS_BYTES, "reference LFS byte total mismatch")
    canonical = "".join(
        f"{entry['path']}\t{entry['bytes']}\t{entry['sha256']}\n"
        for entry in lfs_manifest["entries"]
    ).encode("utf-8")
    require(hashlib.sha256(canonical).hexdigest() == EXPECTED_LFS_CANONICAL_SHA256, "reference LFS canonical hash mismatch")
    SHARD_SIZES = {entry["path"]: entry["bytes"] for entry in lfs_manifest["entries"]}
    require(set(SHARD_SIZES) == set(shards), "reference LFS/index shard set mismatch")

    all_scalars = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for shard, values in pool.map(fetch_shard, shards):
            require(not set(all_scalars).intersection(values), f"duplicate scalar key in {shard}")
            all_scalars.update(values)
            print(json.dumps({"reference_shard": shard, "scalar_count": len(values)}), flush=True)
    require(len(all_scalars) == EXPECTED_SCALARS, "calibrated scalar count mismatch")

    target = REFERENCE / "calibrated-scalars.json"
    atomic_write(
        target,
        json.dumps(all_scalars, separators=(",", ":"), sort_keys=True).encode("utf-8"),
    )
    cache_files = sorted(
        list(LOCKED_FILES)
        + ["calibrated-scalars.json"]
        + [f"{shard}.header.json" for shard in shards]
        + [f"{shard}.scalars.json" for shard in shards]
    )
    receipt = {
        "complete": True,
        "repo": REPO,
        "revision": REVISION,
        "reference_shards": len(shards),
        "indexed_tensors": len(INDEX["weight_map"]),
        "expected_data_bytes": INDEX["metadata"]["total_size"],
        "calibrated_scalars": len(all_scalars),
        "locked_file_sha256": LOCKED_FILES,
        "cache_sha256": {name: sha256_file(REFERENCE / name) for name in cache_files},
    }
    atomic_write(
        REFERENCE / "REFERENCE_RECEIPT.json",
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8"),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"complete": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
