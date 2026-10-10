#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the locked Hugging Face LFS shard manifests.

The Hugging Face tree API exposes each Git LFS object's SHA-256 OID and byte
count without downloading the multi-terabyte model snapshots.  This command
queries only the two revisions locked by this recipe and writes deterministic,
path-free JSON evidence.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

from huggingface_hub import HfApi, RepoFile


ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
SHARD_PATTERN = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
SPECS = (
    {
        "repository": "zai-org/GLM-5.3-BF16",
        "revision": "9d2398f478cab2de883137db3a36ad2c96205e24",
        "shards": 282,
        "total_bytes": 1_506_667_387_408,
        "canonical_sha256": "d7e8f6f8f6d6b03413e7fe5bdefa65c1a4b8d6dd19f30dd963498595ea8c84c2",
        "output": "source-lfs-manifest.json",
    },
    {
        "repository": "nvidia/GLM-5.3-NVFP4",
        "revision": "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
        "shards": 50,
        "total_bytes": 464_143_549_144,
        "canonical_sha256": "b689de6aaa32aa8ce801be0bbf2a46b6ac0fb66afc5d14daecfacfa9f213b66a",
        "output": "reference-lfs-manifest.json",
    },
)


def build_manifest(api: HfApi, spec: dict[str, object]) -> dict[str, object]:
    repository = str(spec["repository"])
    revision = str(spec["revision"])
    expected_count = int(spec["shards"])
    files = sorted(
        (
            item
            for item in api.list_repo_tree(
                repository,
                revision=revision,
                repo_type="model",
                recursive=True,
                expand=True,
            )
            if isinstance(item, RepoFile) and SHARD_PATTERN.fullmatch(item.path)
        ),
        key=lambda item: item.path,
    )
    if len(files) != expected_count:
        raise RuntimeError(
            f"{repository}@{revision}: expected {expected_count} shards, found {len(files)}"
        )

    entries: list[dict[str, object]] = []
    canonical = bytearray()
    for expected_number, item in enumerate(files, 1):
        match = SHARD_PATTERN.fullmatch(item.path)
        if match is None:
            raise RuntimeError(f"unexpected shard name: {item.path}")
        number, denominator = (int(value) for value in match.groups())
        if (number, denominator) != (expected_number, expected_count):
            raise RuntimeError(f"non-contiguous shard sequence: {item.path}")
        if item.lfs is None:
            raise RuntimeError(f"shard has no Git LFS metadata: {item.path}")
        if item.size != item.lfs.size:
            raise RuntimeError(f"tree/LFS size mismatch: {item.path}")
        sha256 = item.lfs.sha256.lower()
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise RuntimeError(f"invalid Git LFS SHA-256: {item.path}")
        entries.append(
            {
                "path": item.path,
                "bytes": item.size,
                "sha256": sha256,
                "lfs_oid": f"sha256:{sha256}",
            }
        )
        canonical.extend(f"{item.path}\t{item.size}\t{sha256}\n".encode("utf-8"))

    total_bytes = sum(int(entry["bytes"]) for entry in entries)
    canonical_sha256 = hashlib.sha256(canonical).hexdigest()
    if total_bytes != int(spec["total_bytes"]):
        raise RuntimeError(
            f"{repository}@{revision}: total bytes {total_bytes} != {spec['total_bytes']}"
        )
    if canonical_sha256 != spec["canonical_sha256"]:
        raise RuntimeError(
            f"{repository}@{revision}: canonical SHA-256 changed to {canonical_sha256}"
        )

    return {
        "schema": "blackfrost.huggingface-lfs-shard-manifest.v1",
        "repository": repository,
        "revision": revision,
        "shard_count": len(entries),
        "total_bytes": total_bytes,
        "canonicalization": {
            "ordering": "ascending UTF-8 path order",
            "record_format": "{path}\\t{bytes}\\t{sha256}\\n",
            "encoding": "UTF-8",
            "sha256": canonical_sha256,
        },
        "entries": entries,
    }


def main() -> None:
    api = HfApi()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        manifest = build_manifest(api, spec)
        target = EVIDENCE / str(spec["output"])
        target.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "path": str(target.relative_to(ROOT)),
                    "repository": manifest["repository"],
                    "revision": manifest["revision"],
                    "shard_count": manifest["shard_count"],
                    "total_bytes": manifest["total_bytes"],
                    "canonical_sha256": manifest["canonicalization"]["sha256"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
