#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed verification for the GLM-5.3 alpha-2.2 recipe package."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILE_HASHES = {
    "assets/raw32.pairs.json": "2b59fd6a6ea02cb22828bab0c6af41df5886836110ddee5b53c5d7deae3b994b",
    "assets/raw32.frozen.jsonl": "bade6eb8be22cd93fe36fe5bbd027b75245bb17ac78e34dd583280de24f92f1d",
    "assets/harmful.txt": "b8f800500faad5f32208044dd979e16233a7793157a7ee7bfa97aa54832cf32d",
    "assets/control.txt": "80b826fd56b555dd8f2b8c813bb4a0972d4ba337fe46eb7aebe91fdcdee3fa9d",
}
DIRECTION_NAME = "glm53-nvfp4-raw32.directions.safetensors"
DIRECTION_BYTES = 193_612_804
DIRECTION_SHA256 = "e12d8fee3e2f75dc7e269bc09c810f079eeb1abce258376ce65f811192db27f8"
DIRECTION_HEADER_BYTES = 1512
DIRECTION_META = {
    "schema": "blackfrost.glm53_nvfp4.raw32_native_capture.v1",
    "repo": "nvidia/GLM-5.3-NVFP4",
    "revision": "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
    "runtime": "sglang_native_modelopt_mixed",
    "primary_site": "post_layer_residual_sum",
    "template_applied": "false",
    "generated_tokens": "0",
    "weights_modified": "false",
    "alpha_selected": "none",
}
RUNTIME_EXCLUSIONS = {
    "SHA256SUMS",
    "config.local.json",
    ".DS_Store",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def value_at(document, dotted: str):
    value = document
    for part in dotted.split("."):
        value = value[part]
    return value


def verify_lock() -> dict:
    lock = read_json(ROOT / "recipe.lock.json")
    expected = {
        "schema": "blackfrost.dwm.exact-replay.v1",
        "recipe_id": "glm-5.3-bf16-dwm32-a2p2-nvfp4",
        "source.repo": "zai-org/GLM-5.3-BF16",
        "source.revision": "9d2398f478cab2de883137db3a36ad2c96205e24",
        "source.config_sha256": "ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca",
        "source.index_sha256": "5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e",
        "source.indexed_tensors": 59585,
        "source.weight_shards": 282,
        "source.lfs_shard_bytes": 1506667387408,
        "source.lfs_manifest_canonical_sha256": "d7e8f6f8f6d6b03413e7fe5bdefa65c1a4b8d6dd19f30dd963498595ea8c84c2",
        "directions.sha256": DIRECTION_SHA256,
        "directions.bytes": DIRECTION_BYTES,
        "directions.human_pairs_sha256": FILE_HASHES["assets/raw32.pairs.json"],
        "directions.frozen_tokens_sha256": FILE_HASHES["assets/raw32.frozen.jsonl"],
        "operation.alpha": 2.2,
        "operation.passes": 1,
        "operation.rank": 1,
        "operation.first_layer": 2,
        "operation.last_layer": 77,
        "operation.excluded_mtp_layer": 78,
        "operation.targets.total": 19352,
        "operation.targets.o_proj": 76,
        "operation.targets.dense_down": 1,
        "operation.targets.expert_down": 19200,
        "operation.targets.shared_down": 75,
        "serialization.reference_revision": "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
        "serialization.config_sha256": "b0b6b9ab922a82e3688bc39a4117bc443e106fcf88969ef48b3627778f4bdf7b",
        "serialization.quant_config_sha256": "13799d4fed353c20dde2264fbdd7f68a780b514782f8d80bba96ac5e44aa9bba",
        "serialization.index_sha256": "157752617100cefaf25a468812e14abefca24c356e6e2f7ae55bce8c3b511863",
        "serialization.weight_shards": 50,
        "serialization.indexed_tensors": 232403,
        "serialization.expected_data_bytes": 464115789896,
        "serialization.lfs_shard_bytes": 464143549144,
        "serialization.lfs_manifest_canonical_sha256": "b689de6aaa32aa8ce801be0bbf2a46b6ac0fb66afc5d14daecfacfa9f213b66a",
        "serialization.activation_scales_reused": 57609,
        "serialization.fresh_post_dwm_activation_calibration": False,
        "serialization.modelopt_commit": "022767c7ab3d7d36211affd85e5c496770cde768",
        "reproducibility.class": "method-equivalent",
        "reproducibility.byte_identical_output_promised": False,
    }
    for key, wanted in expected.items():
        actual = value_at(lock, key)
        require(actual == wanted, f"recipe lock mismatch: {key}: {actual!r} != {wanted!r}")
    require(lock["operation"]["norm_restoration"] is False, "norm restoration must be false")
    require(
        lock["operation"]["quantization_correction_iterations"] == 0,
        "quantization correction iterations must be zero",
    )
    require(lock["source"]["runtime_files_sha256"] == {
        "chat_template.jinja": "3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c",
        "generation_config.json": "ac76b43d8683d3b930126870fc8be73d8679308fe752fa1f381096d8354f6a55",
        "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
        "tokenizer_config.json": "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc",
    }, "source runtime file hashes mismatch")
    require(lock["serialization"]["modelopt_critical_file_sha256"] == {
        "modelopt/__init__.py": "96647188e0afa8dcdcadbe42178feeb4cc19113d9ffcbaf633bd62414bd6d453",
        "nvfp4_tensor.py": "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e",
        "fp8_tensor.py": "ec9caf56fc140720a55732f827dbdd41b9fdf8fcb961a1e37028d4d4b69b7fae",
    }, "ModelOpt critical file hashes mismatch")
    return lock


def verify_lfs_manifest(relative: str, repository: str, revision: str, count: int, total: int, canonical_sha256: str) -> dict:
    document = read_json(ROOT / relative)
    require(document["schema"] == "blackfrost.huggingface-lfs-shard-manifest.v1", f"LFS schema mismatch: {relative}")
    require(document["repository"] == repository, f"LFS repository mismatch: {relative}")
    require(document["revision"] == revision, f"LFS revision mismatch: {relative}")
    require(document["shard_count"] == count, f"LFS shard count mismatch: {relative}")
    require(document["total_bytes"] == total, f"LFS byte total mismatch: {relative}")
    require(len(document["entries"]) == count, f"LFS entry count mismatch: {relative}")
    canonical = bytearray()
    seen = set()
    for number, entry in enumerate(document["entries"], 1):
        name = entry["path"]
        require(name == f"model-{number:05d}-of-{count:05d}.safetensors", f"LFS sequence mismatch: {name}")
        require(name not in seen, f"duplicate LFS path: {name}")
        seen.add(name)
        require(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None, f"invalid LFS SHA-256: {name}")
        require(entry["lfs_oid"] == f"sha256:{entry['sha256']}", f"LFS OID mismatch: {name}")
        canonical.extend(f"{name}\t{entry['bytes']}\t{entry['sha256']}\n".encode("utf-8"))
    require(sum(entry["bytes"] for entry in document["entries"]) == total, f"LFS entry byte total mismatch: {relative}")
    actual = hashlib.sha256(canonical).hexdigest()
    require(actual == canonical_sha256, f"LFS canonical hash mismatch: {relative}")
    require(document["canonicalization"]["sha256"] == actual, f"LFS embedded hash mismatch: {relative}")
    return {"shards": count, "bytes": total, "canonical_sha256": actual}


def verify_descriptors() -> dict:
    asset = read_json(ROOT / "assets/directions.asset.json")
    require(
        asset == {
            "schema": "blackfrost.release-asset.v1",
            "repository": "Blackfrost-AI/GLM-5.3-DWM32-A2.2-Recipe",
            "release_tag": "directions-v1",
            "asset_name": DIRECTION_NAME,
            "bytes": DIRECTION_BYTES,
            "sha256": DIRECTION_SHA256,
            "git_tracked": False,
            "visibility": "public",
            "required_for_execution": True,
            "contains": [
                "78 unit direction vectors of width 6144",
                "32 x 78 x 6144 paired deltas",
                "64 x 78 x 6144 interleaved prompt activations",
                "capture summary tensors",
            ],
        },
        "direction asset descriptor mismatch",
    )
    environment = read_json(ROOT / "environment.lock.json")
    require(
        environment["modelopt_commit"]
        == "022767c7ab3d7d36211affd85e5c496770cde768",
        "ModelOpt commit mismatch",
    )
    require(
        environment["modelopt_module_init_sha256"]
        == "96647188e0afa8dcdcadbe42178feeb4cc19113d9ffcbaf633bd62414bd6d453",
        "ModelOpt module hash mismatch",
    )
    require(
        environment["modelopt_nvfp4_tensor_sha256"]
        == "e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e",
        "ModelOpt NVFP4 codec hash mismatch",
    )
    require(
        environment["modelopt_fp8_tensor_sha256"]
        == "ec9caf56fc140720a55732f827dbdd41b9fdf8fcb961a1e37028d4d4b69b7fae",
        "ModelOpt FP8 codec hash mismatch",
    )
    return {"asset": asset["asset_name"], "environment": environment["reproducibility_class"]}


def verify_pair_material() -> dict:
    for relative, wanted in FILE_HASHES.items():
        path = ROOT / relative
        require(path.is_file(), f"missing locked file: {relative}")
        require(sha256(path) == wanted, f"hash mismatch: {relative}")

    pair_doc = read_json(ROOT / "assets/raw32.pairs.json")
    pairs = pair_doc["pairs"]
    rows = [
        json.loads(line)
        for line in (ROOT / "assets/raw32.frozen.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    harmful = (ROOT / "assets/harmful.txt").read_text(encoding="utf-8").splitlines()
    control = (ROOT / "assets/control.txt").read_text(encoding="utf-8").splitlines()
    require(len(pairs) == len(harmful) == len(control) == 32, "expected exactly 32 pairs")
    require(len(rows) == 64, "expected exactly 64 frozen rows")
    for index, pair in enumerate(pairs):
        require(pair["pair_index"] == index, f"pair index mismatch at {index}")
        require(
            pair["pair_id"] == f"glm53-original32-{index + 1:02d}",
            f"pair id mismatch at {index}",
        )
        require(pair["harmful"] == harmful[index], f"harmful text mismatch at {index}")
        require(pair["control"] == control[index], f"control text mismatch at {index}")
        for offset, label, text in (
            (0, "harmful", harmful[index]),
            (1, "control", control[index]),
        ):
            row = rows[index * 2 + offset]
            require(row["row"] == index * 2 + offset, f"row number mismatch at pair {index}")
            require(
                row["pair_index"] == index and row["pair_id"] == pair["pair_id"],
                f"frozen pair binding mismatch at {index}",
            )
            require(
                row["label"] == label and row["text"] == text,
                f"frozen label/text mismatch at {index}/{label}",
            )
            require(
                row["token_count"] == len(row["input_ids"]) > 0,
                f"frozen token count mismatch at {index}/{label}",
            )
    return {
        "pairs": 32,
        "rows": 64,
        "token_count_min": min(row["token_count"] for row in rows),
        "token_count_max": max(row["token_count"] for row in rows),
    }


def verify_direction(path: pathlib.Path) -> dict:
    require(path.is_file(), f"direction artifact not found: {path}")
    require(path.name == DIRECTION_NAME, f"unexpected direction filename: {path.name}")
    require(path.stat().st_size == DIRECTION_BYTES, "direction byte count mismatch")
    require(sha256(path) == DIRECTION_SHA256, "direction SHA-256 mismatch")
    with path.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        require(header_bytes == DIRECTION_HEADER_BYTES, "direction header length mismatch")
        header = json.loads(handle.read(header_bytes))
    metadata = header.get("__metadata__", {})
    for key, wanted in DIRECTION_META.items():
        require(metadata.get(key) == wanted, f"direction metadata mismatch: {key}")
    dirs = header.get("dirs", {})
    require(dirs.get("dtype") == "F32", "dirs dtype must be F32")
    require(dirs.get("shape") == [78, 6144], "dirs shape must be [78, 6144]")
    require(
        header.get("pair_deltas.layer_end", {}).get("shape") == [32, 78, 6144],
        "pair delta tensor shape mismatch",
    )
    require(
        header.get("activations.interleaved.layer_end", {}).get("shape") == [64, 78, 6144],
        "activation tensor shape mismatch",
    )
    return {
        "path": str(path),
        "bytes": DIRECTION_BYTES,
        "sha256": DIRECTION_SHA256,
        "dirs_shape": [78, 6144],
    }


def verify_manifest() -> dict:
    path = ROOT / "SHA256SUMS"
    require(path.is_file() and not path.is_symlink(), "SHA256SUMS is required")
    count = 0
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wanted, relative = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", wanted) is not None, f"invalid manifest hash: {relative}")
        relative_path = pathlib.PurePosixPath(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts, f"unsafe manifest path: {relative}")
        require(relative not in seen, f"duplicate manifest entry: {relative}")
        seen.add(relative)
        target = ROOT / relative
        require(target.is_file() and not target.is_symlink(), f"manifest target invalid: {relative}")
        require(sha256(target) == wanted, f"manifest hash mismatch: {relative}")
        count += 1
    expected = set()
    for target in ROOT.rglob("*"):
        if not target.is_file() or target.is_symlink():
            continue
        relative = target.relative_to(ROOT).as_posix()
        parts = pathlib.PurePosixPath(relative).parts
        if relative in RUNTIME_EXCLUSIONS or ".git" in parts or "__pycache__" in parts:
            continue
        if relative.endswith(".pyc") or relative.startswith(("run/", "output/")):
            continue
        if relative.startswith("artifacts/") and relative.endswith(".safetensors"):
            continue
        expected.add(relative)
    require(seen == expected, f"manifest coverage mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    return {"present": True, "files": count, "complete": True}


def verify_harness_plan() -> dict:
    plan = read_json(ROOT / "plans/ablagent-read-only-audit.json")
    require(plan["name"] == "glm53-dwm32-a2p2-locked-method-audit", "unexpected plan name")
    require(len(plan["stages"]) == 8, "unexpected harness stage count")
    allowed = {"read_file", "grep", "glob"}
    stage_ids = []
    for stage in plan["stages"]:
        stage_ids.append(stage["id"])
        require(set(stage["allowedTools"]).issubset(allowed), f"write-capable tool in stage {stage['id']}")
        require(stage["maxToolCalls"] >= 0, f"invalid tool budget: {stage['id']}")
        for relative in stage.get("exactPaths", []):
            path = pathlib.PurePosixPath(relative)
            require(not path.is_absolute() and ".." not in path.parts, f"unsafe exact path: {relative}")
            require((ROOT / relative).is_file(), f"missing exact path: {relative}")
    require(stage_ids == ["contract", "lock", "runbook", "capture-evidence", "run-evidence", "asset", "executor-source", "synthesis"], "harness stage order mismatch")
    require(plan["stages"][-1]["allowedTools"] == [], "synthesis stage must have no tools")
    return {"name": plan["name"], "stages": len(plan["stages"]), "tools": sorted(allowed)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directions",
        type=pathlib.Path,
        help="Verify the separately distributed direction tensor",
    )
    args = parser.parse_args()
    for path in ROOT.rglob("*"):
        require(
            not path.is_symlink(),
            f"symlink is not allowed in package: {path.relative_to(ROOT)}",
        )
    result = {
        "passed": True,
        "lock": verify_lock()["recipe_id"],
        "descriptors": verify_descriptors(),
        "source_lfs": verify_lfs_manifest(
            "evidence/source-lfs-manifest.json",
            "zai-org/GLM-5.3-BF16",
            "9d2398f478cab2de883137db3a36ad2c96205e24",
            282,
            1506667387408,
            "d7e8f6f8f6d6b03413e7fe5bdefa65c1a4b8d6dd19f30dd963498595ea8c84c2",
        ),
        "reference_lfs": verify_lfs_manifest(
            "evidence/reference-lfs-manifest.json",
            "nvidia/GLM-5.3-NVFP4",
            "e3b8f9f2fad29727dfa95e766c5cac18ea1414bf",
            50,
            464143549144,
            "b689de6aaa32aa8ce801be0bbf2a46b6ac0fb66afc5d14daecfacfa9f213b66a",
        ),
        "pair_material": verify_pair_material(),
        "manifest": verify_manifest(),
        "harness_plan": verify_harness_plan(),
        "directions": (
            verify_direction(args.directions.resolve()) if args.directions else "not supplied"
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
