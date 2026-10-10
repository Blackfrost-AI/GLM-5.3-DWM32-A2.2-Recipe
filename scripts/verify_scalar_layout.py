#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Verify serialized scalar scales and NVIDIA activation-scale equality."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import struct
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: pathlib.Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--reference-dir", required=True, type=pathlib.Path)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    output = args.output.resolve()
    reference_dir = args.reference_dir.resolve()
    run_dir = args.run_dir.resolve()

    reference = json.loads(
        (reference_dir / "calibrated-scalars.json").read_text(encoding="utf-8")
    )
    index = json.loads(
        (output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    scalars = {}
    for shard in sorted(set(index.values())):
        with (output / shard).open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            entries = {
                key: entry
                for key, entry in header.items()
                if key != "__metadata__"
                and entry["dtype"] == "F32"
                and entry["shape"] in ([], [1])
            }
            maximum = max((entry["data_offsets"][1] for entry in entries.values()), default=0)
            require(maximum < 1_000_000, f"unexpected scalar prefix size in {shard}")
            prefix = handle.read(maximum)
            for key, entry in entries.items():
                start, end = entry["data_offsets"]
                require(end - start == 4 and key not in scalars, f"invalid scalar: {key}")
                value = struct.unpack("<f", prefix[start:end])[0]
                require(math.isfinite(value) and value > 0, f"invalid scale value: {key}")
                scalars[key] = value

    activation_count = 0
    pair_count = 0
    for key, value in scalars.items():
        if key.endswith(".input_scale"):
            require(key in reference and value == reference[key], f"activation scale mismatch: {key}")
            activation_count += 1
        if ".gate_proj.weight_scale" in key:
            sibling = key.replace(".gate_proj.", ".up_proj.")
            require(sibling in scalars and scalars[sibling] == value, f"gate/up scale mismatch: {key}")
            pair_count += 1

    require(len(scalars) == 115_218, "expected 115218 serialized scalar scales")
    require(activation_count == 57_609, "expected 57609 exact NVIDIA activation scales")
    require(pair_count == 19_203, "expected 19203 exact gate/up scale pairs")
    result = {
        "passed": True,
        "serialized_positive_finite_scales": len(scalars),
        "nvidia_activation_scales_exact": activation_count,
        "gate_up_weight_scale_pairs_exact": pair_count,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "serialized-scale-verification.json", result)
    atomic_json(output / "SERIALIZED_SCALE_VERIFICATION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
