#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Blackfrost AI
# SPDX-License-Identifier: Apache-2.0
"""Verify that the locked directions follow the recorded 32-pair derivation."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import torch
from safetensors import safe_open

EXPECTED_BYTES = 193_612_804
EXPECTED_SHA256 = "e12d8fee3e2f75dc7e269bc09c810f079eeb1abce258376ce65f811192db27f8"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directions", type=pathlib.Path)
    args = parser.parse_args()
    path = args.directions.resolve()
    require(path.is_file(), f"missing direction artifact: {path}")
    require(path.stat().st_size == EXPECTED_BYTES, "direction byte count mismatch")
    require(sha256(path) == EXPECTED_SHA256, "direction SHA-256 mismatch")

    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        required = {
            "dirs",
            "pair_deltas.layer_end",
            "activations.interleaved.layer_end",
        }
        require(required.issubset(keys), f"missing tensors: {sorted(required - keys)}")
        directions = handle.get_tensor("dirs").float()
        deltas = handle.get_tensor("pair_deltas.layer_end").float()
        activations = handle.get_tensor("activations.interleaved.layer_end").float()

    require(tuple(directions.shape) == (78, 6144), "direction shape mismatch")
    require(tuple(deltas.shape) == (32, 78, 6144), "pair delta shape mismatch")
    require(tuple(activations.shape) == (64, 78, 6144), "activation shape mismatch")
    require(bool(torch.isfinite(directions).all()), "non-finite direction")
    require(bool(torch.isfinite(deltas).all()), "non-finite pair delta")
    require(bool(torch.isfinite(activations).all()), "non-finite activation")

    reconstructed_deltas = activations[0::2] - activations[1::2]
    require(torch.equal(reconstructed_deltas, deltas), "stored deltas are not exact harmful-control differences")
    mean = deltas.double().mean(dim=0)
    norms = mean.norm(dim=1)
    require(bool((norms > 1e-12).all()), "invalid zero direction norm")
    reconstructed = (mean / norms[:, None]).float()
    max_error = float((reconstructed - directions).abs().max())
    require(torch.equal(reconstructed, directions), f"direction derivation mismatch; max error {max_error}")
    unit_error = float((directions.norm(dim=1) - 1).abs().max())

    result = {
        "passed": True,
        "pairs": 32,
        "layers": 78,
        "width": 6144,
        "delta": "harmful minus control",
        "mean_accumulator": "FP64",
        "normalization": "per-layer L2",
        "direction_max_absolute_error": max_error,
        "unit_norm_max_absolute_error": unit_error,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
