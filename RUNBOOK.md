# Agent runbook: GLM-5.3 BF16 DWM32 alpha 2.2

This is the ordered procedure for the locked alpha 2.2 method. Read
`AGENTS.md`, `recipe.lock.json`, `environment.lock.json`, and both receipts in
`evidence/` before running any command. A mismatch is a hard stop.

## 1. Execution boundary

This runbook performs an offline weight transformation and structural
verification. It does not authorize serving, evaluation, publication, or model
delivery. The historical output was not generation-tested. A new output stays
a candidate until it passes the applicable later gates.

Exact-method replay means the same identified inputs, direction tensor,
projection, target surface, and quantization policy. It does not mean a
byte-identical historical checkpoint because the original build runtime and
output shard hashes were not preserved.

The values under `capture_runtime` in `environment.lock.json` document the
direction-capture environment. They are not claimed as the missing historical
CPU build environment. The executor enforces the surviving ModelOpt and codec
hashes; record all other environment choices with the new run.

## 2. Required resources

- Python with PyTorch, safetensors, and NVIDIA ModelOpt available.
- ModelOpt commit `022767c7ab3d7d36211affd85e5c496770cde768` with:
  - `modelopt/__init__.py` SHA-256
    `96647188e0afa8dcdcadbe42178feeb4cc19113d9ffcbaf633bd62414bd6d453`;
  - `nvfp4_tensor.py` SHA-256
    `e128fd62f0d0c4aa179e057b53a615bccf42f30f8ed9ef7c8f7e3dc84c5eb61e`;
  - `fp8_tensor.py` SHA-256
    `ec9caf56fc140720a55732f827dbdd41b9fdf8fcb961a1e37028d4d4b69b7fae`.
- An authenticated `gh` CLI for the private direction asset.
- Network access to the two pinned model repositories.
- About 1.51 TB for the BF16 source snapshot.
- At least 565 GB free on the output volume. The executor reserves 100 GB
  beyond the expected 464,115,789,896 bytes of tensor data.
- Capacity for eight CPU worker processes, each fixed to eight Torch threads.

The executor disables CUDA. Capture is not repeated; the exact captured tensor
is consumed as a locked input.

## 3. Set isolated paths

Run from the recipe root. Put large model data on a volume with enough space.
Do not point `OUTPUT_ROOT` or `RUN_ROOT` inside `SOURCE_ROOT` or
`REFERENCE_ROOT`.

```bash
export RECIPE_ROOT="$PWD"
export SOURCE_ROOT="/data/glm53-bf16-9d2398f"
export REFERENCE_ROOT="/data/glm53-nvfp4-reference-e3b8f9f"
export DIRECTION_FILE="$RECIPE_ROOT/artifacts/glm53-nvfp4-raw32.directions.safetensors"
export OUTPUT_ROOT="/data/glm53-dwm32-a2p2-nvfp4"
export RUN_ROOT="/data/glm53-dwm32-a2p2-run"
```

Resolve and record these five paths in the operator record before continuing.

## 4. Verify the text package

```bash
python3 scripts/verify_package.py
```

Expected result: `passed` is `true`, the lock ID is
`glm-5.3-bf16-dwm32-a2p2-nvfp4`, the pair count is 32, and `SHA256SUMS` is
present and valid.

## 5. Acquire and verify the exact direction artifact

```bash
python3 scripts/materialize_directions.py --output "$DIRECTION_FILE"
python3 scripts/verify_package.py --directions "$DIRECTION_FILE"
python3 scripts/verify_direction_derivation.py "$DIRECTION_FILE"
```

All three commands must pass. The required artifact is exactly 193,612,804
bytes with SHA-256
`e12d8fee3e2f75dc7e269bc09c810f079eeb1abce258376ce65f811192db27f8`.
The derivation verifier must confirm 32 harmful-minus-control deltas, an FP64
mean, and independent L2 normalization for all 78 layer directions.

## 6. Materialize the pinned BF16 source

Use a Hugging Face credential if the repository requires one. Fetch the pinned
commit, never a moving branch:

```bash
hf download zai-org/GLM-5.3-BF16 \
  --revision 9d2398f478cab2de883137db3a36ad2c96205e24 \
  --local-dir "$SOURCE_ROOT"
```

Before continuing, confirm that the snapshot has 282 safetensors shards. The
executor checks the exact configuration and index hashes, 59,585 indexed
tensors, 282 shard names, and absence of a quantization configuration. Revision
pinning and `evidence/source-lfs-manifest.json` identify every shard by its Git
LFS SHA-256. Hash the complete local snapshot and write the receipt used by all
later executor modes:

```bash
python3 scripts/verify_source_snapshot.py \
  --source "$SOURCE_ROOT" \
  --receipt "$RUN_ROOT/source-verification.json" \
  --workers 4
```

Expected result: 282 shards, 1,506,667,387,408 bytes, and manifest canonical
SHA-256 `d7e8f6f8f6d6b03413e7fe5bdefa65c1a4b8d6dd19f30dd963498595ea8c84c2`.
The probe, build, and verify commands reject a missing receipt or any later
size/mtime change.

## 7. Build the pinned NVIDIA metadata reference

This step downloads the three locked metadata files and uses HTTP byte-range
requests to obtain each safetensors header and scalar prefix. It does not
download the full 464 GB NVIDIA checkpoint.

```bash
python3 scripts/prepare_reference.py \
  --reference-dir "$REFERENCE_ROOT" \
  --workers 6
```

Expected receipt values:

- revision `e3b8f9f2fad29727dfa95e766c5cac18ea1414bf`;
- 50 reference shards;
- 232,403 indexed tensors;
- 464,115,789,896 tensor-data bytes; and
- 115,218 scalar values.

Keep `REFERENCE_RECEIPT.json`, every `*.header.json`, and
`calibrated-scalars.json` together in `REFERENCE_ROOT`.

## 8. Run the CPU-only probe

```bash
python3 -s scripts/run_locked.py probe \
  --source "$SOURCE_ROOT" \
  --reference-dir "$REFERENCE_ROOT" \
  --directions "$DIRECTION_FILE" \
  --output "$OUTPUT_ROOT" \
  --run-dir "$RUN_ROOT"
```

The probe checks representative dense, expert, attention, shared-expert,
gate/up, FP8, and NVFP4 paths at alpha 0 and 2.2. It also enforces the two
ModelOpt source-file hashes. Continue only if `preflight-probes.json` records
`passed: true` and `cuda_initialized: false`.

## 9. Build once

```bash
python3 -s scripts/run_locked.py build \
  --source "$SOURCE_ROOT" \
  --reference-dir "$REFERENCE_ROOT" \
  --directions "$DIRECTION_FILE" \
  --output "$OUTPUT_ROOT" \
  --run-dir "$RUN_ROOT"
```

The command has no alpha, layer, pass, worker, or target override. It writes a
recipe binding before the first shard. Resume is allowed only when completed
shard receipts carry the identical binding. Never reuse a partial output or run
directory from another recipe.

The build performs a structural verification before returning. A failure from
an assertion, checksum, tensor count, shape, dtype, scale, projection metric,
source snapshot, or free-space check invalidates the run.

## 10. Repeat offline verification explicitly

```bash
python3 -s scripts/run_locked.py verify \
  --source "$SOURCE_ROOT" \
  --reference-dir "$REFERENCE_ROOT" \
  --directions "$DIRECTION_FILE" \
  --output "$OUTPUT_ROOT" \
  --run-dir "$RUN_ROOT"

python3 scripts/verify_scalar_layout.py \
  --output "$OUTPUT_ROOT" \
  --reference-dir "$REFERENCE_ROOT" \
  --run-dir "$RUN_ROOT"
```

Required final counts:

| Check | Required value |
|---|---:|
| Output shards | 50 |
| Indexed tensors | 232,403 |
| Edited tensors | 19,352 |
| MTP tensors byte-equal | 791 |
| NVIDIA activation scales exact | 57,609 |
| Positive finite serialized scales | 115,218 |
| Exact gate/up weight-scale pairs | 19,203 |
| CUDA initialized | false |

Retain `build-plan.json`, `build-complete.json`, `preflight-probes.json`, every
shard report, `serialized-scale-verification.json`, `RECIPE_BINDING.json`, and
the two verification files copied into the output.

## 11. Create a new output hash manifest

The historical run lacked an output shard manifest. Every replay must close
that gap:

```bash
(cd "$OUTPUT_ROOT" && \
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 shasum -a 256) \
  > "$RUN_ROOT/output.SHA256SUMS"
shasum -a 256 "$RUN_ROOT/output.SHA256SUMS" \
  > "$RUN_ROOT/output.SHA256SUMS.sha256"
```

Treat these manifests and the complete run directory as part of the candidate.

## 12. Stop point and reporting language

At the end of this runbook, report only an **offline structurally verified,
method-equivalent candidate**. Record that load, generation, capability,
red-line, and deployment tests remain outstanding unless separate evidence
proves otherwise.

Do not describe the candidate as behaviorally qualified, release-ready, safe,
served, or byte-identical to the historical build on the strength of this
runbook alone.

## Ablagent Standalone Harness 1.0.0

The included plan gives an Ablagent-compatible model a bounded audit path:

```bash
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config /path/to/config.local.json \
  --workspace "$RECIPE_ROOT" \
  --plan plans/ablagent-read-only-audit.json
```

Harness 1.0.0 is read-only, so a successful audit does not execute any command
above. The optional paired model is `Alblagent-9B`; use it when its endpoint is
available. The locked recipe itself does not depend on that model.
