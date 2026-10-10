# GLM-5.3 BF16 DWM32 Alpha 2.2 Recipe

This repository isolates the **non-Flash GLM-5.3 alpha 2.2 run** selected by
Blackfrost AI. It is an agent-readable, fail-closed recipe for applying the
same recorded rank-one Domain Weight Modification (DWM) to the same weight
surface and serializing the result in the same NVIDIA GLM-5.3 NVFP4 layout.

Copyright 2026 Blackfrost AI. Blackfrost-authored code, documentation, plans,
and pair data in this repository are licensed under Apache-2.0. Upstream model
weights and third-party software keep their own licenses; see
[`MODEL-NOTICE.md`](MODEL-NOTICE.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## What is locked

| Item | Locked value |
|---|---|
| Clean source | `zai-org/GLM-5.3-BF16@9d2398f478cab2de883137db3a36ad2c96205e24` |
| Flash variant | Excluded; this recipe uses the BF16 source above |
| Direction capture source | `nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf` |
| Pair bank | 32 matched harmful/control pairs, 64 frozen token rows |
| Alpha | `2.2` |
| Rank and passes | Rank one, one pass |
| Main layers edited | `2` through `77` inclusive |
| Exclusions | Layers `0–1`; MTP layer `78` |
| Edited tensors | 76 attention output, 1 dense down, 19,200 expert down, 75 shared-expert down; 19,352 total |
| Arithmetic | FP32 projection, then BF16 rounding, then layout-specific quantization |
| ModelOpt | `0.48.0dev-022767c7`, commit `022767c7ab3d7d36211affd85e5c496770cde768` |
| Direction SHA-256 | `e12d8fee3e2f75dc7e269bc09c810f079eeb1abce258376ce65f811192db27f8` |

The machine-readable authority is [`recipe.lock.json`](recipe.lock.json).
Agents must stop if a locked value differs. Alpha, layer range, target surface,
direction bank, source revisions, arithmetic, and scale policy have no runtime
override in the executor.

## The exact 32-pair input

The human-readable pair set is
[`assets/raw32.pairs.json`](assets/raw32.pairs.json). The actually captured raw
token rows are [`assets/raw32.frozen.jsonl`](assets/raw32.frozen.jsonl). Plain
harmful and control columns are also retained as
[`assets/harmful.txt`](assets/harmful.txt) and
[`assets/control.txt`](assets/control.txt).

Each pair is ordered `harmful`, then its matched `control`. Capture used the raw
tokenizer with special tokens enabled. It used no chat template, system prompt,
generation prompt, or generated tokens. The recorded activation is the last
real prompt token after the post-layer residual sum for each of 78 main layers.

For pair \(i\) and layer \(l\), the direction is:

```text
delta[i,l] = activation_harmful[i,l] - activation_control[i,l]
mean[l]    = FP64_mean_i(delta[i,l])
direction[l] = mean[l] / L2_norm(mean[l])
```

There is no PCA, SVD, orthogonalization, or later sign anchoring. The
harmful-minus-control order defines the sign. The private direction artifact
also contains the captured activations and pair deltas, so
[`scripts/verify_direction_derivation.py`](scripts/verify_direction_derivation.py)
can independently recompute and check all 78 directions.

## The weight operation

For direction \(r_l\) and every selected matrix \(W\) in layer \(l\), the
executor performs one FP32 rank-one projection and rounds the result to BF16:

```text
W' = BF16(W.float() - 2.2 * r_l[:, None] * (r_l @ W.float())[None, :])
```

It applies the operator only to:

- `self_attn.o_proj.weight` in layers 2–77;
- the dense MLP `down_proj.weight` in layer 2;
- every routed expert `down_proj.weight` in layers 3–77; and
- every shared expert `down_proj.weight` in layers 3–77.

The recipe performs no norm restoration, iterative ablation, alpha search, or
post-quantization correction. It recomputes weight scales after the edit.
Gate/up tensors share their per-expert joint maximum. It reuses exactly 57,609
NVIDIA activation scales and performs no fresh activation calibration.

## Why the output uses NVIDIA layout

The clean source is the BF16 checkpoint. The NVIDIA checkpoint supplies the
50-shard tensor layout, quantization metadata, and original activation scales.
Dense MLP layers 0–2 follow the reference FP8 layout. Routed expert weights in
layers 3–77 use NVFP4 E2M1 values with E4M3 block scales and block size 16.
Other retained tensors remain BF16 or F32 according to the reference layout.

[`scripts/run_locked.py`](scripts/run_locked.py) streams from BF16 source
shards and never creates a second full edited BF16 checkpoint. CUDA is hidden
for this stage; the ModelOpt codecs run on CPU. Eight worker processes with
eight Torch threads each are fixed to match the recorded implementation.

## Repository map

| Path | Purpose |
|---|---|
| `RUNBOOK.md` | Exact ordered operator procedure |
| `recipe.lock.json` | Immutable model, math, geometry, count, and hash contract |
| `environment.lock.json` | Recorded capture runtime and surviving build pins |
| `assets/` | Exact pair/token material and private asset descriptor |
| `evidence/` | Sanitized capture and historical structural receipts |
| `scripts/verify_package.py` | Package, lock, pair, token, and optional direction verifier |
| `scripts/materialize_directions.py` | Authenticated private Release asset downloader |
| `scripts/prepare_reference.py` | Pinned reference header and scalar range fetcher |
| `scripts/verify_source_snapshot.py` | Full 282-shard BF16 SHA-256 verifier and receipt writer |
| `scripts/run_locked.py` | Locked projection and serialization executor |
| `scripts/verify_scalar_layout.py` | Serialized scale and gate/up equality verifier |
| `plans/ablagent-read-only-audit.json` | Ablagent Standalone Harness 1.0.0 audit plan |

The repository contains no GLM weights and does not track the 193.6 MB
direction tensor. The direction tensor is distributed as a private GitHub
Release asset and is always checked by byte count and SHA-256 before use.
The retained derivation receipt records a second, independent bitwise
reconstruction of its pair deltas and all 78 normalized directions.

## Quick integrity audit

```bash
python3 scripts/verify_package.py
python3 scripts/materialize_directions.py \
  --output artifacts/glm53-nvfp4-raw32.directions.safetensors
python3 scripts/verify_package.py \
  --directions artifacts/glm53-nvfp4-raw32.directions.safetensors
python3 scripts/verify_direction_derivation.py \
  artifacts/glm53-nvfp4-raw32.directions.safetensors
```

The download command requires an authenticated GitHub CLI account with access
to this private repository. For the complete source preparation, probe, build,
and verification sequence, follow [`RUNBOOK.md`](RUNBOOK.md) without reordering
or replacing steps.

## Ablagent Standalone Harness integration

Ablagent Standalone Harness 1.0.0 can read and audit this package as a bounded
workspace. Copy the sample config, point it at an OpenAI-compatible endpoint,
and run the included plan:

```bash
cp config.ablagent.example.json config.local.json
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" --probe
/path/to/Ablagent-Standalone-Harness-1.0.0/bin/ablagent \
  --config "$PWD/config.local.json" \
  --workspace "$PWD" \
  --plan plans/ablagent-read-only-audit.json
```

The optional paired model is `Alblagent-9B`; its name and endpoint are ordinary
config values, so the recipe is usable while that model is still being tuned.
Harness 1.0.0 exposes only `read_file`, `grep`, and `glob`. It audits the lock,
runbook, executor, assets, and evidence. It cannot launch shell commands or edit
weights. A shell-capable operator or a future separately authorized execution
adapter must perform the commands in the runbook.

For the optional `Alblagent-9B` pairing, the validated SGLang server uses the
`mimo` reasoning parser and `qwen3_coder` tool-call parser. Thinking and
parallel tool calls are disabled, and the harness uses non-streaming requests.
Follow the harness's own `docs/SERVER.md` for the current launch details.

## Evidence and claim boundary

The historical alpha 2.2 build completed offline structural verification:

- 50 output shards and 232,403 indexed tensors;
- all 19,352 intended targets counted;
- all 791 MTP tensors byte-equal to source;
- 1,825 unmodified BF16/F32 tensors byte-equal to source;
- all 57,609 NVIDIA activation scales preserved; and
- 115,218 serialized scales finite and positive, with 19,203 exact gate/up
  weight-scale pairs.

That checkpoint was never loaded, served, generation-tested, transferred to a
B200, or behaviorally qualified. The capture establishes a reproducible
prompt-content contrast; it does not by itself establish response-level refusal
causality.

This package supports **exact-method, method-equivalent reproduction**. It does
not promise byte-identical output from the historical build. The original full
build runtime, launch command, output shards, and output shard hash manifest
were not preserved. Every new run must emit and retain its own output manifest,
then proceed through fresh load, generation, capability, and red-line gates
before any release claim.

## License

See [`LICENSE`](LICENSE) for Apache-2.0 and [`NOTICE`](NOTICE) for the Blackfrost
AI copyright notice. Model weights are separate works and are not included.
