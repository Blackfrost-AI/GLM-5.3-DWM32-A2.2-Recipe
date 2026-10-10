---
name: glm53-dwm32-a2p2-locked-method
description: Audit or execute Blackfrost AI's locked GLM-5.3 BF16-source DWM32 alpha-2.2 method, including exact asset verification, streamed ModelOpt serialization, and offline structural verification.
---

# GLM-5.3 DWM32 alpha 2.2 locked method

1. Read `AGENTS.md`, `recipe.lock.json`, and `RUNBOOK.md` in full.
2. Treat every method field in `recipe.lock.json` as immutable.
3. Run `python3 scripts/verify_package.py` before using external inputs.
4. Acquire the exact direction tensor described by
   `assets/directions.asset.json`; verify it with `--directions`.
5. Use `scripts/prepare_reference.py` to build the pinned NVIDIA metadata and
   scalar cache.
6. Execute probe, build, verify, and scalar verification in the exact order in
   `RUNBOOK.md`.
7. Report structural results and untested behavior separately. Never promote
   a candidate from this recipe without fresh serving and evaluation evidence.
8. Describe a new output as method-equivalent. Do not claim byte identity with
   the historical output because its shard hashes and complete build runtime
   were not preserved.

For Ablagent Standalone Harness 1.0.0, run the bundled read-only audit plan.
That harness reviews the recipe; it does not execute shell or weight edits.

