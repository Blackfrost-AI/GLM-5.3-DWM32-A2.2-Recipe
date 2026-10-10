# Agent contract: GLM-5.3 DWM32 alpha 2.2 exact-method replay

Read these files completely and in this order before proposing a command:

1. `recipe.lock.json`
2. `RUNBOOK.md`
3. `evidence/capture-receipt.json`
4. `evidence/original-run-receipt.json`
5. `assets/directions.asset.json`

This recipe is immutable with respect to method parameters. Alpha is `2.2`,
rank is one, the operation is one pass, edited layers are 2 through 77, and
layer 78 is excluded. Never substitute a direction bank, source revision,
reference revision, target surface, quantizer commit, scale policy, or
arithmetic rule and still call the result this recipe.

"Exact" in this package means the recorded mathematical method and identified
inputs. It does not promise a byte-identical checkpoint: the historical build
runtime and output shard hashes were not preserved. Read the reproducibility
section in `RUNBOOK.md` before execution.

Before execution, run `python3 scripts/verify_package.py`. Materialize the
private direction asset and rerun the verifier with `--directions`. Stop on
any checksum, metadata, count, shape, revision, or environment mismatch.

Ablagent Standalone Harness 1.0.0 can audit this repository through
`plans/ablagent-read-only-audit.json`. Its tools are read-only. A shell-capable
operator or a separately authorized future execution adapter must run the
commands in `RUNBOOK.md`.

The historical output passed offline structural verification only. Do not
describe the historical artifact or a replay as served, generation-tested,
behaviorally qualified, safe, or release-ready without new evidence from the
required validation stages.

