# Third-party notices

## GLM-5.3 BF16

This recipe expects a separately acquired copy of
`zai-org/GLM-5.3-BF16@9d2398f478cab2de883137db3a36ad2c96205e24`.
The checkpoint and related files are governed by their upstream terms.

## NVIDIA GLM-5.3 NVFP4

The recipe reads public metadata and calibration scalars from
`nvidia/GLM-5.3-NVFP4@e3b8f9f2fad29727dfa95e766c5cac18ea1414bf`.
Those materials remain governed by NVIDIA and upstream model terms.

## NVIDIA ModelOpt

The historical run used ModelOpt source build `0.48.0dev-022767c7`, commit
`022767c7ab3d7d36211affd85e5c496770cde768`. ModelOpt is a separately
installed dependency. This repository does not vendor its source.

## Ablagent Standalone Harness

The compatible audit harness is a separate Blackfrost AI Apache-2.0 work.
It is not copied into this repository. Version 1.0.0 is a read-only agent
runner and cannot execute the weight operation.

## Hugging Face Hub client

`scripts/generate_lfs_manifests.py` optionally uses the separately installed
`huggingface_hub` Python package to regenerate pinned Git LFS metadata. Normal
package verification and execution do not require that maintenance utility.

