# VLAN

VLAN is a user-conditioned navigation benchmark and evaluation toolkit for multimodal models.

The codebase supports two task families:

- Action Prediction Navigation
  - `drone`
  - `quadrupedRobot`
- Language-Guided Navigation
  - `humanoid`
  - `blind`
  - `wheelchair`

The public release supports the reorganized dataset layout used in the benchmark release, while retaining compatibility with the older internal layout.

## Repository Layout

```text
VLAN/
  configs/
    models/
    prompts/
    runs/
  tools/
    build_targets.py
    build_contexts.py
    run_inference.py
    evaluate.py
    smoke_test_backend.py
  vlan_final/
    eval/
    models/
    prompts/
    tasks/
    utils/
    targets.py
  README.md
  LICENSE
  requirements.txt
```

## Dataset Layout

The released dataset is expected to use the following layout:

```text
test/<scene>/<traj>/
  KeyFrames/*.png
  metadata.json
  trajectory.json
  target.json
```

Where:

- `metadata.json` stores scene taxonomy and embedded keyframe metadata.
- `trajectory.json` stores obstacle metadata, trajectory data, and per-user accessibility labels.
- `target.json` stores canonical benchmark targets for all user types.

The code also contains compatibility logic for an older layout used during development.

## Tasks

### Language-Guided Navigation

- `humanoid`
- `blind`
- `wheelchair`

Notes:

- `humanoid` is the public task name.
- `pedestrian` is still accepted as a legacy alias for compatibility.
- `blind` is the public task name.
- older data fields such as `bvi` are mapped internally to `blind`.

### Action Prediction Navigation

- `drone`
- `quadrupedRobot`

These tasks predict future action sequences in the form:

```json
[{"dx": ..., "dz": ..., "yaw": ...}, ...]
```

## Installation

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Checkpoints

Model configuration files in `configs/models/` assume checkpoints are available under a local `ckpts/` directory.

Example:

```text
ckpts/Qwen2.5-VL-3B-Instruct
ckpts/Qwen2.5-VL-7B-Instruct
ckpts/Qwen3-VL-8B-Instruct
...
```

## Build Targets and Contexts

```bash
python tools/build_targets.py --data_root /path/to/test --artifacts_root artifacts
python tools/build_contexts.py --data_root /path/to/test --artifacts_root artifacts
```

## Run Inference

Example using the released task name `humanoid`:

```bash
python tools/run_inference.py \
  --config configs/runs/default.yaml \
  --task humanoid \
  --model-config configs/models/qwen2_5_vl_3b_instruct.yaml \
  --data-root /path/to/test \
  --artifacts-root artifacts
```

Legacy alias `pedestrian` is also supported.

## Run Evaluation

```bash
python tools/evaluate.py \
  --config configs/runs/default.yaml \
  --task humanoid \
  --model-config configs/models/qwen2_5_vl_3b_instruct.yaml \
  --data-root /path/to/test \
  --artifacts-root artifacts
```

Action tasks use the same entry point with `--task drone` or `--task quadrupedRobot`.

## Notes on Public Release

This repository intentionally excludes:

- raw datasets
- checkpoints
- generated predictions
- generated evaluations
- local experiment logs

Only code, configs, and lightweight documentation required to reproduce the benchmark pipeline are included.
