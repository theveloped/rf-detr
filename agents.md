# agents.md -- AI Agent Context

This file provides context for AI coding agents working on this project. It documents infrastructure details, hard-won lessons, and architectural decisions that are not obvious from the code alone.

## Project Overview

This project fine-tunes [RF-DETR](https://github.com/roboflow/rf-detr) (RFDETRBase) for object detection on industrial assembly components. Training happens inside Docker on a remote GPU server. Inference runs locally on Windows.

**5 detection classes**: DM, HOUSING, EXCHANGER, CORE, SHAFT.

## Infrastructure

### Remote Training Server

- **Host**: `aime@10.103.200.201`
- **SSH key**: `C:\Users\TobiasScheepers\.ssh\id_ed25519`
- **Password**: `a4000226n5`
- **GPU**: RTX A6000 48GB (use GPU 1 -- GPU 0 is reserved for SAM/ollama containers)
- **Docker**: Training runs via `bash run_training.sh`, which builds the image and launches on GPU 1
- **Repo on server**: `~/rf-detr/` (cloned from GitHub)

### Local Machine (Windows)

- **Working directory**: `C:\Users\TobiasScheepers\OneDrive - Wefabricate\Bureaublad\rf-detr\`
- **Python**: Anaconda distribution, Python 3.12
- **RF-DETR package**: `C:\ProgramData\anaconda3\Lib\site-packages\rfdetr\`
- **Git remote**: `https://github.com/theveloped/rf-detr.git`

### Workflow for Code Changes

The preferred workflow is git, not SCP for code:

1. Edit files locally
2. `git push` from the local machine
3. `git pull` on the remote server
4. Run training on the server

Datasets are too large for git and must be transferred via `scp`.

## Critical Technical Details

### File Transfer: Use SCP, Not rsync

`rsync` from Windows PowerShell to this server fails with "connection unexpectedly closed" due to an incompatibility between Windows OpenSSH and MSYS2 rsync. SSH itself works fine. **Always use `scp` for file transfers.**

```powershell
# This works:
scp -i ~/.ssh/id_ed25519 -r datasets/aruco_and_tray_1 aime@10.103.200.201:~/rf-detr/datasets/

# This does NOT work:
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" datasets/ aime@10.103.200.201:~/rf-detr/datasets/
```

### Dockerfile: No python3.12-distutils

`python3.12-distutils` does not exist as a package (distutils was removed in Python 3.12). Do not add it to the Dockerfile.

### RF-DETR Requires a Test Split

RF-DETR expects `train/`, `valid/`, AND `test/` directories, each containing `_annotations.coco.json`. The training script creates the test split as a copy of the validation split.

### Category ID Ordering (CRITICAL)

The `labelme2coco.convert()` function assigns category IDs in the order it first encounters labels while iterating through alphabetically-sorted annotation files. This ordering is **non-deterministic** from the perspective of the training script -- it depends on which label appears first in whichever annotation file sorts first alphabetically.

**This caused a label-swap bug**: the inference script hardcoded `CLASS_NAMES` in one order, but the model was trained with categories in a different order, causing CORE/SHAFT/EXCHANGER to appear swapped.

**The fix** (already implemented):

1. `train.py` uses the lower-level `get_coco_from_labelme_folder()` API with a predefined `coco_category_list` to force deterministic ordering.
2. `train.py` asserts the produced categories match `CLASS_NAMES` after conversion.
3. `train.py` saves `class_names.json` alongside the checkpoints.
4. `inference.py` loads `class_names.json` from next to the checkpoint file at runtime.

**Never rely on hardcoded class name lists without a `class_names.json` to verify.**

### Model Loading: Constructor, Not .load()

`RFDETRBase` has **no `.load()` method**. Fine-tuned checkpoints must be passed via the constructor:

```python
# CORRECT:
model = RFDETRBase(pretrain_weights="path/to/checkpoint_best_ema.pth")

# WRONG -- this method does not exist:
model = RFDETRBase()
model.load("path/to/checkpoint_best_ema.pth")
```

The model automatically detects the number of classes from the checkpoint and reinitializes the detection head accordingly.

### RFDETRBaseConfig Validation

`RFDETRBaseConfig` (in `rfdetr/config.py`) uses Pydantic with `extra="forbid"`. You cannot pass arbitrary keyword arguments. Key parameters:

- `pretrain_weights` -- path to checkpoint file
- `num_classes` -- default 90 (auto-detected from checkpoint)
- `resolution` -- default 560
- `num_queries` -- default 300

### Checkpoint Naming

Training produces `checkpoint_best_ema.pth` and `checkpoint_best_regular.pth` (NOT `checkpoint_best_total.pth` as the RF-DETR docs suggest). The EMA checkpoint is the primary one for inference.

### Model Optimization Warning

When running inference, RF-DETR may warn: "Model is not optimized for inference" and suggest calling `model.optimize_for_inference()`. This hasn't been addressed yet but could improve latency.

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Full training pipeline: merge datasets, convert LabelMe to COCO, split, train |
| `inference.py` | CLI inference script with model loading, detection, and annotated image output |
| `Dockerfile` | CUDA 12.1 + Python 3.12 Docker image for training |
| `run_training.sh` | Builds Docker image and launches training on GPU 1 |
| `requirements.txt` | Python dependencies (rfdetr, labelme2coco, supervision, etc.) |
| `.gitignore` | Excludes datasets/, training_output/, inference_output/, *.pth |

## RF-DETR Package Internals (for reference)

These are installed package paths on the local Windows machine:

| File | Key Contents |
|------|-------------|
| `rfdetr/detr.py` | `RFDETRBase` class (line ~500), parent `RFDETR` class (line ~63) |
| `rfdetr/config.py` | `RFDETRBaseConfig` (line ~89), `ModelConfig` (line ~48) |
| `rfdetr/main.py` | `Model.__init__` where checkpoint loading happens (line ~82) |

## Datasets

3 source datasets, each containing PNGs with LabelMe JSON annotations:

| Dataset | Prefix | Images |
|---------|--------|--------|
| `aruco_and_tray_1` | `at1_` | ~134 images |
| `fixed_belt_view_1` | `fbv1_` | Varies |
| `mixed_tray_belt_1` | `mtb1_` | Varies |

Additional dataset directories exist locally (`aruco_and_tray_2`, `aruco_and_tray_3`, `aruco_calibration_1`, `aruco_calibration_2`, `mixed_tray_belt_1`) but only the 3 listed above are used for training.

## Training Results (First Run)

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.936 |
| mAP@0.50 | 0.997 |
| mAP@0.75 | 0.985 |
| Early stopping | ~epoch 51 of 60 |
| Training time | ~20 minutes |

## Local Anaconda Environment Notes

The local Anaconda environment has package version constraints:

- `numpy` is stuck at 1.26.4 due to numba/pywavelets/streamlit constraints
- `matplotlib` needed upgrade from 3.8 to 3.10
- `scipy` needed upgrade from 1.16 to 1.17
- `scikit-learn` needed upgrade from 1.4 to 1.8
- `pandas` needed upgrade from 2.2 to 3.0

Be cautious when installing or upgrading packages -- version conflicts are common in this environment.

## Common Operations

### Retrain with new data

1. Add new labeled images to the appropriate `datasets/` directory
2. `scp` the new images to the server
3. SSH in and `bash run_training.sh`
4. `scp` back the new checkpoints and `class_names.json`

### Add a new class

1. Add the class name to `CLASS_NAMES` in `train.py`
2. Label images with the new class name in LabelMe
3. Retrain
4. The `class_names.json` will automatically include the new class
5. `inference.py` picks it up automatically via `class_names.json`

### Test on new images

```bash
python inference.py path/to/new/images/ --threshold 0.3
```
