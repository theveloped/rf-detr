# RF-DETR Fine-Tuning Pipeline

Fine-tuning [RF-DETR](https://github.com/roboflow/rf-detr) (Real-time Foundation DETR) for object detection on industrial assembly components. Training runs inside Docker on a remote GPU server; inference runs locally on Windows.

## Classes

The model detects 5 object classes:

| ID | Class | Description |
|----|-------|-------------|
| 0 | DM | DM component |
| 1 | HOUSING | Housing component |
| 2 | EXCHANGER | Exchanger component |
| 3 | CORE | Core component |
| 4 | SHAFT | Shaft component |

## Project Structure

```
rf-detr/
  Dockerfile           # Docker image for training (CUDA 12.1 + Python 3.12)
  requirements.txt     # Python dependencies for training
  run_training.sh      # Builds Docker image and launches training on GPU
  train.py             # Full training pipeline (merge, convert, split, train)
  inference.py         # Local inference script with CLI arguments
  datasets/            # Image datasets (not in git, transferred via SCP)
    aruco_and_tray_1/  # PNGs + LabelMe JSON annotations
    fixed_belt_view_1/
    mixed_tray_belt_1/
  training_output/     # Downloaded checkpoints from server (not in git)
    checkpoint_best_ema.pth     # Best model checkpoint (EMA weights)
    checkpoint_best_regular.pth # Best model checkpoint (regular weights)
    class_names.json            # Authoritative class ID -> name mapping
  inference_output/    # Annotated detection images (not in git)
```

## Prerequisites

### Remote server (training)

- **Host**: `aime@10.103.200.201`
- **SSH key**: `~/.ssh/id_ed25519`
- **GPU**: RTX A6000 48GB (GPU 1)
- **Docker** with NVIDIA runtime

### Local machine (inference)

- Python 3.10+ (Anaconda recommended)
- Packages: `rfdetr`, `supervision`, `Pillow`

```bash
pip install rfdetr supervision Pillow
```

---

## Step 1: Label the Data

Images are annotated using [LabelMe](https://github.com/labelmeai/labelme) with bounding box annotations. Each image gets a corresponding `.json` file in the same directory.

The 3 source datasets are:

| Dataset | Prefix | Description |
|---------|--------|-------------|
| `aruco_and_tray_1` | `at1_` | Tray images with ArUco markers |
| `fixed_belt_view_1` | `fbv1_` | Fixed camera belt view images |
| `mixed_tray_belt_1` | `mtb1_` | Mixed tray and belt images |

Each dataset directory contains PNG images and their LabelMe JSON annotation files:

```
datasets/aruco_and_tray_1/
  image_000.png
  image_000.json   # LabelMe annotation
  image_001.png
  image_001.json
  ...
```

Label names in the LabelMe JSONs must exactly match the class names defined in `train.py`:
`DM`, `HOUSING`, `EXCHANGER`, `CORE`, `SHAFT`.

## Step 2: Transfer Data to the Remote Server

### Clone the repo on the server

```bash
ssh -i ~/.ssh/id_ed25519 aime@10.103.200.201

# On the server:
git clone https://github.com/theveloped/rf-detr.git
cd rf-detr
mkdir -p datasets
```

### Transfer datasets via SCP

From your local Windows machine, use `scp` to transfer the dataset directories. **Do not use `rsync`** -- it does not work reliably from Windows PowerShell to this server.

```powershell
# Transfer each dataset directory
scp -i ~/.ssh/id_ed25519 -r datasets/aruco_and_tray_1 aime@10.103.200.201:~/rf-detr/datasets/
scp -i ~/.ssh/id_ed25519 -r datasets/fixed_belt_view_1 aime@10.103.200.201:~/rf-detr/datasets/
scp -i ~/.ssh/id_ed25519 -r datasets/mixed_tray_belt_1 aime@10.103.200.201:~/rf-detr/datasets/
```

### Push code changes via git

When you modify `train.py`, `Dockerfile`, etc., push from your local machine and pull on the server:

```powershell
# Local:
git add . && git commit -m "message" && git push

# Server:
cd ~/rf-detr && git pull
```

## Step 3: Train on the Remote Server

SSH into the server and run the training script:

```bash
ssh -i ~/.ssh/id_ed25519 aime@10.103.200.201
cd ~/rf-detr
bash run_training.sh
```

### What `run_training.sh` does

1. Verifies all 3 dataset directories exist
2. Builds the Docker image from `Dockerfile`
3. Runs the container on GPU 1 with datasets mounted read-only

### What `train.py` does inside Docker

1. **Merge datasets** -- Copies all labeled images from the 3 source directories into a single directory, prefixing filenames (`at1_`, `fbv1_`, `mtb1_`) to avoid collisions. Validates all images and removes any corrupted PNGs.
2. **Convert to COCO** -- Uses `labelme2coco` to convert LabelMe annotations to COCO JSON format. Category IDs are explicitly forced via a predefined `coco_category_list` to guarantee deterministic ordering.
3. **Train/val/test split** -- 85% train / 15% validation, with test set as a copy of validation (RF-DETR requires all three splits). Creates the Roboflow-style directory structure with `_annotations.coco.json` per split.
4. **Train RF-DETR** -- Fine-tunes `RFDETRBase` with the configuration in `TRAIN_CONFIG`.

### Training configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 60 | Early stopping at patience 15 |
| Batch size | 16 | Fits in 48GB VRAM |
| Learning rate | 1e-4 | |
| Encoder LR | 1.5e-4 | |
| EMA | Enabled | Produces `checkpoint_best_ema.pth` |
| Multi-scale | Enabled | |
| Seed | 42 | Reproducible |

### Training results (first run)

| Metric | Value |
|--------|-------|
| mAP@0.50:0.95 | 0.936 |
| mAP@0.50 | 0.997 |
| mAP@0.75 | 0.985 |
| Early stopping | ~epoch 51 |
| Training time | ~20 minutes |

### Training output files

```
training_output/
  checkpoint_best_ema.pth       # Best checkpoint (EMA) -- use this for inference
  checkpoint_best_regular.pth   # Best checkpoint (regular weights)
  checkpoint.pth                # Latest checkpoint
  checkpoint0004.pth            # Interval checkpoints (every 5 epochs)
  checkpoint0009.pth
  ...
  class_names.json              # Class ID -> name mapping
  log.txt                       # Training log
  results.json                  # Evaluation metrics
  metrics_plot.png              # Training curves
  eval/                         # Per-epoch evaluation results
```

## Step 4: Download Checkpoints

From your local Windows machine, download the EMA checkpoint and class mapping:

```powershell
mkdir training_output

# Download the best checkpoint (EMA weights)
scp -i ~/.ssh/id_ed25519 aime@10.103.200.201:~/rf-detr/training_output/checkpoint_best_ema.pth ./training_output/

# Download the class name mapping
scp -i ~/.ssh/id_ed25519 aime@10.103.200.201:~/rf-detr/training_output/class_names.json ./training_output/

# (Optional) Download all checkpoints and logs
scp -i ~/.ssh/id_ed25519 -r aime@10.103.200.201:~/rf-detr/training_output/ ./
```

## Step 5: Run Inference Locally

### Basic usage

```bash
# Single image
python inference.py path/to/image.png

# Entire directory of images
python inference.py ./datasets/aruco_and_tray_1/

# Multiple images
python inference.py image1.png image2.png image3.png
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--checkpoint` | Path to model checkpoint | `./training_output/checkpoint_best_ema.pth` |
| `--threshold` | Minimum confidence score | `0.5` |
| `--output-dir` | Directory for annotated output images | `./inference_output` |
| `--no-save` | Only print results, skip saving images | Off |

### Examples

```bash
# Lower threshold to see more detections
python inference.py image.png --threshold 0.3

# Use a different checkpoint
python inference.py image.png --checkpoint ./training_output/checkpoint_best_regular.pth

# Save to a custom output directory
python inference.py ./datasets/ --output-dir ./my_results/

# Print results only, no annotated images
python inference.py image.png --no-save
```

### Output

The script produces:

1. **Console output** -- Each detection with class name, confidence score, and bounding box coordinates
2. **Annotated images** -- Saved to `./inference_output/` with bounding boxes and labels drawn on top

### Class name resolution

The inference script loads class names from `class_names.json` located next to the checkpoint file. If the file is not found, it falls back to a hardcoded default. Always make sure to download `class_names.json` alongside the checkpoint to ensure correct label mapping.

---

## Troubleshooting

### `rsync` fails from Windows

`rsync` does not work from Windows PowerShell to this server (pipe incompatibility between Windows OpenSSH and MSYS2 rsync). Use `scp` instead.

### Corrupted images crash training

`train.py` automatically validates all PNGs after merging and removes any truncated/corrupted files before COCO conversion.

### Class labels appear swapped in inference

This happens when the `CLASS_NAMES` list in `inference.py` doesn't match the category ID ordering produced during training. The fix is already in place:

- **Training**: `train.py` forces a deterministic category ordering via `coco_category_list` and asserts the result matches `CLASS_NAMES`. It also saves `class_names.json` alongside the checkpoints.
- **Inference**: `inference.py` loads `class_names.json` from next to the checkpoint instead of relying on a hardcoded list.

If you suspect a mismatch, check the `categories` array in any `_annotations.coco.json` file from the training dataset.

### `python3.12-distutils` not found during Docker build

This package was removed in Python 3.12. It is not included in the Dockerfile.

### RF-DETR requires a test split

RF-DETR expects `train/`, `valid/`, and `test/` directories each containing `_annotations.coco.json`. The training script creates the test split as a copy of the validation split.

### Model loading: no `.load()` method

Do **not** call `model.load(...)`. The fine-tuned checkpoint must be passed via the constructor:

```python
from rfdetr import RFDETRBase
model = RFDETRBase(pretrain_weights="path/to/checkpoint_best_ema.pth")
```
