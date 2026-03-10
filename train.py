#!/usr/bin/env python3
"""
RF-DETR Fine-Tuning Training Script
====================================
Merges 3 LabelMe-annotated datasets, converts to COCO format,
performs train/val split, and trains RFDETRBase.

Expected directory layout (inside Docker container):
  /workspace/datasets/aruco_and_tray_1/   (PNGs + LabelMe JSONs)
  /workspace/datasets/fixed_belt_view_1/  (PNGs + LabelMe JSONs)
  /workspace/datasets/mixed_tray_belt_1/  (PNGs + LabelMe JSONs)

Output:
  /workspace/training_output/  (checkpoints, logs)
"""

import json
import os
import random
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS = {
    "aruco_and_tray_1": "at1_",
    "fixed_belt_view_1": "fbv1_",
    "mixed_tray_belt_1": "mtb1_",
}

CLASS_NAMES = ["DM", "HOUSING", "EXCHANGER", "CORE", "SHAFT"]

# Paths (inside Docker container)
WORKSPACE = Path("/workspace")
DATASETS_DIR = WORKSPACE / "datasets"
MERGED_DIR = WORKSPACE / "labelme_merged"
COCO_OUTPUT_DIR = WORKSPACE / "coco_output"
DATASET_DIR = WORKSPACE / "dataset"  # Final RF-DETR dataset directory
OUTPUT_DIR = WORKSPACE / "training_output"

# Training hyperparameters (optimized for RTX A6000 48GB)
TRAIN_CONFIG = {
    "epochs": 60,
    "batch_size": 16,
    "grad_accum_steps": 1,  # Not needed with batch_size=16 on 48GB VRAM
    "lr": 1e-4,
    "lr_encoder": 1.5e-4,
    "use_ema": True,
    "num_workers": 8,  # 48 CPU cores available
    "multi_scale": True,
    "early_stopping": True,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 0.001,
    "early_stopping_use_ema": True,
    "tensorboard": True,
    "log_per_class_metrics": True,
    "checkpoint_interval": 5,
    "seed": 42,
}

VAL_SPLIT = 0.15  # 15% for validation
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Step 1: Merge datasets with filename prefixes
# ---------------------------------------------------------------------------
def merge_datasets():
    """Merge 3 LabelMe datasets into one directory, prefixing filenames."""
    print("\n" + "=" * 60)
    print("STEP 1: Merging datasets with filename prefixes")
    print("=" * 60)

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)

    total_images = 0
    total_labeled = 0

    for dataset_name, prefix in DATASETS.items():
        dataset_path = DATASETS_DIR / dataset_name
        if not dataset_path.exists():
            print(f"  WARNING: Dataset directory not found: {dataset_path}")
            continue

        print(f"\n  Processing: {dataset_name} (prefix: {prefix})")

        # Find all PNG files
        png_files = sorted(dataset_path.glob("*.png"))
        print(f"    Found {len(png_files)} PNG files")

        for png_file in png_files:
            stem = png_file.stem  # e.g., "image_001"
            json_file = dataset_path / f"{stem}.json"

            # Only copy images that have LabelMe annotations
            if not json_file.exists():
                continue

            new_stem = f"{prefix}{stem}"

            # Copy PNG with prefix
            dst_png = MERGED_DIR / f"{new_stem}.png"
            shutil.copy2(png_file, dst_png)

            # Copy and update JSON with prefix
            with open(json_file, "r") as f:
                label_data = json.load(f)

            # Update imagePath to match new filename
            label_data["imagePath"] = f"{new_stem}.png"

            dst_json = MERGED_DIR / f"{new_stem}.json"
            with open(dst_json, "w") as f:
                json.dump(label_data, f, indent=2)

            total_labeled += 1

        total_images += len(png_files)

    print(f"\n  Total labeled images merged: {total_labeled}")
    print(f"  Merged directory: {MERGED_DIR}")
    return total_labeled


# ---------------------------------------------------------------------------
# Step 2: Convert LabelMe to COCO format
# ---------------------------------------------------------------------------
def convert_to_coco():
    """Convert merged LabelMe annotations to COCO JSON format."""
    print("\n" + "=" * 60)
    print("STEP 2: Converting LabelMe to COCO format")
    print("=" * 60)

    import labelme2coco

    if COCO_OUTPUT_DIR.exists():
        shutil.rmtree(COCO_OUTPUT_DIR)

    # labelme2coco conversion (category_id starts at 1 for RF-DETR)
    labelme2coco.convert(
        labelme_folder=str(MERGED_DIR),
        export_dir=str(COCO_OUTPUT_DIR),
        category_id_start=1,
    )

    # Fix file_name fields (labelme2coco may use absolute paths)
    coco_json_path = COCO_OUTPUT_DIR / "dataset.json"
    if not coco_json_path.exists():
        # Try alternative name
        for f in COCO_OUTPUT_DIR.glob("*.json"):
            coco_json_path = f
            break

    print(f"  COCO JSON: {coco_json_path}")

    with open(coco_json_path, "r") as f:
        coco_data = json.load(f)

    # Fix file_name to be basename only
    for img in coco_data["images"]:
        img["file_name"] = os.path.basename(img["file_name"])

    # Verify categories match our expected classes
    print(f"  Categories found: {[c['name'] for c in coco_data['categories']]}")
    print(f"  Total images: {len(coco_data['images'])}")
    print(f"  Total annotations: {len(coco_data['annotations'])}")

    # Save fixed COCO JSON
    with open(coco_json_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    return coco_data, coco_json_path


# ---------------------------------------------------------------------------
# Step 3: Train/Validation split
# ---------------------------------------------------------------------------
def create_train_val_split(coco_data, coco_json_path):
    """Split into train/val sets (image-level split, stratified attempt)."""
    print("\n" + "=" * 60)
    print("STEP 3: Creating train/validation split")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    # Get all image IDs
    image_ids = [img["id"] for img in coco_data["images"]]
    random.shuffle(image_ids)

    val_count = max(1, int(len(image_ids) * VAL_SPLIT))
    val_image_ids = set(image_ids[:val_count])
    train_image_ids = set(image_ids[val_count:])

    print(f"  Total images: {len(image_ids)}")
    print(f"  Train images: {len(train_image_ids)}")
    print(f"  Val images:   {len(val_image_ids)}")

    # Create directory structure for RF-DETR
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "valid"
    test_dir = DATASET_DIR / "test"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    # Build image lookup
    id_to_image = {img["id"]: img for img in coco_data["images"]}

    # Build annotation lookup by image_id
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    # Create train, val, and test COCO JSONs
    # Test set reuses validation images (RF-DETR requires a test split)
    for split_name, split_ids, split_dir in [
        ("train", train_image_ids, train_dir),
        ("valid", val_image_ids, val_dir),
        ("test", val_image_ids, test_dir),
    ]:
        split_images = []
        split_annotations = []
        ann_id = 1

        for img_id in sorted(split_ids):
            img_info = id_to_image[img_id]
            split_images.append(img_info)

            # Copy image file
            src_img = MERGED_DIR / img_info["file_name"]
            if src_img.exists():
                shutil.copy2(src_img, split_dir / img_info["file_name"])
            else:
                print(f"  WARNING: Image not found: {src_img}")

            # Add annotations
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    new_ann = ann.copy()
                    new_ann["id"] = ann_id
                    split_annotations.append(new_ann)
                    ann_id += 1

        split_coco = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": coco_data["categories"],
        }

        # Save as _annotations.coco.json (RF-DETR Roboflow format)
        ann_path = split_dir / "_annotations.coco.json"
        with open(ann_path, "w") as f:
            json.dump(split_coco, f, indent=2)

        print(
            f"  {split_name}: {len(split_images)} images, {len(split_annotations)} annotations"
        )

    # Print class distribution per split
    for split_name, split_dir in [
        ("train", train_dir),
        ("valid", val_dir),
        ("test", test_dir),
    ]:
        ann_path = split_dir / "_annotations.coco.json"
        with open(ann_path, "r") as f:
            data = json.load(f)
        cat_map = {c["id"]: c["name"] for c in data["categories"]}
        class_counts = {}
        for ann in data["annotations"]:
            name = cat_map.get(ann["category_id"], "unknown")
            class_counts[name] = class_counts.get(name, 0) + 1
        print(f"  {split_name} class distribution: {class_counts}")


# ---------------------------------------------------------------------------
# Step 4: Train RF-DETR
# ---------------------------------------------------------------------------
def train_model():
    """Fine-tune RFDETRBase on the prepared dataset."""
    print("\n" + "=" * 60)
    print("STEP 4: Training RFDETRBase")
    print("=" * 60)

    from rfdetr import RFDETRBase

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = RFDETRBase()

    print(f"  Dataset dir: {DATASET_DIR}")
    print(f"  Output dir:  {OUTPUT_DIR}")
    print(f"  Config: {TRAIN_CONFIG}")
    print()

    model.train(
        dataset_dir=str(DATASET_DIR),
        output_dir=str(OUTPUT_DIR),
        **TRAIN_CONFIG,
    )

    # Check for best checkpoint
    best_ckpt = OUTPUT_DIR / "checkpoint_best_total.pth"
    if best_ckpt.exists():
        size_mb = best_ckpt.stat().st_size / (1024 * 1024)
        print(f"\n  Best checkpoint: {best_ckpt} ({size_mb:.1f} MB)")
    else:
        print("\n  WARNING: checkpoint_best_total.pth not found in output")
        print(f"  Contents of {OUTPUT_DIR}:")
        for f in sorted(OUTPUT_DIR.iterdir()):
            print(f"    {f.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RF-DETR Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Datasets: {list(DATASETS.keys())}")
    print(f"  Val split: {VAL_SPLIT * 100:.0f}%")

    # Verify datasets exist
    for name in DATASETS:
        path = DATASETS_DIR / name
        if not path.exists():
            print(f"\nERROR: Dataset not found: {path}")
            print("Make sure datasets are mounted at /workspace/datasets/")
            sys.exit(1)

    # Run pipeline
    num_images = merge_datasets()
    if num_images == 0:
        print("\nERROR: No labeled images found. Check dataset directories.")
        sys.exit(1)

    coco_data, coco_json_path = convert_to_coco()
    create_train_val_split(coco_data, coco_json_path)
    train_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Best checkpoint:  {OUTPUT_DIR / 'checkpoint_best_total.pth'}")
    print()
    print("To use the trained model:")
    print("  from rfdetr import RFDETRBase")
    print(
        f'  model = RFDETRBase(pretrain_weights="{OUTPUT_DIR / "checkpoint_best_total.pth"}")'
    )
    print('  detections = model.predict("image.jpg", threshold=0.5)')


if __name__ == "__main__":
    main()
