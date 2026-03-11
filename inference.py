#!/usr/bin/env python3
"""
RF-DETR Inference Script
========================
Run object detection on images using a fine-tuned RF-DETR checkpoint.

Usage:
    # Single image
    python inference.py image.png

    # Multiple images
    python inference.py image1.png image2.png image3.png

    # All PNGs in a directory
    python inference.py ./test_images/

    # With custom checkpoint and threshold
    python inference.py image.png --checkpoint ./training_output/checkpoint_best_ema.pth --threshold 0.5

    # Save results to a specific directory
    python inference.py image.png --output-dir ./results/

    # Print detections without saving annotated images
    python inference.py image.png --no-save

Requirements:
    pip install rfdetr supervision Pillow
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Fallback class names for backward compatibility with checkpoints that
# don't have a class_names.json alongside them.
DEFAULT_CLASS_NAMES = ["DM", "HOUSING", "CORE", "SHAFT", "EXCHANGER"]

DEFAULT_CHECKPOINT = "./training_output/checkpoint_best_ema.pth"
DEFAULT_THRESHOLD = 0.5
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def load_class_names(checkpoint_path: str) -> list[str]:
    """Load class names from class_names.json next to the checkpoint.

    Falls back to DEFAULT_CLASS_NAMES if the file is not found.
    """
    checkpoint = Path(checkpoint_path)
    class_names_path = checkpoint.parent / "class_names.json"

    if class_names_path.exists():
        with open(class_names_path, "r") as f:
            names = json.load(f)
        print(f"Loaded class names from: {class_names_path}")
        print(f"  Classes: {names}")
        return names

    print(f"WARNING: {class_names_path} not found, using default class names.")
    print(f"  Classes: {DEFAULT_CLASS_NAMES}")
    return DEFAULT_CLASS_NAMES


def load_model(checkpoint_path: str):
    """Load RF-DETR model with fine-tuned weights."""
    from rfdetr import RFDETRBase

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print("Download it from the server with:")
        print(
            f"  scp aime@10.103.200.201:~/rf-detr/training_output/checkpoint_best_ema.pth {checkpoint.parent}/"
        )
        sys.exit(1)

    model = RFDETRBase(pretrain_weights=str(checkpoint))
    print(f"Model loaded from: {checkpoint}")
    return model


def predict_image(model, image_path: Path, threshold: float):
    """Run inference on a single image and return detections."""
    image = Image.open(image_path).convert("RGB")
    detections = model.predict(image, threshold=threshold)
    return image, detections


def print_detections(image_path: Path, detections, class_names: list[str]):
    """Print detection results to stdout."""
    n = len(detections.xyxy)
    print(f"\n{'=' * 60}")
    print(f"  {image_path.name}: {n} detection(s)")
    print(f"{'=' * 60}")

    if n == 0:
        print("  No objects detected.")
        return

    for i in range(n):
        box = detections.xyxy[i]
        cls = class_names[detections.class_id[i]]
        conf = detections.confidence[i]
        print(
            f"  {cls:12s}  conf={conf:.3f}  "
            f"box=[{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
        )


def save_annotated_image(
    image: Image.Image,
    detections,
    image_path: Path,
    output_dir: Path,
    class_names: list[str],
):
    """Save image with bounding box annotations overlaid."""
    import numpy as np
    import supervision as sv

    annotated = np.array(image.copy())

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels = [
        f"{class_names[c]} {conf:.2f}"
        for c, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_detections.png"
    Image.fromarray(annotated).save(output_path)
    print(f"  Saved: {output_path}")


def collect_images(paths: list[str]) -> list[Path]:
    """Collect image paths from arguments (files or directories)."""
    images = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in IMAGE_EXTENSIONS:
                images.extend(sorted(path.glob(f"*{ext}")))
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
        else:
            print(f"  WARNING: Skipping {p} (not a valid image or directory)")
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on images with a fine-tuned checkpoint."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file(s) or directory of images to run inference on.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold for detections (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--output-dir",
        default="./inference_output",
        help="Directory to save annotated images (default: ./inference_output)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Only print detections, do not save annotated images.",
    )
    args = parser.parse_args()

    # Collect all image paths
    image_paths = collect_images(args.images)
    if not image_paths:
        print("ERROR: No valid images found.")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to process.")
    print(f"Confidence threshold: {args.threshold}")

    # Load model and class names
    model = load_model(args.checkpoint)
    class_names = load_class_names(args.checkpoint)

    # Run inference
    output_dir = Path(args.output_dir)
    total_detections = 0

    for image_path in image_paths:
        image, detections = predict_image(model, image_path, args.threshold)
        print_detections(image_path, detections, class_names)
        total_detections += len(detections.xyxy)

        if not args.no_save:
            save_annotated_image(image, detections, image_path, output_dir, class_names)

    print(f"\n{'=' * 60}")
    print(
        f"  Done. {total_detections} total detection(s) across {len(image_paths)} image(s)."
    )
    if not args.no_save:
        print(f"  Annotated images saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
