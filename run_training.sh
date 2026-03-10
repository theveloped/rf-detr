#!/bin/bash
# =============================================================================
# RF-DETR Training Launch Script
# Builds Docker image and starts training on GPU 1
# =============================================================================
# Usage: bash run_training.sh
# =============================================================================

set -e

IMAGE_NAME="rfdetr-train"
CONTAINER_NAME="rfdetr-training"
GPU_ID="1"  # Use GPU 1 (GPU 0 has SAM/ollama containers)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
OUTPUT_DIR="${SCRIPT_DIR}/training_output"

echo "============================================="
echo "RF-DETR Training Pipeline"
echo "============================================="
echo "  GPU:         ${GPU_ID}"
echo "  Datasets:    ${DATASETS_DIR}"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

# Verify datasets exist
for ds in aruco_and_tray_1 fixed_belt_view_1 mixed_tray_belt_1; do
    if [ ! -d "${DATASETS_DIR}/${ds}" ]; then
        echo "ERROR: Dataset not found: ${DATASETS_DIR}/${ds}"
        echo "Please SCP datasets to ${DATASETS_DIR}/ first."
        exit 1
    fi
    echo "  Found dataset: ${ds}"
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Stop existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo ""
    echo "Stopping existing container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Build Docker image
echo ""
echo "Building Docker image: ${IMAGE_NAME}"
echo "---------------------------------------------"
docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

# Run training
echo ""
echo "Starting training on GPU ${GPU_ID}..."
echo "---------------------------------------------"
docker run \
    --name "${CONTAINER_NAME}" \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
    -e CUDA_VISIBLE_DEVICES="0" \
    -v "${DATASETS_DIR}:/workspace/datasets:ro" \
    -v "${OUTPUT_DIR}:/workspace/training_output" \
    --shm-size=16g \
    --rm \
    "${IMAGE_NAME}"

echo ""
echo "============================================="
echo "Training complete!"
echo "Output: ${OUTPUT_DIR}"
echo "Best checkpoint: ${OUTPUT_DIR}/checkpoint_best_total.pth"
echo "============================================="
