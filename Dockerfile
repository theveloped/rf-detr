# RF-DETR Fine-Tuning Docker Image
# Optimized for NVIDIA RTX A6000 (48GB VRAM, CUDA 12.1)

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3.12-distutils \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Make python3.12 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA 12.1 support first
RUN python -m pip install --no-cache-dir \
    torch>=2.2.0 torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

# Default command: run training
CMD ["python", "train.py"]
