# Training container for AMD MI100 GPU (ROCm)
# Base image: official ROCm PyTorch image
FROM rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget vim \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY train/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project
COPY . /workspace

ENV PYTHONPATH=/workspace
ENV HSA_OVERRIDE_GFX_VERSION=9.0.0

CMD ["python", "train/train.py", "--config", "train/config_gismo.json"]
