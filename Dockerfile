# --- Runtime Stage ---
# Single-stage build. Flash Attention is installed from a prebuilt wheel, so we
# don't need the CUDA 'devel' image or a compiler toolchain — the lightweight
# 'runtime' image is enough.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.12 and runtime dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch (CUDA 12.8 for RTX 5090 support).
# The Flash Attention wheel below is built against this exact torch/CUDA combo.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Flash Attention from a prebuilt wheel (py3.12 / torch 2.8 / cu128 /
# linux x86_64). This avoids the 20-40 min source compile entirely.
# Wheel source: https://github.com/mjun0812/flash-attention-prebuild-wheels
# If you bump the Python/torch/CUDA versions above, pick the matching wheel there.
RUN pip install --no-cache-dir \
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.8-cp312-cp312-linux_x86_64.whl"

# Copy requirements
COPY requirements.txt .

# Install remaining Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the unified GUI + API port
EXPOSE 8585

# Default command: server mode (bind 0.0.0.0) with the REST API enabled,
# GUI and API served together on port 8585.
CMD ["python", "gui.py", "--server", "--enable-api", "--port", "8585"]
