# --- Builder Stage ---
# We need the full CUDA development environment (devel) to compile Flash Attention.
# This stage is temporary and strictly for building dependencies.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

WORKDIR /build

# Install build dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for isolation
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip wheel packaging psutil setuptools

# Install PyTorch (Required to build Flash Attention)
# Must match the version used in the final stage
# Using CUDA 12.8 for RTX 5090 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Build Flash Attention Wheel
# This can take 20-40+ minutes.
# Each nvcc job can use ~8-10GB RAM; keep MAX_JOBS low to avoid OOM-killing
# the (WSL2/Docker Desktop) builder. Raise only if the builder has plenty of RAM.
ENV MAX_JOBS=4
RUN pip wheel flash-attn --no-build-isolation --no-deps --wheel-dir /build/wheels

# --- Final Runtime Stage ---
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

# Install PyTorch (Same version as builder)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Copy compiled factory-fresh Flash Attention wheel from builder
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

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
