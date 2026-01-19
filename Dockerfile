# Base image with Python and CUDA support
FROM nvidia/cuda:12.6.0-cudnn9-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.12 and system dependencies
# Using deadsnakes ppa for python 3.12 on ubuntu 22.04
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.12 -m venv /app/venv

# Activate venv by adding bin to PATH
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.4 support (closest to 12.6/12.8 compatible wheel)
# We use the specific command for Linux (CUDA 12.4)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 7860 8000

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Default command - launch in server mode with API enabled
CMD ["python", "gui.py", "--server", "--enable-api"]
