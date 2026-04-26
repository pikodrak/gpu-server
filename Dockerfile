FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.1
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install llama-cpp-python with CUDA
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Install Stable Diffusion stack
RUN pip install diffusers transformers accelerate Pillow

# Install server requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY server.py config.py auth.py gpu_backend.py ./

# Create directories and non-root user; grant ownership before volume mount
RUN mkdir -p /app/models /app/outputs && \
    useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

VOLUME ["/app/models", "/app/outputs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python3", "server.py"]
