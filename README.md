# GPU Server

HTTP REST server for GPU-accelerated ML inference on an RTX 4070 Ti (or any CUDA GPU).
Used by Paperclip agents to run LLM inference and generate game assets (textures, concept art) locally.

---

## Prerequisites

### Linux

1. **NVIDIA driver** — 525+ recommended for CUDA 12.1:
   ```bash
   nvidia-smi   # confirm driver is installed
   ```

2. **Docker Engine** (not Docker Desktop):
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER   # log out and back in
   ```

3. **NVIDIA Container Toolkit** — lets Docker access the GPU:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

4. Verify GPU access inside Docker:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

### Windows (Docker Desktop + WSL2)

1. **NVIDIA driver for Windows** (NOT the Linux/WSL driver — the Windows host driver is used):
   - Download from https://www.nvidia.com/drivers — install on Windows
   - `nvidia-smi` should work in PowerShell after install

2. **WSL2** with Ubuntu:
   ```powershell
   wsl --install -d Ubuntu
   wsl --set-default-version 2
   ```

3. **Docker Desktop** with WSL2 backend:
   - Download from https://www.docker.com/products/docker-desktop
   - In Docker Desktop → Settings → General: enable **Use the WSL 2 based engine**
   - In Settings → Resources → WSL Integration: enable your Ubuntu distro

4. **NVIDIA Container Toolkit inside WSL2** (Docker Desktop handles GPU passthrough, but the toolkit must be installed in WSL):
   ```bash
   # Run inside WSL2 (Ubuntu terminal)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
     sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   ```
   > Docker Desktop automatically restarts its engine — no manual `systemctl restart` needed.

5. Verify GPU access (run in WSL2 terminal or PowerShell):
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

---

## Docker Quick Start

```bash
# 1. Clone the repo (or copy the project files)
git clone https://github.com/pikodrak/gpu-server.git
cd gpu-server

# 2. Configure
cp .env.example .env
# Edit .env — set GPU_SERVER_API_KEYS at minimum:
#   GPU_SERVER_API_KEYS=["your-secret-key"]
# For LLM inference, also set:
#   GPU_SERVER_LLM_MODEL_PATH=/app/models/llama-3-8b-instruct.Q4_K_M.gguf

# 3. (Optional) Download a GGUF model for LLM inference
mkdir -p models
# Example: Llama 3 8B Instruct Q4_K_M (~4.7 GB)
wget -O models/llama-3-8b-instruct.Q4_K_M.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

# 4. Build and start
docker compose up --build

# Server is ready at http://localhost:8000
# First start takes longer — PyTorch + Stable Diffusion are installed during build (~5–10 min)
```

To run in the background:
```bash
docker compose up -d --build
docker compose logs -f   # follow logs
docker compose down      # stop
```

### Volume layout

| Host path | Container path | Purpose |
|---|---|---|
| `./models/` | `/app/models` | GGUF model files, SD model cache |
| `./outputs/` | `/app/outputs` | Generated images |

Models downloaded inside the container (e.g. Stable Diffusion from HuggingFace) are cached in `/app/models`, so they survive container restarts.

---

## Running without Docker

```bash
# Python 3.11+
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core server
pip install -r requirements.txt

# LLM inference (optional)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
# Windows: $env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python

# Stable Diffusion (optional)
pip install diffusers transformers accelerate Pillow

cp .env.example .env
# Edit .env — at minimum set GPU_SERVER_API_KEYS

python server.py
```

---

## API Reference

All endpoints except `/health` require:
```
X-API-Key: your-secret-key
```

### `GET /health`
Public status check. Returns GPU info.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "gpu": {
    "name": "NVIDIA GeForce RTX 4070 Ti",
    "type": "cuda",
    "vram_total_gb": 12.0,
    "vram_free_gb": 11.2
  },
  "timestamp": 1714123456.789
}
```

### `GET /api/v1/gpu/info`
Detailed GPU and model status.

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/gpu/info
```

### `POST /api/v1/inference`
LLM text inference (requires `GPU_SERVER_LLM_MODEL_PATH`).

```bash
curl -X POST http://localhost:8000/api/v1/inference \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short game design document for a roguelike dungeon crawler.",
    "model": "llama",
    "max_tokens": 512,
    "temperature": 0.7,
    "system_prompt": "You are a senior game designer."
  }'
```

### `POST /api/v1/image/generate`
Image generation via Stable Diffusion (requires `GPU_SERVER_SD_ENABLE=true`).

```bash
curl -X POST http://localhost:8000/api/v1/image/generate \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "seamless stone dungeon texture, tileable, dark fantasy, high quality",
    "negative_prompt": "blurry, low quality, watermark",
    "width": 512,
    "height": 512,
    "steps": 30,
    "guidance_scale": 7.5,
    "output_format": "base64"
  }'
```

Returns a base64-encoded PNG (or filepath if `output_format: "filepath"`).

> **Note:** The NSFW safety checker is disabled by default. This is intentional for a local developer tool.

### `POST /api/v1/torch`
PyTorch operation on GPU. Operations: `matmul`, `dot`, `norm`, `mean`, `std`, `softmax`, `sigmoid`, `relu`, `cosine_similarity`.

```bash
curl -X POST http://localhost:8000/api/v1/torch \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"operation": "softmax", "data": [1.0, 2.0, 3.0, 4.0]}'
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GPU_SERVER_HOST` | `0.0.0.0` | Bind address |
| `GPU_SERVER_PORT` | `8000` | Port |
| `GPU_SERVER_API_KEYS` | `[]` | JSON array of valid API keys (required) |
| `GPU_SERVER_DEVICE` | `cuda` | `cuda`, `cpu`, or `mps` |
| `GPU_SERVER_LLM_MODEL_PATH` | `` | Path to GGUF model (in Docker: `/app/models/...`) |
| `GPU_SERVER_SD_ENABLE` | `false` | Enable Stable Diffusion |
| `GPU_SERVER_SD_MODEL_ID` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID |
| `GPU_SERVER_OUTPUT_DIR` | `./outputs` | Directory for saved images |

---

## Calling from a Paperclip Agent

```python
import httpx

GPU_SERVER_URL = "http://192.168.254.2:8000"
API_KEY = "your-secret-key"

headers = {"X-API-Key": API_KEY}

# LLM inference
resp = httpx.post(
    f"{GPU_SERVER_URL}/api/v1/inference",
    headers=headers,
    json={"prompt": "Generate 5 enemy names for a fantasy RPG", "max_tokens": 256},
    timeout=120,
)
print(resp.json()["text"])

# Image generation
resp = httpx.post(
    f"{GPU_SERVER_URL}/api/v1/image/generate",
    headers=headers,
    json={"prompt": "game texture, mossy stone wall, seamless, 4k", "width": 512, "height": 512, "steps": 25},
    timeout=300,
)
import base64, pathlib
pathlib.Path("texture.png").write_bytes(base64.b64decode(resp.json()["image_data"]))
```
