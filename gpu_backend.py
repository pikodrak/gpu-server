"""
GPU backend - handles PyTorch, llama-cpp-python, and Stable Diffusion operations.
"""
from __future__ import annotations

import base64
import io
import logging
import random
import time
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


class GPUBackend:
    def __init__(self) -> None:
        self._torch = None
        self._llm = None
        self._sd_pipe = None
        self._device = None

    def initialize(self) -> None:
        try:
            import torch  # type: ignore

            self._torch = torch
            if settings.device == "cuda" and torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
                )
            elif settings.device == "mps" and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon)")
            else:
                self._device = torch.device("cpu")
                logger.warning("GPU not available, falling back to CPU")
        except ImportError:
            logger.warning("PyTorch not installed — GPU operations disabled")

        self._load_llm()

        if settings.sd_enable:
            self._load_sd()

    def _load_llm(self) -> None:
        if not settings.llm_model_path:
            logger.info("No LLM model path set; LLM inference disabled")
            return
        model_path = Path(settings.llm_model_path)
        if not model_path.exists():
            logger.warning(f"LLM model not found at {model_path}")
            return
        try:
            from llama_cpp import Llama  # type: ignore

            n_gpu_layers = -1 if (self._device and self._device.type == "cuda") else 0
            self._llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=4096,
                verbose=False,
            )
            logger.info(f"LLM loaded: {model_path.name} (gpu_layers={n_gpu_layers})")
        except ImportError:
            logger.warning("llama-cpp-python not installed; LLM inference disabled")
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")

    def _load_sd(self) -> None:
        if not self._torch:
            logger.warning("PyTorch not available; Stable Diffusion disabled")
            return
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore

            dtype = self._torch.float16 if self._device.type == "cuda" else self._torch.float32
            self._sd_pipe = StableDiffusionPipeline.from_pretrained(
                settings.sd_model_id,
                torch_dtype=dtype,
                safety_checker=None,
            )
            self._sd_pipe = self._sd_pipe.to(self._device)
            self._sd_pipe.enable_attention_slicing()
            logger.info(f"Stable Diffusion loaded: {settings.sd_model_id}")
        except ImportError:
            logger.warning("diffusers not installed; image generation disabled")
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion: {e}")

    # ── Public API ────────────────────────────────────────────────────────────

    def device_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"name": "cpu", "type": "cpu", "available": True}
        if self._torch and self._device:
            info["type"] = self._device.type
            if self._device.type == "cuda" and self._torch.cuda.is_available():
                props = self._torch.cuda.get_device_properties(0)
                info["name"] = props.name
                info["vram_total_gb"] = round(props.total_memory / 1e9, 2)
                free_bytes, _ = self._torch.cuda.mem_get_info(0)
                info["vram_free_gb"] = round(free_bytes / 1e9, 2)
            else:
                info["name"] = self._device.type
        return info

    def detailed_info(self) -> dict[str, Any]:
        info = self.device_info()
        info["llm_loaded"] = self._llm is not None
        info["llm_model"] = settings.llm_model_path
        info["sd_loaded"] = self._sd_pipe is not None
        info["sd_model"] = settings.sd_model_id
        info["torch_version"] = self._torch.__version__ if self._torch else None
        if self._torch and self._device and self._device.type == "cuda":
            info["cuda_version"] = self._torch.version.cuda
        return info

    def run_inference(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        if self._llm is None:
            raise ValueError(
                "No LLM model loaded. Set GPU_SERVER_LLM_MODEL_PATH in .env "
                "and install llama-cpp-python."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = response["choices"][0]
        return {
            "text": choice["message"]["content"],
            "tokens_generated": response["usage"]["completion_tokens"],
        }

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        output_format: str,
    ) -> dict[str, Any]:
        if self._sd_pipe is None:
            raise ValueError(
                "Stable Diffusion not loaded. Set GPU_SERVER_SD_ENABLE=true "
                "and install diffusers."
            )

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        generator = self._torch.Generator(device=self._device).manual_seed(seed)
        result = self._sd_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        image = result.images[0]

        if output_format == "filepath":
            output_dir = Path(settings.output_dir)
            filename = f"sd_{int(time.time())}_{seed}.png"
            filepath = output_dir / filename
            image.save(filepath)
            return {"image_data": str(filepath), "seed": seed}
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode()
            return {"image_data": encoded, "seed": seed}

    def run_torch_operation(
        self, operation: str, data: Any, **kwargs: Any
    ) -> Any:
        """Run a named PyTorch operation. Supports a curated safe set."""
        if self._torch is None:
            raise ValueError("PyTorch not available")

        allowed = {
            "matmul",
            "dot",
            "norm",
            "mean",
            "std",
            "softmax",
            "sigmoid",
            "relu",
            "cosine_similarity",
        }
        if operation not in allowed:
            raise ValueError(
                f"Operation '{operation}' not allowed. Allowed: {sorted(allowed)}"
            )

        tensor = self._torch.tensor(data, dtype=self._torch.float32, device=self._device)

        if operation == "matmul":
            other = self._torch.tensor(
                kwargs.get("other", data), dtype=self._torch.float32, device=self._device
            )
            result = self._torch.matmul(tensor, other)
        elif operation == "cosine_similarity":
            other = self._torch.tensor(
                kwargs.get("other", data), dtype=self._torch.float32, device=self._device
            )
            result = self._torch.nn.functional.cosine_similarity(
                tensor.unsqueeze(0), other.unsqueeze(0)
            )
        elif operation == "softmax":
            result = self._torch.nn.functional.softmax(tensor, dim=kwargs.get("dim", -1))
        elif operation == "sigmoid":
            result = self._torch.sigmoid(tensor)
        elif operation == "relu":
            result = self._torch.nn.functional.relu(tensor)
        else:
            fn = getattr(self._torch, operation)
            result = fn(tensor, **kwargs)

        return result.cpu().tolist()

    def cleanup(self) -> None:
        self._llm = None
        self._sd_pipe = None
        if self._torch and self._device and self._device.type == "cuda":
            self._torch.cuda.empty_cache()
