"""
Configuration management - reads from .env or config.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load config.yaml if it exists, otherwise return empty dict."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with config_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseSettings):
    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    allowed_origins: list[str] = Field(default=["*"])

    # Auth
    api_keys: list[str] = Field(default_factory=list)

    # GPU / Models
    device: str = Field(default="cuda")  # cuda, cpu, mps
    llm_model_path: str = Field(default="")
    sd_model_id: str = Field(default="runwayml/stable-diffusion-v1-5")
    sd_enable: bool = Field(default=False)

    # Output
    output_dir: str = Field(default="./outputs")

    model_config = {"env_prefix": "GPU_SERVER_", "env_file": ".env", "extra": "ignore"}

    def model_post_init(self, __context: Any) -> None:
        yaml_config = _load_yaml_config()
        for key, value in yaml_config.items():
            # Only apply yaml value when the field still holds its default — pydantic-settings
            # does not inject .env values into os.environ, so os.environ.get() cannot be used
            # to detect whether a .env value was set.
            if key in self.model_fields and getattr(self, key) == self.model_fields[key].default:
                object.__setattr__(self, key, value)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
