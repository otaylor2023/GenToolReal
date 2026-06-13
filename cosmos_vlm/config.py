from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class CosmosConfig:
    cosmos_reason2_model_id: str
    cosmos_device: str
    cosmos_dtype: str
    cosmos_max_new_tokens: int
    wavespeed_api_key: str
    wavespeed_base_url: str
    video_provider: str
    veo_model_id: str
    veo_output_gcs_uri: str
    veo_seed: int
    veo_duration_seconds: int
    veo_aspect_ratio: str
    cosmos_runs_dir: Path


def load_config() -> CosmosConfig:
    _load_env_file(Path(".env").resolve())
    return CosmosConfig(
        cosmos_reason2_model_id=os.getenv("COSMOS_REASON2_MODEL_ID", "nvidia/Cosmos-Reason2-8B"),
        cosmos_device=os.getenv("COSMOS_DEVICE", "cuda"),
        cosmos_dtype=os.getenv("COSMOS_DTYPE", "bfloat16"),
        cosmos_max_new_tokens=int(os.getenv("COSMOS_MAX_NEW_TOKENS", "8192")),
        wavespeed_api_key=os.getenv("WAVESPEED_API_KEY", ""),
        wavespeed_base_url=os.getenv("WAVESPEED_BASE_URL", "https://api.wavespeed.ai/api/v3"),
        video_provider=os.getenv("VIDEO_PROVIDER", "wavespeed"),
        veo_model_id=os.getenv("VEO_MODEL_ID", "veo-3.1-generate-001"),
        veo_output_gcs_uri=os.getenv("VEO_OUTPUT_GCS_URI", ""),
        veo_seed=int(os.getenv("VEO_SEED", "42")),
        veo_duration_seconds=int(os.getenv("VEO_DURATION_SECONDS", "4")),
        veo_aspect_ratio=os.getenv("VEO_ASPECT_RATIO", "16:9"),
        cosmos_runs_dir=Path(os.getenv("COSMOS_RUNS_DIR", "cosmos_vlm/runs")),
    )

