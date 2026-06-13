from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


def _headers(api_key: str) -> dict[str, str]:
    if not api_key:
        raise ValueError("WAVESPEED_API_KEY is missing")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def submit_image_to_video(
    *,
    base_url: str,
    api_key: str,
    prompt: str,
    image: str,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/wavespeed-ai/cosmos-predict-2.5/image-to-video"
    if not image.startswith("http://") and not image.startswith("https://"):
        raise ValueError("image must be a publicly accessible URL for Wavespeed API")
    payload = {"prompt": prompt, "image": image}
    resp = requests.post(endpoint, json=payload, headers=_headers(api_key), timeout=timeout_seconds)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Wavespeed submit failed: status={resp.status_code}, body={resp.text[:500]}"
        )
    return resp.json()


def get_prediction_result(
    *,
    base_url: str,
    api_key: str,
    task_id: str,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    endpoint = f"{base_url.rstrip('/')}/predictions/{task_id}/result"
    resp = requests.get(endpoint, headers=_headers(api_key), timeout=timeout_seconds)
    if resp.status_code >= 400:
        raise RuntimeError(
            f"Wavespeed result fetch failed: status={resp.status_code}, body={resp.text[:500]}"
        )
    return resp.json()


def download_url_to_file(url: str, output_path: Path, timeout_seconds: int = 120) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_seconds) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"download failed: status={resp.status_code}, url={url}")
        with output_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

