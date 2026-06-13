from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from google import genai
from google.genai import types


def _load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def _resolve_vertex_service_account_json() -> Path | None:
    explicit = (os.environ.get("VEO_SERVICE_ACCOUNT_JSON") or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p
        raise RuntimeError(f"VEO_SERVICE_ACCOUNT_JSON file not found: {p}")

    repo_default = Path("/home/ubuntu/Generative_STR/simtoolreal-93aa22063ba0.json")
    if repo_default.is_file():
        return repo_default

    gac = (os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if gac:
        p = Path(gac).expanduser()
        if p.is_file():
            return p
        raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {p}")
    return None


def _resolve_vertex_project_id(cred_path: Path | None) -> str:
    project = (
        (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
        or (os.environ.get("GCLOUD_PROJECT") or "").strip()
    )
    if project:
        return project
    if cred_path and cred_path.is_file():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return str(data.get("project_id") or "").strip()
        except (OSError, json.JSONDecodeError):
            pass
    return ""


def _vertex_storage_client():
    from google.cloud import storage
    from google.oauth2 import service_account

    cred_path = _resolve_vertex_service_account_json()
    if cred_path is None:
        return storage.Client()
    creds = service_account.Credentials.from_service_account_file(
        str(cred_path), scopes=("https://www.googleapis.com/auth/cloud-platform",)
    )
    project = _resolve_vertex_project_id(cred_path) or None
    return storage.Client(project=project, credentials=creds)


def init_veo_client() -> genai.Client:
    _load_env_file(Path("/home/ubuntu/Generative_STR/.env"))
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    force_dev = (os.environ.get("VEO_FORCE_DEVELOPER_API") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cred_path = _resolve_vertex_service_account_json()

    if cred_path is not None and not (force_dev and api_key):
        from google.oauth2 import service_account

        creds = service_account.Credentials.from_service_account_file(
            str(cred_path), scopes=("https://www.googleapis.com/auth/cloud-platform",)
        )
        project = (
            (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
            or (os.environ.get("GCLOUD_PROJECT") or "").strip()
        )
        if not project:
            try:
                project = json.loads(cred_path.read_text(encoding="utf-8")).get("project_id", "").strip()
            except (OSError, json.JSONDecodeError):
                project = ""
        if not project:
            raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT/project_id for Vertex VEO")
        location = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip() or None
        kwargs: dict[str, Any] = {
            "vertexai": True,
            "credentials": creds,
            "project": project,
        }
        if location:
            kwargs["location"] = location
        return genai.Client(**kwargs)

    if api_key:
        return genai.Client(api_key=api_key)

    raise RuntimeError(
        "VEO auth not configured. Set GOOGLE_API_KEY/GEMINI_API_KEY or Vertex credentials env vars."
    )


def submit_veo_image_to_video(
    *,
    prompt: str,
    image: str,
    model: str,
    output_gcs_uri: str = "",
    duration_seconds: int = 4,
    aspect_ratio: str = "16:9",
    seed: int = 42,
    negative_prompt: str = "",
) -> dict[str, Any]:
    client = init_veo_client()
    image_arg: Any
    if image.startswith("gs://"):
        image_arg = types.Image(gcs_uri=image, mime_type="image/png")
    else:
        p = Path(image).expanduser()
        if p.is_file():
            image_arg = types.Image.from_file(location=str(p.resolve()))
        else:
            image_arg = image
    cfg_kwargs: dict[str, Any] = {
        "number_of_videos": 1,
        "duration_seconds": int(duration_seconds),
        "aspect_ratio": aspect_ratio,
        "seed": int(seed),
    }
    if output_gcs_uri.strip():
        cfg_kwargs["output_gcs_uri"] = output_gcs_uri.strip()
    if negative_prompt.strip():
        cfg_kwargs["negative_prompt"] = negative_prompt.strip()

    operation = client.models.generate_videos(
        model=model,
        prompt=prompt,
        image=image_arg,
        config=types.GenerateVideosConfig(**cfg_kwargs),
    )
    return {
        "provider": "veo",
        "operation_name": getattr(operation, "name", ""),
        "done": bool(getattr(operation, "done", False)),
        "model": model,
        "output_gcs_uri": output_gcs_uri.strip(),
        "duration_seconds": int(duration_seconds),
        "aspect_ratio": aspect_ratio,
        "seed": int(seed),
        "negative_prompt": negative_prompt.strip(),
        "status": "completed" if bool(getattr(operation, "done", False)) else "processing",
    }


def _extract_generated_videos(operation: Any) -> list[Any]:
    result = getattr(operation, "result", None)
    if result is None:
        return []
    return list(getattr(result, "generated_videos", []) or [])


def poll_veo_operation(
    *,
    operation_name: str,
    model: str,
) -> dict[str, Any]:
    client = init_veo_client()
    operation = client.operations.get(operation=types.GenerateVideosOperation(name=operation_name))
    done = bool(getattr(operation, "done", False))
    outputs: list[str] = []
    generated = _extract_generated_videos(operation)
    for gv in generated:
        v = getattr(gv, "video", None)
        uri = getattr(v, "uri", None) if v is not None else None
        if isinstance(uri, str) and uri:
            outputs.append(uri)
    return {
        "provider": "veo",
        "operation_name": operation_name,
        "done": done,
        "status": "completed" if done else "processing",
        "outputs": outputs,
        "model": model,
        "operation_obj": operation,
    }


def download_veo_outputs(
    *,
    operation_obj: Any,
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client = init_veo_client()
    downloaded: list[str] = []
    generated = _extract_generated_videos(operation_obj)

    for idx, gv in enumerate(generated):
        video = getattr(gv, "video", None)
        if video is None:
            continue
        uri = getattr(video, "uri", None)
        out = output_dir / f"output_{idx:04d}.mp4"

        if isinstance(uri, str) and uri.startswith("http"):
            with requests.get(uri, stream=True, timeout=120) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"veo output download failed: status={resp.status_code}")
                with out.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            downloaded.append(str(out.resolve()))
            continue
        if isinstance(uri, str) and uri.startswith("gs://"):
            parsed = urlparse(uri)
            bucket = parsed.netloc
            blob_name = parsed.path.lstrip("/")
            if bucket and blob_name:
                storage_client = _vertex_storage_client()
                blob = storage_client.bucket(bucket).blob(blob_name)
                blob.download_to_filename(str(out))
                downloaded.append(str(out.resolve()))
                continue

        try:
            client.files.download(file=video)
            data = getattr(video, "video_bytes", None)
            if data:
                out.write_bytes(data)
                downloaded.append(str(out.resolve()))
        except Exception:
            # Vertex clients may not support files.download without a direct URI.
            continue
    return downloaded

