#!/usr/bin/env python3
"""
Generate a short video from a single image (first frame) using **Google Veo on Vertex AI**.

Edit the **CONFIG** block below (paths, bucket, prompt). Run::

    conda activate SimToolReal-veo
    pip install -r veo/requirements.txt
    python veo/veo_frame_to_video.py

Vertex returns a ``gs://`` reference; this script downloads it next to the frame as
``{frame_stem}_veo_{YYYYMMDD_HHMMSS}.mp4``.

Vertex Veo: https://cloud.google.com/vertex-ai/generative-ai/docs/video/overview
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

if sys.version_info < (3, 10):
    print(
        "Error: Python 3.10+ required for google-genai. "
        "Try: conda activate SimToolReal-veo",
        file=sys.stderr,
    )
    sys.exit(1)

from google import genai
from google.genai import types


# --- Repo layout: SimToolReal/veo/veo_frame_to_video.py ---
_REPO_ROOT = Path(__file__).resolve().parent.parent

# --- CONFIG: edit these ---
CREDENTIALS_JSON = _REPO_ROOT / "simtoolreal-93aa22063ba0.json"
FRAME_IMAGE = _REPO_ROOT / "veo" / "data" / "frame_0000.jpg"
# Output path is ``{FRAME_IMAGE parent}/{stem}_veo_{YYYYMMDD_HHMMSS}.mp4`` (set at run time).
OUTPUT_GCS_URI = "gs://simtoolreal-veo-out/veo/"

MODEL_ID = "veo-3.1-generate-001"
ASPECT_RATIO = None  # or "16:9" / "9:16"
DURATION_SECONDS = None  # or 4, 6, 8
POLL_SECONDS = 10.0

# Veo does not support a "temperature" parameter (that exists for text models only).
# For more reproducible runs on Vertex AI, set VEO_SEED (override with env VEO_SEED).
# Veo 3.x does not allow turning off prompt enhancement (API error if enhance_prompt=False).
VEO_SEED = 42  # int, or None to let the API pick a random seed each run

VIDEO_PROMPT = """
Continue this exact scene with realistic, continuous motion from the first frame.

The person reaches down with their hand, grasps the large brush by its handle, and lifts
it up from the table. They use the brush to sweep the small orange scraps in a smooth,
deliberate motion across the surface, pushing the scraps into the green rectangular tray
or bucket at the side—everything ends up inside that container.

Important constraints: keep the green tray or bucket fixed in place on the table; do not
slide, tilt, or reposition it. Only the person's arms and the brush move. The robotic arm
and background stay as they are. Maintain the same lighting, camera angle, and lab setting;
natural hand motion and believable physics. The camera should not move at all. The person should use the same hand to sweep the scraps into the tray.
""".strip()


def _output_veo_path(frame_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return frame_path.parent / f"{frame_path.stem}_veo_{ts}.mp4"


def _parse_gs_uri(gs_uri: str) -> tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Not a gs:// URI: {gs_uri!r}")
    rest = gs_uri[5:]
    if "/" not in rest:
        raise ValueError(f"Invalid gs:// URI (no object path): {gs_uri!r}")
    bucket, _, blob = rest.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid gs:// URI: {gs_uri!r}")
    return bucket, blob


def _persist_generated_video(
    client: genai.Client,
    video: types.Video,
    out_path: Path,
    *,
    vertex: bool,
) -> None:
    """Write API output to disk (inline bytes, GCS, HTTPS, or Gemini file download)."""
    if video.video_bytes:
        out_path.write_bytes(video.video_bytes)
        return

    uri = (video.uri or "").strip()
    if uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ImportError as e:
            raise RuntimeError(
                "Vertex returned a gs:// video URI. Install: pip install google-cloud-storage"
            ) from e
        bucket_name, blob_name = _parse_gs_uri(uri)
        storage.Client().bucket(bucket_name).blob(blob_name).download_to_filename(
            str(out_path)
        )
        return

    if uri.startswith("http://") or uri.startswith("https://"):
        urllib.request.urlretrieve(uri, str(out_path))
        return

    if vertex:
        raise RuntimeError(
            "Vertex response has no video bytes and no gs:// or https URI on the video. "
            "Set OUTPUT_GCS_URI (or env VEO_OUTPUT_GCS_URI) to gs://your-bucket/prefix/ "
            "and grant the service account write access to that bucket."
        )

    client.files.download(file=video)
    video.save(str(out_path))


def _configure_vertex_service_account(credentials_path: Path) -> None:
    """Point ADC at the JSON key and set Vertex env defaults."""
    path = credentials_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Credentials file not found: {path}")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(path)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    project_id = data.get("project_id")
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id


def main() -> int:
    try:
        _configure_vertex_service_account(CREDENTIALS_JSON)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading credentials: {e}", file=sys.stderr)
        return 1
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print(
            "Error: GOOGLE_CLOUD_PROJECT is not set and could not be read from the "
            "credentials JSON (missing project_id).",
            file=sys.stderr,
        )
        return 1

    frame_path = FRAME_IMAGE.resolve()
    if not frame_path.is_file():
        print(f"Error: frame file not found: {frame_path}", file=sys.stderr)
        return 1

    out_path = _output_veo_path(frame_path).resolve()

    vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in (
        "1",
        "true",
        "yes",
    )
    model_id = os.environ.get("VEO_MODEL") or MODEL_ID

    client = genai.Client()
    print(f"Using model: {model_id}")

    try:
        image = types.Image.from_file(location=str(frame_path))
    except Exception as e:
        print(f"Error loading image: {e}", file=sys.stderr)
        return 1

    output_gcs_uri = os.environ.get("VEO_OUTPUT_GCS_URI") or OUTPUT_GCS_URI
    if output_gcs_uri and not output_gcs_uri.startswith("gs://"):
        print(
            f"Error: output GCS URI must start with gs:// (got {output_gcs_uri!r}).",
            file=sys.stderr,
        )
        return 1

    if vertex and not output_gcs_uri:
        print(
            "Warning: Vertex Veo usually needs OUTPUT_GCS_URI or env VEO_OUTPUT_GCS_URI.",
            file=sys.stderr,
        )

    cfg_kwargs: dict = {}
    if ASPECT_RATIO:
        cfg_kwargs["aspect_ratio"] = ASPECT_RATIO
    if DURATION_SECONDS is not None:
        cfg_kwargs["duration_seconds"] = DURATION_SECONDS
    if output_gcs_uri:
        cfg_kwargs["output_gcs_uri"] = output_gcs_uri
    seed_env = os.environ.get("VEO_SEED")
    if seed_env is not None and seed_env.strip() != "":
        try:
            cfg_kwargs["seed"] = int(seed_env, 10)
        except ValueError:
            print(
                f"Warning: invalid VEO_SEED={seed_env!r}, using VEO_SEED from script.",
                file=sys.stderr,
            )
            if VEO_SEED is not None:
                cfg_kwargs["seed"] = VEO_SEED
    elif VEO_SEED is not None:
        cfg_kwargs["seed"] = VEO_SEED

    config = types.GenerateVideosConfig(**cfg_kwargs) if cfg_kwargs else None

    try:
        operation = client.models.generate_videos(
            model=model_id,
            prompt=VIDEO_PROMPT,
            image=image,
            config=config,
        )
    except Exception as e:
        print(f"Error starting video generation: {e}", file=sys.stderr)
        return 1

    print("Video generation started; polling until complete…")
    while not operation.done:
        time.sleep(POLL_SECONDS)
        try:
            operation = client.operations.get(operation)
        except Exception as e:
            print(f"Error polling operation: {e}", file=sys.stderr)
            return 1
        if operation.error:
            print(f"Operation failed: {operation.error}", file=sys.stderr)
            return 1

    response = operation.response
    if not response or not response.generated_videos:
        print("Error: no video in response.", file=sys.stderr)
        return 1

    generated = response.generated_videos[0]
    vid = generated.video
    if vid is None:
        print("Error: response contained no video object.", file=sys.stderr)
        return 1
    try:
        _persist_generated_video(client, vid, out_path, vertex=vertex)
    except Exception as e:
        print(f"Error saving video: {e}", file=sys.stderr)
        return 1

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
