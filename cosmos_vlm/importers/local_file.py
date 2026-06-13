from __future__ import annotations

from pathlib import Path

from cosmos_vlm.artifacts.io import copy_file, sha256_file


def import_local_image(source_path: Path, destination_path: Path) -> dict:
    src = source_path.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"source image not found: {src}")
    copy_file(src, destination_path)
    return {
        "source_type": "local_file",
        "source_path": str(src),
        "copied_to": str(destination_path.resolve()),
        "sha256": sha256_file(destination_path),
    }

