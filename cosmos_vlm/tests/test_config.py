from __future__ import annotations

import os
from pathlib import Path

from cosmos_vlm.config import load_config


def test_load_config_reads_dotenv(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('WAVESPEED_API_KEY = "abc123"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WAVESPEED_API_KEY", raising=False)
    cfg = load_config()
    assert cfg.wavespeed_api_key == "abc123"
    assert os.environ.get("WAVESPEED_API_KEY") == "abc123"

