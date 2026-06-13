from __future__ import annotations

import json
from pathlib import Path

from cosmos_vlm.importers.sim_runs import parse_log_rows, resolve_sim_image


def test_parse_log_rows(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    rows = [{"step": 0, "rgb_image": "frame_00000/00_main.png"}, {"step": 1, "rgb_image": "frame_00001/00_main.png"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    parsed = parse_log_rows(log)
    assert parsed == rows


def test_resolve_sim_image_by_step(tmp_path: Path) -> None:
    session = tmp_path / "vlm_foo"
    frame = session / "frame_00000"
    frame.mkdir(parents=True)
    image = frame / "00_main.png"
    image.write_bytes(b"png-bytes")
    log = session / "log.jsonl"
    log.write_text(json.dumps({"step": 0, "rgb_image": "frame_00000/00_main.png", "frame_dir": "frame_00000"}) + "\n")

    resolved, meta = resolve_sim_image(session, step=0)
    assert resolved == image.resolve()
    assert meta["selected_step"] == 0
    assert meta["rgb_image_relative"] == "frame_00000/00_main.png"

