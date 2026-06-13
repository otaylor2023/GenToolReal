from pathlib import Path

import numpy as np

from generative_str.cam_k_io import load_cam_k_txt


def test_load_cam_k_txt(tmp_path: Path):
    p = tmp_path / "cam_K.txt"
    p.write_text("700 0 320\n0 700 240\n0 0 1\n", encoding="utf-8")
    k = load_cam_k_txt(p)
    assert k.shape == (3, 3)
    assert np.allclose(k[0, 0], 700.0)
