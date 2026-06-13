import json
from pathlib import Path

import pytest

from generative_str.foundationpose_io import load_robot_frame_poses, save_robot_frame_poses


def test_load_flat_list(tmp_path: Path):
    p = tmp_path / "p.json"
    data = [[0, 0, 1, 0, 0, 0, 1], [0.1, 0, 1, 0, 0, 0, 1]]
    p.write_text(json.dumps(data))
    poses = load_robot_frame_poses(p)
    assert len(poses) == 2


def test_load_foundationpose_dict(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(
        json.dumps(
            {
                "poses_cam": [],
                "poses_robot": [[0, 0, 0.7, 0, 0, 0, 1]],
            }
        )
    )
    poses = load_robot_frame_poses(p)
    assert len(poses) == 1


def test_invalid_dict(tmp_path: Path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"foo": []}))
    with pytest.raises(ValueError):
        load_robot_frame_poses(p)


def test_roundtrip(tmp_path: Path):
    p = tmp_path / "a.json"
    q = tmp_path / "b.json"
    data = [[0, 0, 1, 0, 0, 0, 1]]
    save_robot_frame_poses(p, data)
    save_robot_frame_poses(q, load_robot_frame_poses(p))
    assert json.loads(p.read_text()) == json.loads(q.read_text())
