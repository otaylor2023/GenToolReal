"""Sample hammer-a-nail RL/eval scenes for the action-trajectory VLA.

The pretrain dataset (``dataset_0014_hammer_nail_reactive``) randomizes the table
height per scene, but the IsaacGym sim uses a *fixed* table surface (z=0.53, like
the brush sweep RL scenes). So we re-sample the same hammer scene geometry with
``table_z`` pinned to the sim table height and return only the first
(window-index 0) reactive datapoint -- the hammer at its home rest pose and the
nail head at its starting protrusion. The VLA then plans the strikes at rollout
time (closed-loop receding horizon), exactly like ``sample_rl_scenes`` does for
the brush sweep.

The returned scene dicts are compatible with ``WaypointTrajectoryDataset`` /
``_scene_to_sample`` and carry the hammer-task extras (board, nail target depth,
nail-head size) used to place the sim "nail" post.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from generative_str_pipeline.build_dataset_0008_brush_procedural import _datapoint_rng
from generative_str_pipeline.build_dataset_0014_hammer_nail_reactive import (
    HammerNailGenConfig,
    scene_to_datapoints,
)

# Sim table top (matches sample_rl_scenes / the brush-sweep RL table surface).
SIM_TABLE_Z = 0.53

# Sim "nail": a square post stood up so it is clearly visible protruding from the
# board and sinking when struck. Its TOP is aligned to the dataset nail-head
# height; see HAMMER_NAIL_POST_HEIGHT_M usage in the rollout driver.
HAMMER_NAIL_POST_SIZE = [0.016, 0.016, 0.06]
HAMMER_NAIL_POST_HEIGHT_M = float(HAMMER_NAIL_POST_SIZE[2])


def sample_hammer_scenes(
    num_scenes: int,
    *,
    seed: int = 0,
    table_z: float = SIM_TABLE_Z,
) -> List[Dict[str, Any]]:
    """Sample ``num_scenes`` hammer-a-nail scenes on a fixed-height sim table.

    Each returned dict is the window-0 (initial) state of one reactive scene:
    the hammer resting at its home pose and the nail head at its full starting
    protrusion above the board, with ``destination_xyz_world`` set to the target
    sink depth.
    """
    # Pin the table height (no per-scene table-z randomization) so every scene
    # lands on the same sim table surface.
    cfg = HammerNailGenConfig(
        table_xyz_world=(0.0, 0.0, float(table_z)),
        table_z_range=(float(table_z), float(table_z)),
        table_xy_jitter_m=0.0,
    )
    scenes: List[Dict[str, Any]] = []
    for i in range(int(num_scenes)):
        rng = _datapoint_rng(int(seed), 9000, i)
        dps = scene_to_datapoints(
            rng,
            cfg,
            shard_id="hammer_rl",
            scene_index=i,
            base_datapoint_index=0,
        )
        scene = dict(dps[0])
        # The trajectory model expects 6x9 placeholder waypoints filled at
        # rollout time; the dataset stores 15x9 analytic targets here. Replace
        # with zeros to mirror sample_rl_scenes (the VLA predicts them).
        scene["waypoints"] = np.zeros((15, 9), dtype=np.float32).tolist()
        scenes.append(scene)
    return scenes
