"""Scene datatypes for VLA inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SceneState:
    instruction: str
    tool_label: str
    tool_contact_xyz_world: np.ndarray
    tool_current_normal: np.ndarray
    tool_current_surface_dir: np.ndarray
    material_xyz_world: np.ndarray
    destination_xyz_world: np.ndarray
    table_xyz_world: np.ndarray
    table_z: float = 0.53
    # Optional per-scene pan rim center (xy); when set, pan viz is material-relative.
    pan_center_xy_world: np.ndarray | None = None

    @property
    def has_material(self) -> bool:
        return True

    @property
    def has_destination(self) -> bool:
        return True
