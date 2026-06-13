"""SimToolReal narrow-table workspace bounds (shared by dataset + RL sampling)."""

from __future__ import annotations

import numpy as np

# table_narrow.urdf: 0.475 x 0.4 m, centered at world origin; top z ~= 0.53.
SIM_TABLE_SIZE_X_M = 0.475
SIM_TABLE_SIZE_Y_M = 0.40
SIM_TABLE_TOP_Z_DEFAULT = 0.53

# Usable half-extents on the tabletop (margin inside the collision box).
WORKSPACE_X_EXTENT_M = 0.21
WORKSPACE_Y_EXTENT_M = 0.17

TABLE_Z_RANGE = (0.45, 0.60)
TABLE_XY_JITTER_M = 0.03
BEHIND_OFFSET_M_RANGE = (0.03, 0.10)

# Blue-brush flat-on-table footprint relative to the contact (front-edge) point.
# The handle extends ~0.29 m behind the contact along -heading; the body is
# ~0.17 m wide. Used to keep the ball clear of the brush body at spawn.
BRUSH_BODY_BACK_M = 0.31
BRUSH_BODY_FRONT_M = 0.05
BRUSH_BODY_HALF_WIDTH_M = 0.11

# Table top half-extents (table_narrow.urdf is 0.475 x 0.40 m, centered).
TABLE_HALF_X_M = SIM_TABLE_SIZE_X_M / 2.0
TABLE_HALF_Y_M = SIM_TABLE_SIZE_Y_M / 2.0

# Visual goal-region (blue patch) half-size in sim (matches vlaGoalRegionRadius).
GOAL_REGION_RADIUS_M = 0.05
GOAL_REGION_MARGIN_M = 0.01

# ---------------------------------------------------------------------------
# Wide-table brush-staging layout (RL/sim only; the dataset is unchanged).
# table_wide_brush.urdf extends the original table +0.28 m in +x: the original
# near edge stays at x=-0.2375 and the far edge moves to x=+0.5175. The brush
# spawns with its head on the +x extension and its handle resting over the
# original region, in a different quadrant than the ball + goal (which stay in
# the original -x region).
WIDE_TABLE_EXT_X_M = 0.28
WIDE_TABLE_X_MIN_M = -TABLE_HALF_X_M                       # -0.2375
WIDE_TABLE_X_MAX_M = TABLE_HALF_X_M + WIDE_TABLE_EXT_X_M   # +0.5175
WIDE_TABLE_Y_MIN_M = -TABLE_HALF_Y_M
WIDE_TABLE_Y_MAX_M = TABLE_HALF_Y_M

# Brush contact (head) staging band on the +x extension. The handle reaches
# ~0.29 m back into the original region. Pushed further +x so the handle clears
# the ball region entirely (the brush sweeps things when first picked up, so it
# must start well away from the cube; it may sit over the goal patch though).
BRUSH_STAGE_X_RANGE_M = (0.30, 0.44)
BRUSH_STAGE_Y_RANGE_M = (0.03, 0.13)  # +y half → NE quadrant with +x staging

# Ball + goal region on the original -x side. The far edge (-0.02) stays just
# behind the brush handle's rear reach (~-0.01 with the +x staging band) so the
# cube never spawns under the brush; widened in y for a longer possible sweep.
# The per-scene brush/ball separation check keeps the cube clear regardless.
OBJ_REGION_X_RANGE_M = (-0.21, -0.02)
OBJ_REGION_Y_RANGE_M = (-0.15, -0.02)


def dest_extent_for_goal_region(
    x_extent: float,
    y_extent: float,
    *,
    region_radius: float = GOAL_REGION_RADIUS_M,
    margin: float = GOAL_REGION_MARGIN_M,
) -> tuple[float, float]:
    """Destination xy half-extents that keep the goal region fully on the table.

    Caps the workspace extent so a ``region_radius``-half-size patch centered at
    the destination stays inside the table (minus ``margin``).
    """
    dx = min(float(x_extent), TABLE_HALF_X_M - float(region_radius) - float(margin))
    dy = min(float(y_extent), TABLE_HALF_Y_M - float(region_radius) - float(margin))
    return max(0.0, dx), max(0.0, dy)


def ball_clear_of_brush(
    ball_xy: np.ndarray,
    contact_xy: np.ndarray,
    heading_xy: np.ndarray,
    *,
    ball_radius: float = 0.03,
    margin: float = 0.02,
) -> bool:
    """True if the ball is fully outside the brush's oriented footprint.

    ``heading_xy`` is the in-plane brush heading (the contact-frame normal used
    by the flat-rest spawn); the brush extends backward (-heading) for the
    handle and slightly forward (+heading) past the front edge.
    """
    h = np.asarray(heading_xy, dtype=np.float64).reshape(2)
    nrm = float(np.linalg.norm(h))
    if nrm < 1e-9:
        return True
    h = h / nrm
    perp = np.array([-h[1], h[0]], dtype=np.float64)
    d = np.asarray(ball_xy, dtype=np.float64).reshape(2) - np.asarray(
        contact_xy, dtype=np.float64
    ).reshape(2)
    along = float(d @ h)
    lateral = float(d @ perp)
    pad = float(ball_radius) + float(margin)
    inside_along = -(BRUSH_BODY_BACK_M + pad) <= along <= (BRUSH_BODY_FRONT_M + pad)
    inside_lat = abs(lateral) <= (BRUSH_BODY_HALF_WIDTH_M + pad)
    return not (inside_along and inside_lat)


def clip_xy_rect(
    xy: np.ndarray,
    *,
    x_extent: float = WORKSPACE_X_EXTENT_M,
    y_extent: float = WORKSPACE_Y_EXTENT_M,
) -> np.ndarray:
    out = np.asarray(xy, dtype=np.float32).reshape(2).copy()
    out[0] = float(np.clip(out[0], -x_extent, x_extent))
    out[1] = float(np.clip(out[1], -y_extent, y_extent))
    return out


def sample_xy_rect(
    rng: np.random.Generator,
    *,
    x_extent: float = WORKSPACE_X_EXTENT_M,
    y_extent: float = WORKSPACE_Y_EXTENT_M,
) -> np.ndarray:
    return np.array(
        [
            float(rng.uniform(-x_extent, x_extent)),
            float(rng.uniform(-y_extent, y_extent)),
        ],
        dtype=np.float32,
    )


def clip_waypoints_xy(
    wp_contacts: np.ndarray,
    *,
    x_extent: float = WORKSPACE_X_EXTENT_M,
    y_extent: float = WORKSPACE_Y_EXTENT_M,
) -> None:
    """In-place clip waypoint contact xy columns."""
    wp = np.asarray(wp_contacts, dtype=np.float32)
    wp[:, 0] = np.clip(wp[:, 0], -x_extent, x_extent)
    wp[:, 1] = np.clip(wp[:, 1], -y_extent, y_extent)
