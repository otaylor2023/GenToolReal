"""Procedurally generate brush waypoint-trajectory shards for dataset_0008."""

from __future__ import annotations

import argparse
import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np

TEMPLATES_BRUSH_STROKE_SWEEP = [
    "Sweep {material} into {destination}",
    "Brush {material} into {destination}",
    "Brush {material} toward {destination}",
    "Push {material} into {destination}",
    "Clear {material} into {destination}",
    "Move {material} to {destination}",
    "Sweep {material} off the surface into {destination}",
    "Brush {material} off the table into {destination}",
    "Clear {material} off the surface toward {destination}",
    "Sweep everything into {destination}",
    "Brush everything toward {destination}",
    "Clear the surface into {destination}",
    "Use the brush to sweep {material} into {destination}",
    "Use the brush to push {material} toward {destination}",
    "Use the brush to clear {material} into {destination}",
    "With the brush sweep {material} into {destination}",
    "Sweep {material} into {destination} with the brush",
    # tool-variant templates
    "Sweep {material} into {destination} with {tool}",
    "Use {tool} to sweep {material} into {destination}",
    "Use {tool} to push {material} toward {destination}",
    "Use {tool} to clear {material} into {destination}",
    "With {tool}, sweep {material} into {destination}",
    "With {tool}, push {material} into {destination}",
    "Brush {material} into {destination} with {tool}",
    "Sweep everything into {destination} with {tool}",
    "Clear the surface into {destination} with {tool}",
    # additional phrasings
    "Collect {material} into {destination}",
    "Gather {material} into {destination}",
    "Push {material} toward {destination}",
    "Get {material} into {destination}",
    "Sweep up {material} into {destination}",
    "Tidy {material} into {destination}",
    "Whisk {material} into {destination}",
    "Round up {material} into {destination}",
    "Funnel {material} into {destination}",
    "Direct {material} into {destination}",
    "Herd {material} toward {destination}",
    # adverb variants
    "Quickly sweep {material} into {destination}",
    "Carefully sweep {material} into {destination}",
    "Gently brush {material} toward {destination}",
    "Firmly push {material} into {destination}",
    "Slowly sweep {material} into {destination}",
]

MATERIALS_STROKE_SWEEP = [
    "the crumbs", "the debris", "the dust", "the powder", "the rice",
    "the beads", "the balls", "the coins", "the cereal", "the sand",
    "the dirt", "the chips", "the pieces", "the mess", "the scraps",
    "the seeds", "the pellets", "the shavings", "the flakes", "the crumbles",
    # additional nouns
    "the flour", "the sugar", "the salt", "the leaves", "the glitter",
    "the sawdust", "the ash", "the lint", "the soil", "the gravel",
    "the granola", "the oats", "the confetti", "the spice", "the breadcrumbs",
    "the eraser shavings", "the pencil shavings", "the trimmings",
    "the clippings", "the residue",
    # adjective / quantity variants
    "some flour", "some sugar", "some salt", "some sand", "some dirt",
    "some crumbs", "some debris", "some glitter", "the dry rice",
    "the loose dirt", "the fine dust", "the coarse sand",
    "the dry crumbs", "the loose powder", "a bit of dust",
    "a pile of crumbs", "the little crumbs", "the tiny seeds",
    "the spilled rice", "the scattered beads",
]

DESTINATIONS_STROKE_SWEEP = [
    "the dustpan", "the bowl", "the bin", "the box", "the tray",
    "the plate", "the cup", "the basket", "the container", "the pile",
    "the bag", "the pot", "the pan", "the bucket", "the dish",
    # additional destinations
    "the corner", "the edge", "the scoop", "the trash", "the trash bin",
    "the compost bin", "the side", "the gap", "the hole", "the tin",
    "the trough", "the pan corner",
]

TEMPLATES_BRUSH_PAINT_STROKE = [
    "Paint {destination} with the brush",
    "Apply {material} to {destination}",
    "Coat {destination} with {material}",
    "Spread {material} across {destination}",
    "Paint {material} onto {destination}",
    "Apply {material} across {destination} with the brush",
    "Use the brush to coat {destination} with {material}",
    "Brush {material} onto {destination}",
    "Cover {destination} with {material}",
    "Use the brush to apply {material} to {destination}",
    # tool-variant templates
    "Paint {destination} with {tool}",
    "Apply {material} to {destination} with {tool}",
    "Use {tool} to coat {destination} with {material}",
    "Use {tool} to paint {destination} with {material}",
    "Use {tool} to apply {material} to {destination}",
    "With {tool}, paint {destination} with {material}",
    "Brush {material} onto {destination} with {tool}",
    "Spread {material} across {destination} with {tool}",
    # additional phrasings
    "Glaze {destination} with {material}",
    "Layer {material} onto {destination}",
    "Smooth {material} over {destination}",
    "Wash {destination} with {material}",
    "Add {material} to {destination}",
    "Finish {destination} with {material}",
    "Stroke {material} across {destination}",
    "Daub {material} onto {destination}",
    "Lay {material} onto {destination}",
    "Roll {material} onto {destination}",
    "Spread some {material} across {destination}",
    # adverb variants
    "Carefully paint {destination} with {material}",
    "Smoothly apply {material} to {destination}",
    "Evenly coat {destination} with {material}",
    "Gently brush {material} onto {destination}",
    "Firmly press {material} into {destination}",
]

MATERIALS_PAINT_STROKE = [
    "the paint", "the glue", "the sauce", "the frosting", "the butter",
    "the varnish", "the oil", "the primer", "the stain", "the coating",
    "the paste", "the gel", "the cream", "the finish", "the lacquer",
    "the wax", "the sealant", "the adhesive", "the dye", "the ink",
    # additional nouns
    "the polish", "the shellac", "the gesso", "the acrylic", "the watercolor",
    "the oil paint", "the topcoat", "the basecoat", "the enamel", "the wash",
    "the plaster", "the mortar", "the spread", "the marinade", "the egg wash",
    "the syrup", "the honey", "the conditioner", "the resin", "the epoxy",
    # adjective / quantity variants
    "some paint", "some glue", "some sauce", "some butter", "some oil",
    "some ink", "some honey", "some syrup", "the thick paint",
    "the thin paint", "the white paint", "the red paint", "the blue paint",
    "the dark stain", "the clear varnish", "the warm butter",
    "the runny glue", "the sticky paste", "a bit of paint",
    "a thin layer of paint", "the wet ink",
]

DESTINATIONS_PAINT_STROKE = [
    "the wall", "the surface", "the board", "the canvas", "the wood",
    "the shelf", "the door", "the fence", "the table", "the panel",
    "the frame", "the box", "the tile", "the paper", "the card",
    # additional destinations
    "the cabinet", "the drawer", "the mug", "the vase", "the bowl",
    "the sculpture", "the model", "the sign", "the post", "the deck",
    "the floor", "the ceiling", "the bench", "the chair", "the lid",
    "the brick", "the stone", "the beam", "the plank", "the column",
    "the cabinet door", "the bookcase", "the easel",
]

TEMPLATES_BRUSH_PAINT_DIP = [
    "Dip the brush into {destination}",
    "Dip the brush in {destination}",
    "Dunk the brush into {destination}",
    "Load the brush from {destination}",
    "Lower the brush into {destination}",
    "Soak the brush in {destination}",
    "Dab the brush in {destination}",
    "Coat the brush in {destination}",
    "Get the brush wet in {destination}",
    "Wet the brush in {destination}",
    # tool-variant templates
    "Dip {tool} into {destination}",
    "Dip {tool} in {destination}",
    "Dunk {tool} into {destination}",
    "Load {tool} from {destination}",
    "Lower {tool} into {destination}",
    "Soak {tool} in {destination}",
    "Dab {tool} in {destination}",
    "Coat {tool} in {destination}",
    "Wet {tool} in {destination}",
    "Use {tool} to scoop from {destination}",
    # additional phrasings
    "Reload the brush in {destination}",
    "Refill the brush from {destination}",
    "Charge the brush in {destination}",
    "Plunge the brush into {destination}",
    "Re-wet the brush in {destination}",
    "Pick up some {destination} with the brush",
    "Scoop the brush through {destination}",
    "Stick the brush in {destination}",
    "Submerge the brush in {destination}",
    "Rinse the brush in {destination}",
    # adverb variants
    "Quickly dip the brush into {destination}",
    "Gently dip the brush into {destination}",
    "Carefully load the brush from {destination}",
    "Lightly dip the brush in {destination}",
]

DESTINATIONS_PAINT_DIP = [
    "the paint", "the paint pot", "the paint bucket", "the paint tray",
    "the bucket", "the pot", "the bowl", "the can", "the tray", "the jar",
    "the cup", "the container", "the glue", "the ink", "the dye",
    # additional destinations
    "the well", "the palette", "the basin", "the reservoir", "the dish",
    "the saucer", "the trough", "the water cup", "the rinse cup",
    "the paint can", "the paint jar", "the paint dish", "the mixing bowl",
    "the glaze", "the sauce", "the oil", "the wax", "the syrup",
    "the watercolor well", "the ink well",
]

TEMPLATES_BRUSH_SCRUB = [
    "Scrub {destination} with the brush",
    "Scrub {destination} clean",
    "Clean {destination} with the brush",
    "Scrub {material} off {destination}",
    "Scrub {material} from {destination}",
    "Use the brush to scrub {destination}",
    "Brush {destination}",
    "Clean {material} off {destination} with the brush",
    "Remove {material} from {destination} by scrubbing",
    "Scrub away {material} from {destination}",
    # tool-variant templates
    "Scrub {destination} with {tool}",
    "Clean {destination} with {tool}",
    "Use {tool} to scrub {destination}",
    "Scrub {material} off {destination} with {tool}",
    "Clean {material} off {destination} with {tool}",
    "With {tool}, scrub {material} off {destination}",
    "With {tool}, clean {destination}",
    "Remove {material} from {destination} with {tool}",
    # additional phrasings
    "Scour {destination} with the brush",
    "Buff {destination} with the brush",
    "Polish {destination} with the brush",
    "Work {material} out of {destination}",
    "Scrub at {destination}",
    "Get {material} off {destination}",
    "Wipe {material} off {destination} with the brush",
    "Work over {destination} with the brush",
    "Rub {destination} clean with the brush",
    "Take {material} off {destination}",
    # adverb variants
    "Vigorously scrub {destination} with the brush",
    "Carefully scrub {material} off {destination}",
    "Firmly scrub {destination} with the brush",
    "Quickly scrub {material} from {destination}",
    "Gently scrub {destination} with the brush",
]

MATERIALS_SCRUB = [
    "the stain", "the residue", "the grime", "the dirt", "the mark",
    "the mess", "the grease", "the buildup", "the crud", "the smudge",
    # additional nouns
    "the gunk", "the soot", "the rust", "the mold", "the mildew",
    "the soap scum", "the scuff", "the food", "the spill", "the gum",
    "the splatter", "the streak", "the layer of dust", "the dried paint",
    "the burned residue", "the caked dirt", "the wax", "the ink stain",
    "the oil spot", "the sticky residue",
    # adjective variants
    "the dark stain", "the dried stain", "the stubborn stain",
    "the heavy buildup", "the thick grime", "the old grease",
    "the dried mud", "the dirty smudge",
]

DESTINATIONS_SCRUB = [
    "the surface", "the table", "the pan", "the pot", "the tile",
    "the board", "the counter", "the plate", "the tray", "the shelf",
    "the wall", "the door", "the fence", "the wood", "the panel",
    "the frame", "the box",
    # additional destinations
    "the bowl", "the sink", "the basin", "the bathtub", "the floor",
    "the deck", "the grill", "the oven", "the lid", "the mat",
    "the cutting board", "the mirror", "the window", "the stove",
    "the dish", "the cup", "the saucepan", "the skillet",
    "the bathtub wall", "the shower wall",
]

TEMPLATES_BRUSH_PRESS = [
    "Dust {destination} with the brush",
    "Tap the brush against {destination}",
    "Dab the brush onto {destination}",
    "Apply the brush to {destination}",
    "Press the brush against {destination}",
    "Touch the brush to {destination}",
    "Dab {material} onto {destination} with the brush",
    "Lightly brush {destination}",
    "Gently brush {destination} with the brush",
    # tool-variant templates
    "Dust {destination} with {tool}",
    "Tap {tool} against {destination}",
    "Dab {tool} onto {destination}",
    "Apply {tool} to {destination}",
    "Press {tool} against {destination}",
    "Touch {tool} to {destination}",
    "Lightly brush {destination} with {tool}",
    "Gently brush {destination} with {tool}",
    "Dab {material} onto {destination} with {tool}",
    "Use {tool} to dust {destination}",
    # additional phrasings
    "Stamp the brush onto {destination}",
    "Pat the brush against {destination}",
    "Set the brush on {destination}",
    "Hold the brush against {destination}",
    "Mark {destination} with the brush",
    "Spot {material} onto {destination} with the brush",
    "Pat {material} onto {destination} with the brush",
    "Place the brush on {destination}",
    "Rest the brush on {destination}",
    "Stamp {material} onto {destination}",
    # adverb variants
    "Lightly tap the brush on {destination}",
    "Softly dab the brush onto {destination}",
    "Gently press the brush against {destination}",
    "Carefully dab {material} onto {destination}",
    "Lightly press {tool} against {destination}",
]

MATERIALS_PRESS = [
    "the paint", "the glue", "the ink", "the dye", "the coating",
    "the powder", "the dust",
    # additional nouns
    "the pigment", "the polish", "the lint", "the pollen", "the crumbs",
    "the seasoning", "the chalk", "the sealant", "the toner",
    "the gilding", "the wax", "the resin",
    # adjective variants
    "some paint", "some glue", "some ink", "some powder", "some dust",
    "the fine powder", "the loose dust",
]

DESTINATIONS_PRESS = [
    "the surface", "the object", "the top", "the face", "the side",
    "the edge", "the corner", "the center", "the spot", "the area",
    # additional concrete destinations
    "the button", "the lever", "the switch", "the sticker", "the label",
    "the badge", "the decal", "the cushion", "the pad", "the stamp",
    "the canvas", "the lid", "the tile", "the cap", "the patch",
    "the seal", "the dot", "the marking", "the keycap", "the cover",
]

VERTICAL_CAPABLE_PAINT_DESTINATIONS = frozenset({
    "the wall", "the door", "the fence",
    "the canvas", "the board", "the panel", "the frame",
    "the wood", "the box", "the tile",
    # additional vertical-friendly surfaces
    "the cabinet", "the cabinet door", "the sign", "the post",
    "the brick", "the stone", "the column", "the beam",
    "the plank", "the bookcase", "the easel", "the deck",
})

# Per-movement tool name pools (kept disjoint enough that nonsense pairings like
# "Dip the broom in the paint" do not appear). The default `TOOL_LABEL` is
# preserved for backwards compatibility, but all generators now sample a tool
# from their movement-specific list when the chosen template references {tool}.
TOOLS_STROKE_SWEEP = [
    "the brush", "the paintbrush", "the bristle brush", "the wide brush",
    "the flat brush", "the soft brush", "the stiff brush", "the scrub brush",
    "the cleaning brush", "the dust brush", "the bench brush",
    "the hand brush", "the broom", "the whisk brush", "the push broom",
]

TOOLS_PAINT_DIP = [
    "the brush", "the paintbrush", "the bristle brush", "the wide brush",
    "the small brush", "the flat brush", "the round brush",
    "the fine brush", "the soft brush", "the detail brush",
    "the foam brush", "the trim brush", "the artist's brush",
]

TOOLS_PAINT_STROKE = [
    "the brush", "the paintbrush", "the bristle brush", "the wide brush",
    "the small brush", "the flat brush", "the round brush",
    "the fine brush", "the soft brush", "the detail brush",
    "the foam brush", "the trim brush", "the artist's brush",
    "the angled brush",
]

TOOLS_SCRUB = [
    "the brush", "the scrub brush", "the cleaning brush", "the stiff brush",
    "the bristle brush", "the wire brush", "the bench brush",
    "the hand brush", "the wide brush", "the small brush",
    "the heavy brush", "the kitchen brush",
]

TOOLS_PRESS = [
    "the brush", "the paintbrush", "the soft brush", "the detail brush",
    "the fine brush", "the small brush", "the foam brush",
    "the round brush", "the duster", "the dust brush",
    "the dabbing brush", "the powder brush",
]

MOVEMENT_TYPES = ("stroke_sweep", "paint_dip", "paint_stroke", "scrub", "press")
TABLE_LABEL = "table surface center"
TOOL_LABEL = "the brush"
TOOLS_BY_MOVEMENT: dict[str, list[str]] = {
    "stroke_sweep": TOOLS_STROKE_SWEEP,
    "paint_dip": TOOLS_PAINT_DIP,
    "paint_stroke": TOOLS_PAINT_STROKE,
    "scrub": TOOLS_SCRUB,
    "press": TOOLS_PRESS,
}


def _pick_tool_and_format(
    rng: np.random.Generator,
    template: str,
    movement_token: str,
    *,
    material: str | None = None,
    destination: str | None = None,
) -> tuple[str, str]:
    """Pick a tool label compatible with the template and format the instruction.

    Templates that include ``{tool}`` get a movement-appropriate sampled tool
    name; templates that don't reference ``{tool}`` keep the literal "the brush"
    so the datapoint's tool_label always matches the spoken instruction.
    """
    if "{tool}" in template:
        pool = TOOLS_BY_MOVEMENT.get(movement_token, [TOOL_LABEL])
        tool_word = str(rng.choice(pool))
    else:
        tool_word = TOOL_LABEL
    instruction = _safe_format(
        template,
        material=material or "",
        destination=destination or "",
        tool=tool_word,
    )
    return tool_word, instruction


@dataclass
class BrushGenConfig:
    table_xyz_world: tuple[float, float, float] = (0.0, 0.0, 0.53)
    table_extent_m: float = 0.25
    approach_height_m_range: tuple[float, float] = (0.05, 0.20)
    contact_offset_m_range: tuple[float, float] = (0.001, 0.015)
    sweep_dist_m_range: tuple[float, float] = (0.06, 0.20)
    paint_dist_m_range: tuple[float, float] = (0.10, 0.25)
    scrub_radius_m_range: tuple[float, float] = (0.04, 0.10)
    tool_home_z_above_table_m_range: tuple[float, float] = (0.10, 0.25)
    surface_tilt_max_deg: float = 10.0
    paint_vertical_prob: float = 0.75
    paint_vertical_z_jitter_m_range: tuple[float, float] = (0.0, 0.25)
    scrub_vertical_prob: float = 0.7
    scrub_vertical_z_jitter_m_range: tuple[float, float] = (0.0, 0.25)
    table_clearance_m: float = 0.005
    sweep_destination_tilt_max_deg: float = 40.0
    dip_destination_tilt_max_deg: float = 30.0
    table_lift_jitter_m_range: tuple[float, float] = (0.0, 0.05)
    waypoint_z_jitter_m: float = 0.015


def _safe_format(template: str, **kwargs: str) -> str:
    names = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    filtered = {k: v for k, v in kwargs.items() if k in names}
    return template.format(**filtered)


def _sample_near_up_normal(rng: np.random.Generator, max_tilt_deg: float) -> np.ndarray:
    tilt = float(rng.uniform(0.0, max_tilt_deg)) * np.pi / 180.0
    yaw = float(rng.uniform(0.0, 2 * np.pi))
    return np.array(
        [np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), np.cos(tilt)],
        dtype=np.float32,
    )


def _sample_near_horizontal_normal(rng: np.random.Generator, max_tilt_deg: float) -> np.ndarray:
    yaw = float(rng.uniform(0.0, 2 * np.pi))
    tilt = float(rng.uniform(-max_tilt_deg, max_tilt_deg)) * np.pi / 180.0
    n = np.array(
        [np.cos(tilt) * np.cos(yaw), np.cos(tilt) * np.sin(yaw), np.sin(tilt)],
        dtype=np.float32,
    )
    return n / max(np.linalg.norm(n), 1e-6)


def _sample_xy(rng: np.random.Generator, extent: float) -> np.ndarray:
    return rng.uniform(-extent, extent, size=2).astype(np.float32)


def _sample_unit_vec(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3).astype(np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _orthonormal_basis_in_plane(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, dtype=np.float64).reshape(3)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, n) * n
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-6:
        u = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), n) * n
        u_norm = np.linalg.norm(u)
    u = (u / max(u_norm, 1e-6)).astype(np.float32)
    v = np.cross(n, u).astype(np.float32)
    v = v / max(np.linalg.norm(v), 1e-6)
    return u, v


def _sample_tool_pose(
    rng: np.random.Generator, cfg: BrushGenConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table_z = float(cfg.table_xyz_world[2])
    xy = _sample_xy(rng, cfg.table_extent_m)
    z_above = float(rng.uniform(*cfg.tool_home_z_above_table_m_range))
    contact = np.array([xy[0], xy[1], table_z + z_above], dtype=np.float32)
    normal = _sample_unit_vec(rng)
    surf = _sample_unit_vec(rng)
    surf = surf - np.dot(surf, normal) * normal
    sn = np.linalg.norm(surf)
    if sn < 1e-6:
        u, v = _orthonormal_basis_in_plane(normal)
        surf = u
    else:
        surf = (surf / sn).astype(np.float32)
    return contact, normal, surf


def _project_surface_dir(
    from_xyz: np.ndarray, to_xyz: np.ndarray, surface_normal: np.ndarray
) -> np.ndarray:
    raw = to_xyz.astype(np.float64) - from_xyz.astype(np.float64)
    n = surface_normal.astype(np.float64)
    raw = raw - np.dot(raw, n) * n
    return (raw / max(np.linalg.norm(raw), 1e-6)).astype(np.float32)


ToolHomePose = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _intermediate_pose(
    tool_home_xyz: np.ndarray,
    tool_home_normal: np.ndarray,
    tool_home_surface_dir: np.ndarray,
    first_xyz: np.ndarray,
    first_normal: np.ndarray,
    first_surface_dir: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_frac = float(rng.uniform(0.3, 0.5))
    ori_frac = float(rng.uniform(0.5, 0.8))
    pos = (
        (1.0 - pos_frac) * tool_home_xyz.astype(np.float64)
        + pos_frac * first_xyz.astype(np.float64)
    ).astype(np.float32)
    n = (1.0 - ori_frac) * tool_home_normal.astype(np.float64) + ori_frac * first_normal.astype(
        np.float64
    )
    n = (n / max(np.linalg.norm(n), 1e-9)).astype(np.float32)
    sd = (1.0 - ori_frac) * tool_home_surface_dir.astype(np.float64) + ori_frac * (
        first_surface_dir.astype(np.float64)
    )
    sd = sd - np.dot(sd, n) * n.astype(np.float64)
    sd_norm = np.linalg.norm(sd)
    if sd_norm < 1e-6:
        u, v = _orthonormal_basis_in_plane(n)
        sd = u
    else:
        sd = (sd / sd_norm).astype(np.float32)
    return pos, n, sd


def _jitter_orientation(
    normal: np.ndarray,
    surface_dir: np.ndarray,
    rng: np.random.Generator,
    max_tilt_rad: float = 0.20,
    max_yaw_rad: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly perturb a (normal, surface_dir) frame; keeps them orthogonal."""
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    n = n / max(np.linalg.norm(n), 1e-9)
    sd = np.asarray(surface_dir, dtype=np.float64).reshape(3)
    sd = sd - np.dot(sd, n) * n
    sd_norm = float(np.linalg.norm(sd))
    if sd_norm < 1e-6:
        u_basis, _ = _orthonormal_basis_in_plane(n.astype(np.float32))
        sd = np.asarray(u_basis, dtype=np.float64)
    else:
        sd = sd / sd_norm
    tilt = float(rng.uniform(0.0, max_tilt_rad))
    yaw = float(rng.uniform(0.0, 2 * np.pi))
    u_axis, v_axis = _orthonormal_basis_in_plane(n.astype(np.float32))
    tilt_dir = np.cos(yaw) * np.asarray(u_axis, dtype=np.float64) + np.sin(yaw) * np.asarray(
        v_axis, dtype=np.float64
    )
    new_n = np.cos(tilt) * n + np.sin(tilt) * tilt_dir
    new_n = new_n / max(np.linalg.norm(new_n), 1e-9)
    sd_proj = sd - np.dot(sd, new_n) * new_n
    sd_proj_norm = float(np.linalg.norm(sd_proj))
    if sd_proj_norm < 1e-6:
        nu, _ = _orthonormal_basis_in_plane(new_n.astype(np.float32))
        sd_proj = np.asarray(nu, dtype=np.float64)
    else:
        sd_proj = sd_proj / sd_proj_norm
    sd_yaw = float(rng.uniform(-max_yaw_rad, max_yaw_rad))
    cross = np.cross(new_n, sd_proj)
    new_sd = np.cos(sd_yaw) * sd_proj + np.sin(sd_yaw) * cross
    new_sd_norm = float(np.linalg.norm(new_sd))
    if new_sd_norm < 1e-6:
        new_sd = sd_proj
    else:
        new_sd = new_sd / new_sd_norm
    return new_n.astype(np.float32), new_sd.astype(np.float32)


def _apply_intermediate_wp0(
    waypoints: np.ndarray,
    tool_home_pose: ToolHomePose,
    first_xyz: np.ndarray,
    first_normal: np.ndarray,
    first_surface_dir: np.ndarray,
    rng: np.random.Generator,
) -> None:
    th_xyz, th_n, th_sd = tool_home_pose
    pos, n, sd = _intermediate_pose(
        th_xyz, th_n, th_sd, first_xyz, first_normal, first_surface_dir, rng
    )
    waypoints[0, 0:3] = pos
    waypoints[0, 3:6] = n
    waypoints[0, 6:9] = sd


def _jitter_waypoint_z(
    wp_contacts: np.ndarray,
    rng: np.random.Generator,
    max_jitter_m: float,
    floor_z: float,
) -> None:
    """Add per-waypoint z noise, capped so z >= floor_z. Mutates in place."""
    if max_jitter_m <= 0.0:
        return
    jitter = rng.uniform(-max_jitter_m, max_jitter_m, size=wp_contacts.shape[0]).astype(
        np.float32
    )
    wp_contacts[:, 2] = np.maximum(wp_contacts[:, 2] + jitter, np.float32(floor_z))


def _lift_above_floor(
    wp_contacts: np.ndarray,
    extra_anchors: list[np.ndarray],
    floor_z: float,
) -> float:
    """Shift wp_contacts (and any extra anchor arrays) up so min z >= floor_z.

    Mutates `wp_contacts` and every array in `extra_anchors` in place. Returns
    the applied shift (0 if no shift was needed).
    """
    min_z = float(wp_contacts[:, 2].min())
    if min_z >= floor_z:
        return 0.0
    shift = float(floor_z - min_z)
    wp_contacts[:, 2] += shift
    for anchor in extra_anchors:
        anchor[2] = float(anchor[2]) + shift
    return shift


def _waypoints_from_contacts(
    wp_contacts: np.ndarray,
    surface_normal: np.ndarray,
    surface_dir: np.ndarray | None = None,
    surface_dirs: np.ndarray | None = None,
) -> np.ndarray:
    wp = np.zeros((6, 9), dtype=np.float32)
    wp_normal = (-surface_normal).astype(np.float32)
    for i in range(6):
        wp[i, 0:3] = wp_contacts[i]
        wp[i, 3:6] = wp_normal
        if surface_dirs is not None:
            wp[i, 6:9] = surface_dirs[i]
        else:
            sd = surface_dir if surface_dir is not None else np.zeros(3, dtype=np.float32)
            wp[i, 6:9] = sd
    return wp


def gen_stroke_sweep(
    rng: np.random.Generator,
    cfg: BrushGenConfig,
    table_normal: np.ndarray,
    tool_home_pose: ToolHomePose,
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    material_word = str(rng.choice(MATERIALS_STROKE_SWEEP))
    destination_word = str(rng.choice(DESTINATIONS_STROKE_SWEEP))

    material_xy = _sample_xy(rng, cfg.table_extent_m)
    dist = float(rng.uniform(*cfg.sweep_dist_m_range))
    theta = float(rng.uniform(0.0, 2 * np.pi))
    destination_xy = material_xy + dist * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    destination_xy = np.clip(destination_xy, -cfg.table_extent_m, cfg.table_extent_m)

    material_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    destination_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    material_xyz = np.array([material_xy[0], material_xy[1], material_z], dtype=np.float32)
    destination_xyz = np.array(
        [destination_xy[0], destination_xy[1], destination_z], dtype=np.float32
    )

    surface_normal = table_normal.astype(np.float32).copy()
    material_normal = surface_normal.copy()
    destination_normal = _sample_near_up_normal(
        rng, cfg.sweep_destination_tilt_max_deg
    )
    surface_dir = _project_surface_dir(destination_xyz, material_xyz, surface_normal)

    approach_h = float(rng.uniform(*cfg.approach_height_m_range))
    lift_h = float(rng.uniform(*cfg.approach_height_m_range))
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    approach_offset = surface_normal * approach_h
    lift_offset = surface_normal * lift_h
    contact_offset = surface_normal * contact_h

    sweep_vec = destination_xyz - material_xyz
    sweep_vec = sweep_vec - np.dot(sweep_vec, surface_normal) * surface_normal
    sweep_norm = float(np.linalg.norm(sweep_vec))
    sweep_unit = (sweep_vec / sweep_norm).astype(np.float32) if sweep_norm > 1e-6 else surface_dir

    behind_offset_m = float(rng.uniform(0.025, 0.06))
    behind_xyz = (material_xyz - sweep_unit * behind_offset_m).astype(np.float32)

    mid_frac = float(rng.uniform(0.4, 0.6))
    midpoint_xyz = (1.0 - mid_frac) * material_xyz + mid_frac * destination_xyz

    first_xyz = (behind_xyz + contact_offset).astype(np.float32)
    wp_contacts = np.stack(
        [
            first_xyz,
            first_xyz,
            material_xyz + contact_offset,
            midpoint_xyz + contact_offset,
            destination_xyz + contact_offset,
            destination_xyz + lift_offset,
        ],
        axis=0,
    ).astype(np.float32)
    _jitter_waypoint_z(
        wp_contacts,
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    waypoints = _waypoints_from_contacts(wp_contacts, surface_normal, surface_dir=surface_dir)
    _apply_intermediate_wp0(
        waypoints,
        tool_home_pose,
        wp_contacts[1],
        waypoints[1, 3:6],
        waypoints[1, 6:9],
        rng,
    )

    template = str(rng.choice(TEMPLATES_BRUSH_STROKE_SWEEP))
    tool_word, instruction = _pick_tool_and_format(
        rng,
        template,
        "stroke_sweep",
        material=material_word,
        destination=destination_word,
    )
    return {
        "movement_token": "stroke_sweep",
        "instruction": instruction,
        "tool_label": tool_word,
        "has_material": True,
        "material_label": material_word,
        "material_xyz_world": material_xyz.tolist(),
        "material_normal": material_normal.tolist(),
        "has_destination": True,
        "destination_label": destination_word,
        "destination_xyz_world": destination_xyz.tolist(),
        "destination_normal": destination_normal.tolist(),
        "waypoints": waypoints.reshape(6, 9).tolist(),
    }


def gen_paint_dip(
    rng: np.random.Generator,
    cfg: BrushGenConfig,
    table_normal: np.ndarray,
    tool_home_pose: ToolHomePose,
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    destination_word = str(rng.choice(DESTINATIONS_PAINT_DIP))
    target_xy = _sample_xy(rng, cfg.table_extent_m)
    target_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    target_xyz = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
    destination_normal = _sample_near_up_normal(
        rng, cfg.dip_destination_tilt_max_deg
    )
    trajectory_normal = table_normal.astype(np.float32).copy()

    approach_h = float(rng.uniform(*cfg.approach_height_m_range))
    lift_h = float(rng.uniform(*cfg.approach_height_m_range))
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    approach_offset = trajectory_normal * approach_h
    lift_offset = trajectory_normal * lift_h
    contact_offset = trajectory_normal * contact_h

    u, v = _orthonormal_basis_in_plane(trajectory_normal)
    phi = float(rng.uniform(0.0, 2 * np.pi))
    surface_dir = (np.cos(phi) * u + np.sin(phi) * v).astype(np.float32)

    approach_xyz = (target_xyz + approach_offset).astype(np.float32)
    dwell_jitter_mag = 0.004
    dwell_offsets = np.stack(
        [
            dwell_jitter_mag * (np.cos(t) * u + np.sin(t) * v)
            for t in rng.uniform(0.0, 2 * np.pi, size=3)
        ],
        axis=0,
    ).astype(np.float32)
    wp_contacts = np.stack(
        [
            approach_xyz,
            approach_xyz,
            target_xyz + contact_offset + dwell_offsets[0],
            target_xyz + contact_offset + dwell_offsets[1],
            target_xyz + contact_offset + dwell_offsets[2],
            target_xyz + lift_offset,
        ],
        axis=0,
    ).astype(np.float32)
    _jitter_waypoint_z(
        wp_contacts,
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    waypoints = _waypoints_from_contacts(wp_contacts, trajectory_normal, surface_dir=surface_dir)
    base_normal = waypoints[1, 3:6].copy()
    for i in (2, 3, 4):
        jn, jsd = _jitter_orientation(base_normal, surface_dir, rng)
        waypoints[i, 3:6] = jn
        waypoints[i, 6:9] = jsd
    _apply_intermediate_wp0(
        waypoints,
        tool_home_pose,
        wp_contacts[1],
        waypoints[1, 3:6],
        waypoints[1, 6:9],
        rng,
    )

    template = str(rng.choice(TEMPLATES_BRUSH_PAINT_DIP))
    tool_word, instruction = _pick_tool_and_format(
        rng,
        template,
        "paint_dip",
        destination=destination_word,
    )
    return {
        "movement_token": "paint_dip",
        "instruction": instruction,
        "tool_label": tool_word,
        "has_material": False,
        "material_label": None,
        "material_xyz_world": None,
        "material_normal": None,
        "has_destination": True,
        "destination_label": destination_word,
        "destination_xyz_world": target_xyz.tolist(),
        "destination_normal": destination_normal.tolist(),
        "waypoints": waypoints.reshape(6, 9).tolist(),
    }


def gen_paint_stroke(
    rng: np.random.Generator,
    cfg: BrushGenConfig,
    table_normal: np.ndarray,
    tool_home_pose: ToolHomePose,
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    destination_word = str(rng.choice(DESTINATIONS_PAINT_STROKE))
    material_word = str(rng.choice(MATERIALS_PAINT_STROKE))
    is_vertical = float(rng.random()) < cfg.paint_vertical_prob

    if is_vertical:
        destination_normal = _sample_near_horizontal_normal(rng, cfg.surface_tilt_max_deg)
        z_offset = float(rng.uniform(*cfg.paint_vertical_z_jitter_m_range))
        center = np.array(
            [*_sample_xy(rng, cfg.table_extent_m), table_z + z_offset], dtype=np.float32
        )
    else:
        destination_normal = table_normal.astype(np.float32).copy()
        z_offset = float(rng.uniform(*cfg.table_lift_jitter_m_range))
        center = np.array(
            [*_sample_xy(rng, cfg.table_extent_m), table_z + z_offset], dtype=np.float32
        )

    u, v = _orthonormal_basis_in_plane(destination_normal)
    phi = float(rng.uniform(0.0, 2 * np.pi))
    stroke_dir = (np.cos(phi) * u + np.sin(phi) * v).astype(np.float32)
    half = float(rng.uniform(*cfg.paint_dist_m_range)) / 2.0
    start_pt = center - half * stroke_dir
    end_pt = center + half * stroke_dir
    surface_dir = (-stroke_dir).astype(np.float32)

    approach_h = float(rng.uniform(*cfg.approach_height_m_range))
    lift_h = float(rng.uniform(*cfg.approach_height_m_range))
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    approach_offset = destination_normal * approach_h
    lift_offset = destination_normal * lift_h
    contact_offset = destination_normal * contact_h

    mid_frac = float(rng.uniform(0.4, 0.6))
    midpoint = (1.0 - mid_frac) * start_pt + mid_frac * end_pt
    start_approach = (start_pt + approach_offset).astype(np.float32)
    wp_contacts = np.stack(
        [
            start_approach,
            start_approach,
            start_pt + contact_offset,
            midpoint + contact_offset,
            end_pt + contact_offset,
            end_pt + lift_offset,
        ],
        axis=0,
    ).astype(np.float32)
    _lift_above_floor(
        wp_contacts,
        [center],
        floor_z=table_z + cfg.table_clearance_m,
    )
    _jitter_waypoint_z(
        wp_contacts,
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    waypoints = _waypoints_from_contacts(wp_contacts, destination_normal, surface_dir=surface_dir)
    _apply_intermediate_wp0(
        waypoints,
        tool_home_pose,
        wp_contacts[1],
        waypoints[1, 3:6],
        waypoints[1, 6:9],
        rng,
    )

    template = str(rng.choice(TEMPLATES_BRUSH_PAINT_STROKE))
    tool_word, instruction = _pick_tool_and_format(
        rng,
        template,
        "paint_stroke",
        material=material_word,
        destination=destination_word,
    )
    return {
        "movement_token": "paint_stroke",
        "instruction": instruction,
        "tool_label": tool_word,
        "has_material": False,
        "material_label": None,
        "material_xyz_world": None,
        "material_normal": None,
        "has_destination": True,
        "destination_label": destination_word,
        "destination_xyz_world": center.tolist(),
        "destination_normal": destination_normal.tolist(),
        "waypoints": waypoints.reshape(6, 9).tolist(),
    }


def gen_scrub(
    rng: np.random.Generator,
    cfg: BrushGenConfig,
    table_normal: np.ndarray,
    tool_home_pose: ToolHomePose,
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    destination_word = str(rng.choice(DESTINATIONS_SCRUB))
    material_word = str(rng.choice(MATERIALS_SCRUB))

    goal_xy = _sample_xy(rng, cfg.table_extent_m)
    half_len = float(rng.uniform(*cfg.scrub_radius_m_range))
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    lift_h = float(rng.uniform(*cfg.approach_height_m_range))

    is_vertical = float(rng.random()) < cfg.scrub_vertical_prob
    if is_vertical:
        destination_normal = _sample_near_horizontal_normal(rng, cfg.surface_tilt_max_deg)
        z_offset = float(rng.uniform(*cfg.scrub_vertical_z_jitter_m_range))
        goal_xyz = np.array([goal_xy[0], goal_xy[1], table_z + z_offset], dtype=np.float32)
    else:
        destination_normal = table_normal.astype(np.float32).copy()
        z_offset = float(rng.uniform(*cfg.table_lift_jitter_m_range))
        goal_xyz = np.array(
            [goal_xy[0], goal_xy[1], table_z + z_offset], dtype=np.float32
        )
    u, v = _orthonormal_basis_in_plane(destination_normal)
    contact_offset = destination_normal * contact_h
    lift_offset = destination_normal * lift_h

    phi = float(rng.uniform(0.0, 2 * np.pi))
    axis_in_plane = (np.cos(phi) * u + np.sin(phi) * v).astype(np.float32)
    tangent_in_plane = (-np.sin(phi) * u + np.cos(phi) * v).astype(np.float32)
    p_a1 = half_len * axis_in_plane
    tangent_offset = float(rng.uniform(0.01, 0.03)) * float(rng.choice([1.0, -1.0]))
    p_a2 = p_a1 + tangent_offset * tangent_in_plane

    a1 = (goal_xyz + p_a1 + contact_offset).astype(np.float32)
    b1 = (goal_xyz - p_a1 + contact_offset).astype(np.float32)
    a2 = (goal_xyz + p_a2 + contact_offset).astype(np.float32)
    b2 = (goal_xyz - p_a2 + contact_offset).astype(np.float32)
    lift = (b2 + lift_offset).astype(np.float32)

    brush_surface_dir = _project_surface_dir(a1, b1, destination_normal)

    max_yaw_rad = float(np.deg2rad(10.0))
    jittered_dirs = np.zeros((6, 3), dtype=np.float32)
    for i in range(6):
        yaw = float(rng.uniform(-max_yaw_rad, max_yaw_rad))
        cos_y = float(np.cos(yaw))
        sin_y = float(np.sin(yaw))
        cross = np.cross(destination_normal, brush_surface_dir)
        jittered_dirs[i] = (
            brush_surface_dir * cos_y + cross.astype(np.float32) * sin_y
        ).astype(np.float32)

    wp_contacts = np.stack([a1, a1, b1, a2, b2, lift], axis=0).astype(np.float32)
    _lift_above_floor(
        wp_contacts,
        [goal_xyz],
        floor_z=table_z + cfg.table_clearance_m,
    )
    _jitter_waypoint_z(
        wp_contacts,
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    waypoints = _waypoints_from_contacts(
        wp_contacts, destination_normal, surface_dirs=jittered_dirs
    )
    _apply_intermediate_wp0(
        waypoints,
        tool_home_pose,
        wp_contacts[0],
        waypoints[1, 3:6],
        waypoints[1, 6:9],
        rng,
    )

    template = str(rng.choice(TEMPLATES_BRUSH_SCRUB))
    tool_word, instruction = _pick_tool_and_format(
        rng,
        template,
        "scrub",
        material=material_word,
        destination=destination_word,
    )
    return {
        "movement_token": "scrub",
        "instruction": instruction,
        "tool_label": tool_word,
        "has_material": False,
        "material_label": None,
        "material_xyz_world": None,
        "material_normal": None,
        "has_destination": True,
        "destination_label": destination_word,
        "destination_xyz_world": goal_xyz.tolist(),
        "destination_normal": destination_normal.tolist(),
        "waypoints": waypoints.reshape(6, 9).tolist(),
    }


def gen_press(
    rng: np.random.Generator,
    cfg: BrushGenConfig,
    table_normal: np.ndarray,
    tool_home_pose: ToolHomePose,
) -> dict[str, Any]:
    table_z = float(cfg.table_xyz_world[2])
    destination_word = str(rng.choice(DESTINATIONS_PRESS))
    material_word = str(rng.choice(MATERIALS_PRESS))

    target_xy = _sample_xy(rng, cfg.table_extent_m)
    target_z = table_z + float(rng.uniform(*cfg.table_lift_jitter_m_range))
    target_xyz = np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)
    destination_normal = table_normal.astype(np.float32).copy()

    approach_h = float(rng.uniform(*cfg.approach_height_m_range))
    lift_h = float(rng.uniform(*cfg.approach_height_m_range))
    contact_h = float(rng.uniform(*cfg.contact_offset_m_range))
    approach_offset = destination_normal * approach_h
    lift_offset = destination_normal * lift_h
    contact_offset = destination_normal * contact_h

    u, v = _orthonormal_basis_in_plane(destination_normal)
    phi = float(rng.uniform(0.0, 2 * np.pi))
    surface_dir = (np.cos(phi) * u + np.sin(phi) * v).astype(np.float32)

    approach_xyz = (target_xyz + approach_offset).astype(np.float32)
    lift_xyz = (target_xyz + lift_offset).astype(np.float32)
    lift_jitter_mag = 0.008
    lift_offsets = np.stack(
        [
            lift_jitter_mag * (np.cos(t) * u + np.sin(t) * v)
            for t in rng.uniform(0.0, 2 * np.pi, size=3)
        ],
        axis=0,
    ).astype(np.float32)
    wp_contacts = np.stack(
        [
            approach_xyz,
            approach_xyz,
            target_xyz + contact_offset,
            lift_xyz + lift_offsets[0],
            lift_xyz + lift_offsets[1],
            lift_xyz + lift_offsets[2],
        ],
        axis=0,
    ).astype(np.float32)
    _jitter_waypoint_z(
        wp_contacts,
        rng,
        cfg.waypoint_z_jitter_m,
        floor_z=table_z + cfg.table_clearance_m,
    )
    waypoints = _waypoints_from_contacts(wp_contacts, destination_normal, surface_dir=surface_dir)
    _apply_intermediate_wp0(
        waypoints,
        tool_home_pose,
        wp_contacts[1],
        waypoints[1, 3:6],
        waypoints[1, 6:9],
        rng,
    )

    template = str(rng.choice(TEMPLATES_BRUSH_PRESS))
    tool_word, instruction = _pick_tool_and_format(
        rng,
        template,
        "press",
        material=material_word,
        destination=destination_word,
    )
    return {
        "movement_token": "press",
        "instruction": instruction,
        "tool_label": tool_word,
        "has_material": False,
        "material_label": None,
        "material_xyz_world": None,
        "material_normal": None,
        "has_destination": True,
        "destination_label": destination_word,
        "destination_xyz_world": target_xyz.tolist(),
        "destination_normal": destination_normal.tolist(),
        "waypoints": waypoints.reshape(6, 9).tolist(),
    }


_GENERATORS = {
    "stroke_sweep": gen_stroke_sweep,
    "paint_dip": gen_paint_dip,
    "paint_stroke": gen_paint_stroke,
    "scrub": gen_scrub,
    "press": gen_press,
}


def _datapoint_rng(seed: int, shard_idx: int, datapoint_index: int) -> np.random.Generator:
    mixed = (int(seed) << 32) ^ (int(shard_idx) << 16) ^ int(datapoint_index)
    return np.random.default_rng(mixed & 0xFFFFFFFFFFFFFFFF)


def build_datapoint(
    rng: np.random.Generator,
    cfg: BrushGenConfig,
    *,
    shard_id: str,
    datapoint_index: int,
    movement_token: str,
) -> dict[str, Any]:
    table_normal = _sample_near_up_normal(rng, cfg.surface_tilt_max_deg)
    tool_contact, tool_normal, tool_surface_dir = _sample_tool_pose(rng, cfg)
    gen_fn = _GENERATORS[movement_token]
    body = gen_fn(
        rng,
        cfg,
        table_normal,
        (tool_contact, tool_normal, tool_surface_dir),
    )

    datapoint_id = f"{shard_id}_{datapoint_index:06d}"
    return {
        "datapoint_id": datapoint_id,
        "datapoint_index": int(datapoint_index),
        "movement_token": movement_token,
        "instruction": body["instruction"],
        "tool_label": body.get("tool_label", TOOL_LABEL),
        "tool_contact_xyz_world": tool_contact.tolist(),
        "tool_current_normal": tool_normal.tolist(),
        "tool_current_surface_dir": tool_surface_dir.tolist(),
        "has_material": bool(body["has_material"]),
        "material_label": body["material_label"],
        "material_xyz_world": body["material_xyz_world"],
        "material_normal": body["material_normal"],
        "has_destination": bool(body["has_destination"]),
        "destination_label": body["destination_label"],
        "destination_xyz_world": body["destination_xyz_world"],
        "destination_normal": body["destination_normal"],
        "table_label": TABLE_LABEL,
        "table_xyz_world": list(cfg.table_xyz_world),
        "table_normal": table_normal.tolist(),
        "waypoints": body["waypoints"],
    }


def build_shard(
    *,
    shard_idx: int,
    seed: int,
    datapoints_per_shard: int,
    cfg: BrushGenConfig,
) -> dict[str, Any]:
    shard_id = f"brush_procedural_{shard_idx:04d}"
    datapoints: list[dict[str, Any]] = []
    for dp_idx in range(datapoints_per_shard):
        movement = MOVEMENT_TYPES[dp_idx % len(MOVEMENT_TYPES)]
        rng = _datapoint_rng(seed, shard_idx, dp_idx)
        datapoints.append(
            build_datapoint(
                rng,
                cfg,
                shard_id=shard_id,
                datapoint_index=dp_idx,
                movement_token=movement,
            )
        )
    return {
        "dataset_id": "dataset_0008_brush_procedural",
        "shard_id": shard_id,
        "scene_id": shard_id,
        "generator": "brush_procedural_v1",
        "seed": int(seed),
        "num_datapoints": int(datapoints_per_shard),
        "datapoints": datapoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build brush procedural trajectory shards.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/datasets/dataset_0008_brush_procedural/shards",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--datapoints_per_shard", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = BrushGenConfig()
    summary: dict[str, Any] = {
        "dataset_id": "dataset_0008_brush_procedural",
        "num_shards": int(args.num_shards),
        "datapoints_per_shard": int(args.datapoints_per_shard),
        "seed": int(args.seed),
        "shard_paths": [],
    }

    for shard_idx in range(int(args.num_shards)):
        shard = build_shard(
            shard_idx=shard_idx,
            seed=int(args.seed),
            datapoints_per_shard=int(args.datapoints_per_shard),
            cfg=cfg,
        )
        out_path = out_dir / f"{shard['shard_id']}_shard.json"
        out_path.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        summary["shard_paths"].append(str(out_path))
        print(f"Wrote {out_path} ({shard['num_datapoints']} datapoints)")

    summary_path = out_dir.parent / "dataset_0008_brush_procedural_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
