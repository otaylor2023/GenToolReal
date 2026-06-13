from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
def smooth_poses_json(
    poses_in: Path,
    poses_out: Path,
    *,
    window: int = 11,
    polyorder: int = 2,
) -> None:
    data = json.loads(poses_in.read_text(encoding="utf-8"))
    poses = np.asarray(data["poses_cam"], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("poses_cam must be N×7 [x,y,z,qx,qy,qz,qw]")
    n = poses.shape[0]
    wlen = min(window, n if n % 2 == 1 else n - 1)
    if wlen < 3:
        wlen = 3
    if wlen % 2 == 0:
        wlen -= 1
    wlen = max(3, wlen)

    xyz = poses[:, :3]
    quat = poses[:, 3:7]
    xyz_s = savgol_filter(xyz, window_length=wlen, polyorder=min(polyorder, wlen - 1), axis=0, mode="nearest")

    # Component-wise Savitzky–Golay on quaternions then re-normalize (MVP).
    qu_s = savgol_filter(
        quat,
        window_length=wlen,
        polyorder=min(polyorder, wlen - 1),
        axis=0,
        mode="nearest",
    )
    norms = np.linalg.norm(qu_s, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    qu_s = qu_s / norms
    # Fix sign flips for continuity
    for i in range(1, n):
        if np.dot(qu_s[i], qu_s[i - 1]) < 0:
            qu_s[i] *= -1.0

    out_xyzw = np.concatenate([xyz_s, qu_s], axis=1)
    out = dict(data)
    out["poses_cam"] = out_xyzw.tolist()
    if "poses_robot" in data:
        ro = np.asarray(data["poses_robot"], dtype=np.float64)
        if ro.shape == poses.shape:
            xyzr = ro[:, :3]
            qr = ro[:, 3:7]
            xyzr_s = savgol_filter(
                xyzr, window_length=wlen, polyorder=min(polyorder, wlen - 1), axis=0, mode="nearest"
            )
            qr_s = savgol_filter(
                qr,
                window_length=wlen,
                polyorder=min(polyorder, wlen - 1),
                axis=0,
                mode="nearest",
            )
            qr_s /= np.maximum(np.linalg.norm(qr_s, axis=1, keepdims=True), 1e-9)
            for i in range(1, n):
                if np.dot(qr_s[i], qr_s[i - 1]) < 0:
                    qr_s[i] *= -1.0
            out["poses_robot"] = np.concatenate([xyzr_s, qr_s], axis=1).tolist()

    poses_out.parent.mkdir(parents=True, exist_ok=True)
    poses_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    meta = poses_out.parent / "smoothing.json"
    meta.write_text(
        json.dumps({"window": wlen, "polyorder": polyorder, "source": str(poses_in)}, indent=2) + "\n",
        encoding="utf-8",
    )
