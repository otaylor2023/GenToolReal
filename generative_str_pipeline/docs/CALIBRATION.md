# Calibration and scale (`cam_K`, `T_RC`, MDE alignment)

## Robot hardware

- **`cam_K.txt`**: intrinsic matrix from camera calibration (ZED / RealSense / etc.), saved as 3×3 ASCII (row-major or standard FoundationPose layout—match your `extract_poses.py` reader).
- **`T_RC.txt`**: rigid transform from **robot base (or frame R)** to **camera**, in the convention expected by the [FoundationPose fork](https://github.com/kushal2000/FoundationPose). Keep one authoritative definition and reuse for all runs from the same mount.

## DexToolBench parity

When comparing to benchmark demos, you may use the dataset’s `cam_K.txt` from `dextoolbench/data/.../` so intrinsics match recorded RGB. **`T_RC` must still be consistent** with how poses are expressed in `poses.json` (robot frame as SimToolReal expects).

## MDE alignment (t = 0 ground truth)

Monocular depth is **affine-ambiguous**. After each MDE forward pass:

1. Build a **valid mask** on frame 0 (depth > 0, optional object mask).
2. Fit **scale `s` and shift `b`** minimizing \(\| s \cdot \hat{d}_0 + b - d^{\mathrm{GT}}_0 \|\) on that mask (least squares or robust variant).
3. Apply the **same** `s`, `b` to every MDE map for `t \geq 1`.
4. Write **`depth/frame_0000.png` from the camera**, not from MDE.

Log `s`, `b`, and mask policy to `meta/mde_alignment.json` for every run.

## Simulation-only testing

You may use a **synthetic capture** from Isaac / rendered RGB-D with known `cam_K` and identity or sim-derived `T_RC`. Document that runs are **sim-referenced** so numbers are not mixed with physical robot runs.
