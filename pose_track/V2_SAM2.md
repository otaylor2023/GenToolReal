# v2: SAM2 masks and re-initialization (deferred)

Optional robustness pass (not required for MVP):

1. Propagate a SAM2 mask from frame 0 through generated RGB frames (or refresh every K frames).
2. When translation jerk or reprojection error exceeds thresholds, call FoundationPose `register` again with the new mask and continue `track_one`.

This stays out of `pose_track` MVP to avoid extra conda stacks and tuning surface area.
