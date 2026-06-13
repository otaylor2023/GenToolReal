#!/usr/bin/env bash
# Create empty stage folders for one pipeline run.
set -euo pipefail
RUN="${1:?run root path}"
mkdir -p "$RUN"/{capture/rgb,capture/depth,capture/masks,video_gen/rgb,mde_raw,mesh,aligned_rgbd/rgb,aligned_rgbd/depth,foundationpose,meta}
