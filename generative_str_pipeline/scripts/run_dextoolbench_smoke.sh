#!/usr/bin/env bash
# Smoke-test SimToolBench visualization / eval when trajectories and Isaac are available.
# Usage: from simtoolreal repo root with env activated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../simtoolreal" && pwd)"
cd "$ROOT"

export DEMO_CATEGORY="${DEMO_CATEGORY:-hammer}"
export DEMO_OBJECT="${DEMO_OBJECT:-claw_hammer}"
export DEMO_TASK="${DEMO_TASK:-swing_down}"

echo "=== visualize_task (requires trajectory JSON under dextoolbench/trajectories/) ==="
python dextoolbench/visualize_task.py \
  --object_category "$DEMO_CATEGORY" \
  --object_name "$DEMO_OBJECT" \
  --task_name "$DEMO_TASK" \
  || { echo "Skip: missing data or trajectory"; exit 0; }

echo "=== eval.py single run (requires Isaac Gym, pretrained policy, full install) ==="
# Example only — user supplies checkpoint and ports:
# python dextoolbench/eval.py --help
