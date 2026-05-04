#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"
ENV_ID="${2:-LunarLander-v2}"
EPISODES="${3:-20}"
SEED="${4:-0}"

for algo in ppo a2c reinforce; do
  python "${PROJECT_ROOT}/src/evaluate.py" \
    --project_root "${PROJECT_ROOT}" \
    --algo "${algo}" \
    --env "${ENV_ID}" \
    --seed "${SEED}" \
    --episodes "${EPISODES}"
done
