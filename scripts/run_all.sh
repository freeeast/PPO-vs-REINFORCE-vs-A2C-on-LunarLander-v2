#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"
ENV_ID="${2:-LunarLander-v2}"
TOTAL_STEPS="${3:-300000}"

SEEDS=(0 1 2 3 4)

echo "Project root: ${PROJECT_ROOT}"
echo "Environment: ${ENV_ID}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  python "${PROJECT_ROOT}/src/train_ppo.py" \
    --project_root "${PROJECT_ROOT}" \
    --env "${ENV_ID}" \
    --seed "${seed}" \
    --total_steps "${TOTAL_STEPS}"
done

for seed in "${SEEDS[@]}"; do
  python "${PROJECT_ROOT}/src/train_a2c.py" \
    --project_root "${PROJECT_ROOT}" \
    --env "${ENV_ID}" \
    --seed "${seed}" \
    --total_steps "${TOTAL_STEPS}"
done

for seed in "${SEEDS[@]}"; do
  python "${PROJECT_ROOT}/src/train_reinforce.py" \
    --project_root "${PROJECT_ROOT}" \
    --env "${ENV_ID}" \
    --seed "${seed}" \
    --total_steps "${TOTAL_STEPS}"
done

python "${PROJECT_ROOT}/src/plot_curves.py" \
  --log_dir "${PROJECT_ROOT}/results/raw_logs" \
  --out_dir "${PROJECT_ROOT}/results/figures" \
  --env "${ENV_ID}"

python "${PROJECT_ROOT}/src/summarize_results.py" \
  --log_dir "${PROJECT_ROOT}/results/raw_logs" \
  --out_dir "${PROJECT_ROOT}/results/tables" \
  --threshold 200

echo "All done."
