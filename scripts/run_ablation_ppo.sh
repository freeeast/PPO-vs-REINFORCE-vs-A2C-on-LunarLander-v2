#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-.}"
ENV_ID="${2:-LunarLander-v2}"
TOTAL_STEPS="${3:-200000}"
ABLATION_TYPE="${4:-clip}" # clip or lr
SEEDS=(0 1 2 3 4)

if [[ "${ABLATION_TYPE}" != "clip" && "${ABLATION_TYPE}" != "lr" ]]; then
  echo "ABLATION_TYPE must be one of: clip, lr"
  exit 1
fi

if [[ "${ABLATION_TYPE}" == "clip" ]]; then
  VALUES=("0.1" "0.2" "0.3")
else
  VALUES=("1e-4" "3e-4" "1e-3")
fi

ABLATION_ROOT="${PROJECT_ROOT}/results/ablation/ppo_${ABLATION_TYPE}"

echo "=== PPO Ablation Run ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Environment: ${ENV_ID}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Ablation type: ${ABLATION_TYPE}"
echo "Values: ${VALUES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Output root: ${ABLATION_ROOT}"

for value in "${VALUES[@]}"; do
  SAFE_VALUE="${value//./p}"
  SAFE_VALUE="${SAFE_VALUE//-/m}"
  VARIANT_TAG="${ABLATION_TYPE}_${SAFE_VALUE}"
  RESULTS_ROOT="results/ablation/ppo_${ABLATION_TYPE}/${VARIANT_TAG}"

  echo "--- Running variant ${VARIANT_TAG} (value=${value}) ---"
  for seed in "${SEEDS[@]}"; do
    CMD=(
      python "${PROJECT_ROOT}/src/train_ppo.py"
      --project_root "${PROJECT_ROOT}"
      --results_root "${RESULTS_ROOT}"
      --run_tag "${VARIANT_TAG}"
      --env "${ENV_ID}"
      --seed "${seed}"
      --total_steps "${TOTAL_STEPS}"
    )

    if [[ "${ABLATION_TYPE}" == "clip" ]]; then
      CMD+=(--clip_range "${value}")
    else
      CMD+=(--learning_rate "${value}")
    fi

    "${CMD[@]}"
  done
done

python "${PROJECT_ROOT}/src/plot_ablation.py" \
  --ablation_root "${ABLATION_ROOT}" \
  --out_dir "${ABLATION_ROOT}/figures" \
  --env "${ENV_ID}" \
  --title "PPO ${ABLATION_TYPE} ablation on ${ENV_ID}"

python "${PROJECT_ROOT}/src/summarize_ablation.py" \
  --ablation_root "${ABLATION_ROOT}" \
  --out_dir "${ABLATION_ROOT}/tables" \
  --threshold 200

echo "Ablation done."
