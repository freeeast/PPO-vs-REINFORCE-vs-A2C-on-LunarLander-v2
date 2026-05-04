# PPO vs REINFORCE vs A2C on LunarLander-v2

This repository provides a complete, reproducible engineering setup for the course project:

- Environment: `LunarLander-v2`
- Algorithms: `PPO`, `A2C`, `REINFORCE`
- Outputs: training logs, models, learning curves, summary tables

## 1) Project Structure

```text
.
├── configs/
├── scripts/
├── src/
├── results/
├── report/
├── requirements.txt
└── README.md
```

Generated artifacts are saved under:

- `results/raw_logs/` - per-run CSV logs
- `results/models/` - trained checkpoints
- `results/figures/` - figures
- `results/tables/` - summary CSV tables

## 2) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If Box2D installation fails, first update pip/setuptools and retry:

```bash
pip install -U pip setuptools wheel
pip install gymnasium[box2d]
```

## 3) Quick Start (Single Run)

Train one seed for each algorithm:

```bash
python src/train_ppo.py --seed 0 --total_steps 200000
python src/train_a2c.py --seed 0 --total_steps 200000
python src/train_reinforce.py --seed 0 --total_steps 200000
```

Evaluate models:

```bash
python src/evaluate.py --algo ppo --seed 0 --episodes 20
python src/evaluate.py --algo a2c --seed 0 --episodes 20
python src/evaluate.py --algo reinforce --seed 0 --episodes 20
```

Plot and summarize:

```bash
python src/plot_curves.py --log_dir results/raw_logs --out_dir results/figures
python src/summarize_results.py --log_dir results/raw_logs --out_dir results/tables --threshold 200
```

## 4) Full Experiment (5 Seeds)

Run all experiments and auto-generate figures/tables:

```bash
bash scripts/run_all.sh .
```

Arguments:

```bash
bash scripts/run_all.sh <project_root> <env_id> <total_steps>
```

Example:

```bash
bash scripts/run_all.sh . LunarLander-v2 300000
```

## 5) Logging Format

Each run writes one CSV in `results/raw_logs/`, with fields:

- `algo`
- `seed`
- `train_step`
- `eval_mean_return`
- `eval_std_return`
- `wall_time_sec`

## 6) Core Engineering Decisions

- **Fair comparison**: same env, same training budget, same seed list, same eval protocol.
- **Periodic evaluation**: all algorithms are evaluated every `eval_freq` steps.
- **Reproducibility**: fixed seed control + deterministic evaluation policy.
- **Statistics-first output**: mean/std curves and aggregated tables by default.

## 7) Recommended Course Settings

- `total_steps`: 300000
- `seeds`: `[0, 1, 2, 3, 4]`
- `eval_freq`: 10000
- `n_eval_episodes`: 10
- threshold for sample efficiency: `200`

If time is tight, run:

- `total_steps`: 200000
- `seeds`: `[0, 1, 2]`

## 8) Report Integration

Use these outputs directly in report:

- `results/figures/learning_curves.png`
- `results/tables/final_performance.csv`
- `results/tables/sample_efficiency.csv`

Suggested report sections:

1. Problem + MDP formulation
2. Methodology (PPO/A2C/REINFORCE)
3. Experimental setup
4. Results
5. Findings and discussion

## 9) Troubleshooting

- **Model file missing**: verify training finished and `results/models` contains corresponding file.
- **No plot generated**: check `results/raw_logs` contains files like `ppo_seed0.csv`.
- **REINFORCE unstable**: lower learning rate (e.g., `5e-4`) or increase total steps.

## 10) Ablation Experiment (PPO)

This project includes an ablation pipeline for PPO on:

- `clip_range`: `0.1 / 0.2 / 0.3`
- `learning_rate`: `1e-4 / 3e-4 / 1e-3`

### 10.1 Run clip-range ablation

```bash
bash scripts/run_ablation_ppo.sh . LunarLander-v2 200000 clip
```

### 10.2 Run learning-rate ablation

```bash
bash scripts/run_ablation_ppo.sh . LunarLander-v2 200000 lr
```

### 10.3 Ablation outputs

For clip ablation:

- `results/ablation/ppo_clip/<variant>/raw_logs/*.csv`
- `results/ablation/ppo_clip/<variant>/models/*.zip`
- `results/ablation/ppo_clip/figures/ablation_learning_curves.png`
- `results/ablation/ppo_clip/figures/ablation_curve_summary.csv`
- `results/ablation/ppo_clip/tables/ablation_final_performance.csv`
- `results/ablation/ppo_clip/tables/ablation_sample_efficiency.csv`

For learning-rate ablation, replace `ppo_clip` with `ppo_lr`.

### 10.4 Optional: summarize or plot only

```bash
python src/plot_ablation.py \
  --ablation_root results/ablation/ppo_clip \
  --out_dir results/ablation/ppo_clip/figures \
  --env LunarLander-v2 \
  --title "PPO clip ablation"

python src/summarize_ablation.py \
  --ablation_root results/ablation/ppo_clip \
  --out_dir results/ablation/ppo_clip/tables \
  --threshold 200
```

## 11) Statistical summaries (bootstrap)

After logs exist, run bootstrap confidence intervals on **per-seed final evaluation returns** (last row of each `*_seed*.csv`).

### 11.1 Main experiment (PPO / A2C / REINFORCE)

```bash
python src/stats_tests.py main \
  --log_dir results/raw_logs \
  --algos ppo a2c reinforce \
  --out_dir results/tables \
  --n_bootstrap 10000 \
  --ci 0.95
```

Outputs:

- `results/tables/stats_main_summary.csv` — mean and 95% bootstrap CI for each algorithm
- `results/tables/stats_main_pairwise.csv` — pairwise mean differences (A − B) with CI and a simple overlap interpretation
- `results/tables/stats_main_meta.json`

### 11.2 Ablation (after `run_ablation_ppo.sh` finishes)

```bash
python src/stats_tests.py ablation \
  --ablation_root results/ablation/ppo_clip \
  --n_bootstrap 10000
```

Default output directory: `results/ablation/ppo_clip/tables_stats/`

- `stats_ablation_summary.csv`
- `stats_ablation_pairwise.csv`
- `stats_ablation_meta.json`

Repeat with `results/ablation/ppo_lr` for the learning-rate ablation.
