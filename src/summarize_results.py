import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final performance and sample efficiency.")
    parser.add_argument("--log_dir", type=str, default="results/raw_logs")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--algos", nargs="+", default=["ppo", "a2c", "reinforce"])
    parser.add_argument("--threshold", type=float, default=200.0)
    return parser.parse_args()


def load_algo_logs(log_dir: Path, algo: str) -> pd.DataFrame:
    files = sorted(log_dir.glob(f"{algo}_seed*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def final_performance_table(df: pd.DataFrame, algo: str) -> Dict:
    final_rows = (
        df.sort_values(["seed", "train_step"])
        .groupby("seed", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    values = final_rows["eval_mean_return"].values
    return {
        "algo": algo,
        "n_seeds": int(len(values)),
        "final_mean_return": float(np.mean(values)),
        "final_std_return": float(np.std(values)),
    }


def sample_efficiency_table(df: pd.DataFrame, algo: str, threshold: float) -> Dict:
    steps_to_threshold: List[float] = []
    seeds = sorted(df["seed"].unique().tolist())
    for seed in seeds:
        seed_df = df[df["seed"] == seed].sort_values("train_step")
        hit = seed_df[seed_df["eval_mean_return"] >= threshold]
        if len(hit) == 0:
            steps_to_threshold.append(np.nan)
        else:
            steps_to_threshold.append(float(hit.iloc[0]["train_step"]))

    valid = np.array([x for x in steps_to_threshold if not np.isnan(x)])
    success_rate = float(len(valid) / max(len(steps_to_threshold), 1))
    mean_steps = float(np.mean(valid)) if len(valid) > 0 else np.nan
    std_steps = float(np.std(valid)) if len(valid) > 0 else np.nan

    return {
        "algo": algo,
        "threshold": threshold,
        "success_rate": success_rate,
        "mean_steps_to_threshold": mean_steps,
        "std_steps_to_threshold": std_steps,
    }


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    final_rows = []
    efficiency_rows = []

    for algo in args.algos:
        df = load_algo_logs(log_dir, algo)
        if df.empty:
            print(f"Skip {algo}: no logs in {log_dir}")
            continue

        final_rows.append(final_performance_table(df, algo))
        efficiency_rows.append(sample_efficiency_table(df, algo, args.threshold))

    if not final_rows:
        raise RuntimeError(f"No valid log files found in {log_dir}")

    final_df = pd.DataFrame(final_rows)
    eff_df = pd.DataFrame(efficiency_rows)

    final_path = out_dir / "final_performance.csv"
    eff_path = out_dir / "sample_efficiency.csv"
    final_df.to_csv(final_path, index=False)
    eff_df.to_csv(eff_path, index=False)

    print(f"Saved table: {final_path}")
    print(f"Saved table: {eff_path}")


if __name__ == "__main__":
    main()
