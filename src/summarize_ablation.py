import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PPO ablation variants.")
    parser.add_argument("--ablation_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=200.0)
    return parser.parse_args()


def discover_variant_dirs(ablation_root: Path) -> List[Path]:
    variant_dirs = []
    for d in sorted(ablation_root.iterdir()):
        if d.is_dir() and (d / "raw_logs").exists():
            variant_dirs.append(d)
    return variant_dirs


def load_variant_df(variant_dir: Path) -> pd.DataFrame:
    files = sorted((variant_dir / "raw_logs").glob("ppo_*_seed*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def final_stats(df: pd.DataFrame, variant: str) -> Dict:
    final_rows = (
        df.sort_values(["seed", "train_step"])
        .groupby("seed", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    values = final_rows["eval_mean_return"].values
    return {
        "variant": variant,
        "n_seeds": int(len(values)),
        "final_mean_return": float(np.mean(values)),
        "final_std_return": float(np.std(values)),
    }


def threshold_stats(df: pd.DataFrame, variant: str, threshold: float) -> Dict:
    steps_to_threshold = []
    seeds = sorted(df["seed"].unique().tolist())
    for seed in seeds:
        seed_df = df[df["seed"] == seed].sort_values("train_step")
        hit = seed_df[seed_df["eval_mean_return"] >= threshold]
        if len(hit) == 0:
            steps_to_threshold.append(np.nan)
        else:
            steps_to_threshold.append(float(hit.iloc[0]["train_step"]))

    valid = np.array([x for x in steps_to_threshold if not np.isnan(x)])
    return {
        "variant": variant,
        "threshold": threshold,
        "success_rate": float(len(valid) / max(len(steps_to_threshold), 1)),
        "mean_steps_to_threshold": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "std_steps_to_threshold": float(np.std(valid)) if len(valid) > 0 else np.nan,
    }


def main() -> None:
    args = parse_args()
    ablation_root = Path(args.ablation_root)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    variant_dirs = discover_variant_dirs(ablation_root)
    if not variant_dirs:
        raise RuntimeError(f"No variants found in {ablation_root}")

    final_rows = []
    efficiency_rows = []

    for variant_dir in variant_dirs:
        variant = variant_dir.name
        df = load_variant_df(variant_dir)
        if df.empty:
            print(f"Skip {variant}: no logs")
            continue
        final_rows.append(final_stats(df, variant))
        efficiency_rows.append(threshold_stats(df, variant, args.threshold))

    if not final_rows:
        raise RuntimeError("No valid variant logs available for summary.")

    final_df = pd.DataFrame(final_rows).sort_values("final_mean_return", ascending=False)
    efficiency_df = pd.DataFrame(efficiency_rows).sort_values("success_rate", ascending=False)

    final_path = out_dir / "ablation_final_performance.csv"
    efficiency_path = out_dir / "ablation_sample_efficiency.csv"
    final_df.to_csv(final_path, index=False)
    efficiency_df.to_csv(efficiency_path, index=False)

    print(f"Saved table: {final_path}")
    print(f"Saved table: {efficiency_path}")


if __name__ == "__main__":
    main()
