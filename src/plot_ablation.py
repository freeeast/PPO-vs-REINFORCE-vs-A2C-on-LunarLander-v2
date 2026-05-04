import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO ablation learning curves.")
    parser.add_argument("--ablation_root", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--title", type=str, default="PPO ablation")
    return parser.parse_args()


def discover_variants(ablation_root: Path) -> List[Tuple[str, Path]]:
    variants = []
    for variant_dir in sorted(ablation_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        raw_logs_dir = variant_dir / "raw_logs"
        if raw_logs_dir.exists():
            variants.append((variant_dir.name, raw_logs_dir))
    return variants


def load_variant_logs(raw_logs_dir: Path) -> pd.DataFrame:
    files = sorted(raw_logs_dir.glob("ppo_*_seed*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main() -> None:
    args = parse_args()
    ablation_root = Path(args.ablation_root)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    sns.set_theme(style="whitegrid")

    variants = discover_variants(ablation_root)
    if not variants:
        raise RuntimeError(f"No ablation variants found in {ablation_root}")

    plt.figure(figsize=(9, 6))
    has_data = False
    summary_rows = []

    for variant_name, raw_logs_dir in variants:
        df = load_variant_logs(raw_logs_dir)
        if df.empty:
            print(f"Skip {variant_name}: no CSV logs in {raw_logs_dir}")
            continue

        has_data = True
        summary = (
            df.groupby("train_step", as_index=False)["eval_mean_return"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "mean_return", "std": "std_return"})
        )

        x = summary["train_step"]
        y = summary["mean_return"]
        std = summary["std_return"].fillna(0.0)
        plt.plot(x, y, label=variant_name, linewidth=2)
        plt.fill_between(x, y - std, y + std, alpha=0.2)

        for _, row in summary.iterrows():
            summary_rows.append(
                {
                    "variant": variant_name,
                    "train_step": int(row["train_step"]),
                    "mean_return": float(row["mean_return"]),
                    "std_return": float(row["std_return"]) if pd.notna(row["std_return"]) else 0.0,
                }
            )

    if not has_data:
        raise RuntimeError(f"No valid ablation logs found under {ablation_root}")

    plt.title(args.title)
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Mean Return")
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / "ablation_learning_curves.png"
    plt.savefig(fig_path, dpi=200)
    print(f"Saved figure: {fig_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "ablation_curve_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
