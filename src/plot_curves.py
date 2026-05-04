import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learning curves across seeds.")
    parser.add_argument("--log_dir", type=str, default="results/raw_logs")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--algos", type=str, nargs="+", default=["ppo", "a2c", "reinforce"])
    return parser.parse_args()


def load_logs(log_dir: Path, algo: str) -> pd.DataFrame:
    files = sorted(log_dir.glob(f"{algo}_seed*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 6))
    has_data = False

    for algo in args.algos:
        df = load_logs(log_dir, algo)
        if df.empty:
            print(f"Skip {algo}: no logs in {log_dir}")
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

        plt.plot(x, y, label=algo.upper(), linewidth=2)
        plt.fill_between(x, y - std, y + std, alpha=0.2)

        summary_out = out_dir / f"{algo}_curve_summary.csv"
        summary.to_csv(summary_out, index=False)

    if not has_data:
        raise RuntimeError(f"No valid log files found in {log_dir}")

    plt.title(f"Learning Curves on {args.env}")
    plt.xlabel("Training Steps")
    plt.ylabel("Evaluation Mean Return")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "learning_curves.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")


if __name__ == "__main__":
    main()
