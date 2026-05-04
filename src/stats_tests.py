"""
Bootstrap confidence intervals and pairwise comparisons for RL experiment logs.

No SciPy dependency: uses numpy-only bootstrap resampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import ensure_dir


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Return (point_mean, ci_low, ci_high) for the mean."""
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(x))
    if x.size == 1:
        return point, point, point
    boots = []
    n = x.size
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(x[idx])))
    boots_arr = np.array(boots)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots_arr, alpha))
    hi = float(np.quantile(boots_arr, 1.0 - alpha))
    return point, lo, hi


def bootstrap_mean_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:
    """Bootstrap distribution of (mean(A) - mean(B)). Returns (point_diff, ci_low, ci_high)."""
    if rng is None:
        rng = np.random.default_rng(1)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(a) - np.mean(b))
    if a.size == 1 and b.size == 1:
        return point, point, point
    diffs = []
    for _ in range(n_bootstrap):
        sa = a[rng.integers(0, a.size, size=a.size)]
        sb = b[rng.integers(0, b.size, size=b.size)]
        diffs.append(float(np.mean(sa) - np.mean(sb)))
    diffs_arr = np.array(diffs)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(diffs_arr, alpha))
    hi = float(np.quantile(diffs_arr, 1.0 - alpha))
    return point, lo, hi


def final_returns_per_seed_from_logs(log_dir: Path, algo: str) -> np.ndarray:
    """algo matches filename prefix, e.g. 'ppo' -> ppo_seed*.csv; 'ppo_clip_0p2' -> ppo_clip_0p2_seed*.csv."""
    files = sorted(log_dir.glob(f"{algo}_seed*.csv"))
    if not files:
        return np.array([])
    finals = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty or "eval_mean_return" not in df.columns:
            continue
        last = df.sort_values("train_step").iloc[-1]
        finals.append(float(last["eval_mean_return"]))
    return np.array(finals, dtype=float)


def discover_ablation_variants(ablation_root: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for d in sorted(ablation_root.iterdir()):
        if d.is_dir() and (d / "raw_logs").exists():
            out.append((d.name, d / "raw_logs"))
    return out


def prefix_for_variant(variant_name: str) -> str:
    """CSV files are named ppo_{variant}_seed{k}.csv."""
    return f"ppo_{variant_name}"


def run_main(
    log_dir: Path,
    algos: List[str],
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    per_algo: Dict[str, np.ndarray] = {}
    for algo in algos:
        per_algo[algo] = final_returns_per_seed_from_logs(log_dir, algo)

    summary_rows = []
    for algo, vals in per_algo.items():
        if vals.size == 0:
            summary_rows.append(
                {
                    "algo": algo,
                    "n_seeds": 0,
                    "mean_return": np.nan,
                    "std_return": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue
        m, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        summary_rows.append(
            {
                "algo": algo,
                "n_seeds": int(vals.size),
                "mean_return": m,
                "std_return": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    pair_rows = []
    for i, a in enumerate(algos):
        for b in algos[i + 1 :]:
            va, vb = per_algo.get(a, np.array([])), per_algo.get(b, np.array([]))
            if va.size == 0 or vb.size == 0:
                continue
            d, lo, hi = bootstrap_mean_diff_ci(va, vb, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            pair_rows.append(
                {
                    "algo_a": a,
                    "algo_b": b,
                    "mean_diff_a_minus_b": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "interpretation": "A>B (95%)" if lo > 0 else ("B>A (95%)" if hi < 0 else "no clear separation (95%)"),
                }
            )

    return {"summary": summary_rows, "pairwise": pair_rows}


def run_ablation(
    ablation_root: Path,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    variants = discover_ablation_variants(ablation_root)
    per_variant: Dict[str, np.ndarray] = {}
    for name, raw_logs in variants:
        prefix = prefix_for_variant(name)
        per_variant[name] = final_returns_per_seed_from_logs(raw_logs, prefix)

    summary_rows = []
    for name, vals in per_variant.items():
        if vals.size == 0:
            summary_rows.append(
                {
                    "variant": name,
                    "n_seeds": 0,
                    "mean_return": np.nan,
                    "std_return": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                }
            )
            continue
        m, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
        summary_rows.append(
            {
                "variant": name,
                "n_seeds": int(vals.size),
                "mean_return": m,
                "std_return": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    names = list(per_variant.keys())
    pair_rows = []
    for i, va_name in enumerate(names):
        for vb_name in names[i + 1 :]:
            va, vb = per_variant[va_name], per_variant[vb_name]
            if va.size == 0 or vb.size == 0:
                continue
            d, lo, hi = bootstrap_mean_diff_ci(va, vb, n_bootstrap=n_bootstrap, ci=ci, rng=rng)
            pair_rows.append(
                {
                    "variant_a": va_name,
                    "variant_b": vb_name,
                    "mean_diff_a_minus_b": d,
                    "ci_low": lo,
                    "ci_high": hi,
                    "interpretation": "A>B (95%)" if lo > 0 else ("B>A (95%)" if hi < 0 else "no clear separation (95%)"),
                }
            )

    return {"summary": summary_rows, "pairwise": pair_rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap stats for main or ablation experiments.")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("main", help="Main experiment: compare algos in one log_dir.")
    m.add_argument("--log_dir", type=str, default="results/raw_logs")
    m.add_argument("--algos", nargs="+", default=["ppo", "a2c", "reinforce"])
    m.add_argument("--out_dir", type=str, default="results/tables")
    m.add_argument("--n_bootstrap", type=int, default=10_000)
    m.add_argument("--ci", type=float, default=0.95)
    m.add_argument("--seed", type=int, default=42)

    a = sub.add_parser("ablation", help="Ablation: compare variants under ablation_root.")
    a.add_argument("--ablation_root", type=str, required=True)
    a.add_argument("--out_dir", type=str, default=None, help="Default: <ablation_root>/tables_stats")
    a.add_argument("--n_bootstrap", type=int, default=10_000)
    a.add_argument("--ci", type=float, default=0.95)
    a.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "main":
        log_dir = Path(args.log_dir)
        out = run_main(log_dir, list(args.algos), args.n_bootstrap, args.ci, args.seed)
        out_dir = Path(args.out_dir)
        ensure_dir(str(out_dir))
        pd.DataFrame(out["summary"]).to_csv(out_dir / "stats_main_summary.csv", index=False)
        pd.DataFrame(out["pairwise"]).to_csv(out_dir / "stats_main_pairwise.csv", index=False)
        meta = {
            "mode": "main",
            "log_dir": str(log_dir),
            "algos": list(args.algos),
            "n_bootstrap": args.n_bootstrap,
            "ci": args.ci,
            "seed": args.seed,
        }
        (out_dir / "stats_main_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote {out_dir / 'stats_main_summary.csv'}")
        print(f"Wrote {out_dir / 'stats_main_pairwise.csv'}")
    else:
        ablation_root = Path(args.ablation_root)
        out_dir = Path(args.out_dir) if args.out_dir else ablation_root / "tables_stats"
        ensure_dir(str(out_dir))
        out = run_ablation(ablation_root, args.n_bootstrap, args.ci, args.seed)
        pd.DataFrame(out["summary"]).to_csv(out_dir / "stats_ablation_summary.csv", index=False)
        pd.DataFrame(out["pairwise"]).to_csv(out_dir / "stats_ablation_pairwise.csv", index=False)
        meta = {
            "mode": "ablation",
            "ablation_root": str(ablation_root),
            "n_bootstrap": args.n_bootstrap,
            "ci": args.ci,
            "seed": args.seed,
        }
        (out_dir / "stats_ablation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Wrote {out_dir / 'stats_ablation_summary.csv'}")
        print(f"Wrote {out_dir / 'stats_ablation_pairwise.csv'}")


if __name__ == "__main__":
    main()
