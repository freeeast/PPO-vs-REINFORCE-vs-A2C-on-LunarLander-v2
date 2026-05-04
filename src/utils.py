import csv
import os
import random
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_csv(csv_path: str, fieldnames: Iterable[str]) -> None:
    ensure_dir(str(Path(csv_path).parent))
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_csv_row(csv_path: str, row: Dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


class EvalCSVLoggerCallback(BaseCallback):
    """Evaluate SB3 model periodically and write CSV logs."""

    def __init__(
        self,
        eval_env,
        algo: str,
        seed: int,
        csv_path: str,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.algo = algo
        self.seed = seed
        self.csv_path = csv_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.start_time = time.time()
        self.fieldnames = [
            "algo",
            "seed",
            "train_step",
            "eval_mean_return",
            "eval_std_return",
            "wall_time_sec",
        ]
        init_csv(self.csv_path, self.fieldnames)

    def _log_eval(self, step: int) -> None:
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        row = {
            "algo": self.algo,
            "seed": self.seed,
            "train_step": int(step),
            "eval_mean_return": float(mean_reward),
            "eval_std_return": float(std_reward),
            "wall_time_sec": float(time.time() - self.start_time),
        }
        append_csv_row(self.csv_path, row)

        if self.verbose > 0:
            print(
                f"[{self.algo}] seed={self.seed} step={step} "
                f"mean={mean_reward:.2f} std={std_reward:.2f}"
            )

    def _on_training_start(self) -> None:
        self._log_eval(step=0)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.num_timesteps % self.eval_freq == 0:
            self._log_eval(step=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        # Ensure the final step has a logged evaluation.
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            self._log_eval(step=self.num_timesteps)


class ReinforcePolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def discounted_returns(rewards: Iterable[float], gamma: float) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(list(rewards)):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()
    returns_t = torch.tensor(returns, dtype=torch.float32)
    if returns_t.numel() > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t


@torch.no_grad()
def evaluate_reinforce_policy(
    policy: ReinforcePolicyNet,
    env_id: str,
    seed: int,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    device: str = "cpu",
) -> Tuple[float, float]:
    env = gym.make(env_id)
    returns = []
    policy.eval()

    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        trunc = False
        ep_return = 0.0

        while not (done or trunc):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_t)
            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = int(dist.sample().item())

            obs, reward, done, trunc, _ = env.step(action)
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def default_result_paths(project_root: str, results_root: str = "results") -> Dict[str, str]:
    root = Path(project_root)
    results_base = root / results_root
    paths = {
        "raw_logs": str(results_base / "raw_logs"),
        "models": str(results_base / "models"),
        "figures": str(results_base / "figures"),
        "tables": str(results_base / "tables"),
    }
    for p in paths.values():
        ensure_dir(p)
    return paths
