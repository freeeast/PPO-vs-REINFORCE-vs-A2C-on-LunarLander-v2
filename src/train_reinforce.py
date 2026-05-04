import argparse
import time
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from utils import (
    ReinforcePolicyNet,
    append_csv_row,
    default_result_paths,
    discounted_returns,
    ensure_dir,
    evaluate_reinforce_policy,
    init_csv,
    set_global_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train REINFORCE on LunarLander-v2.")
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project_root", type=str, default=".")
    return parser.parse_args()


def maybe_log_eval(
    policy: ReinforcePolicyNet,
    args: argparse.Namespace,
    csv_path: str,
    current_step: int,
    start_time: float,
) -> None:
    mean_ret, std_ret = evaluate_reinforce_policy(
        policy=policy,
        env_id=args.env,
        seed=args.seed + 10_000,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        device=args.device,
    )
    row = {
        "algo": "reinforce",
        "seed": args.seed,
        "train_step": int(current_step),
        "eval_mean_return": float(mean_ret),
        "eval_std_return": float(std_ret),
        "wall_time_sec": float(time.time() - start_time),
    }
    append_csv_row(csv_path, row)
    print(
        f"[reinforce] seed={args.seed} step={current_step} "
        f"mean={mean_ret:.2f} std={std_ret:.2f}"
    )


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    ensure_dir(args.project_root)

    paths = default_result_paths(args.project_root)
    csv_path = str(Path(paths["raw_logs"]) / f"reinforce_seed{args.seed}.csv")
    model_path = str(Path(paths["models"]) / f"reinforce_seed{args.seed}.pt")

    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device(args.device)
    policy = ReinforcePolicyNet(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

    fieldnames = [
        "algo",
        "seed",
        "train_step",
        "eval_mean_return",
        "eval_std_return",
        "wall_time_sec",
    ]
    init_csv(csv_path, fieldnames)

    total_steps = 0
    next_eval_step = 0
    start_time = time.time()
    maybe_log_eval(policy, args, csv_path, current_step=0, start_time=start_time)
    next_eval_step += args.eval_freq

    while total_steps < args.total_steps:
        obs, _ = env.reset(seed=args.seed + total_steps)
        done = False
        trunc = False

        log_probs = []
        rewards = []

        while not (done or trunc):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, done, trunc, _ = env.step(int(action.item()))
            log_probs.append(dist.log_prob(action))
            rewards.append(float(reward))

            obs = next_obs
            total_steps += 1

            if total_steps >= next_eval_step and total_steps <= args.total_steps:
                maybe_log_eval(
                    policy,
                    args,
                    csv_path,
                    current_step=total_steps,
                    start_time=start_time,
                )
                next_eval_step += args.eval_freq

            if total_steps >= args.total_steps:
                break

        returns = discounted_returns(rewards, gamma=args.gamma).to(device)
        loss = torch.tensor(0.0, device=device)
        for log_prob, ret in zip(log_probs, returns):
            loss = loss - log_prob * ret

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    if (total_steps % args.eval_freq) != 0:
        maybe_log_eval(policy, args, csv_path, current_step=total_steps, start_time=start_time)

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
        },
        model_path,
    )
    env.close()
    print(f"Saved model: {model_path}")
    print(f"Saved logs: {csv_path}")


if __name__ == "__main__":
    main()
