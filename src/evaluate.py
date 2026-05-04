import argparse
from pathlib import Path

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch

from utils import ReinforcePolicyNet, evaluate_reinforce_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model.")
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "a2c", "reinforce"])
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def evaluate_sb3(algo: str, model_path: Path, env_id: str, seed: int, episodes: int) -> None:
    eval_env = make_vec_env(env_id, n_envs=1, seed=seed + 777)
    if algo == "ppo":
        model = PPO.load(str(model_path), env=eval_env)
    else:
        model = A2C.load(str(model_path), env=eval_env)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        warn=False,
    )
    eval_env.close()
    print(f"[{algo}] episodes={episodes} mean_return={mean_reward:.2f} std={std_reward:.2f}")


def evaluate_reinforce(model_path: Path, env_id: str, seed: int, episodes: int, device: str) -> None:
    ckpt = torch.load(str(model_path), map_location=device)
    policy = ReinforcePolicyNet(
        obs_dim=ckpt["obs_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dim=ckpt.get("hidden_dim", 128),
    ).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    mean_reward, std_reward = evaluate_reinforce_policy(
        policy=policy,
        env_id=env_id,
        seed=seed + 777,
        n_eval_episodes=episodes,
        deterministic=True,
        device=device,
    )
    print(f"[reinforce] episodes={episodes} mean_return={mean_reward:.2f} std={std_reward:.2f}")


def main() -> None:
    args = parse_args()
    models_dir = Path(args.project_root) / "results" / "models"

    if args.algo in {"ppo", "a2c"}:
        model_path = models_dir / f"{args.algo}_seed{args.seed}.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        evaluate_sb3(args.algo, model_path, args.env, args.seed, args.episodes)
    else:
        model_path = models_dir / f"reinforce_seed{args.seed}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        evaluate_reinforce(model_path, args.env, args.seed, args.episodes, args.device)


if __name__ == "__main__":
    main()
