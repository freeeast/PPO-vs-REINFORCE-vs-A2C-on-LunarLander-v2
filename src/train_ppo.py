import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from utils import EvalCSVLoggerCallback, default_result_paths, set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v2.")
    parser.add_argument("--env", type=str, default="LunarLander-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--run_tag", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)

    paths = default_result_paths(args.project_root, results_root=args.results_root)
    prefix = f"ppo_{args.run_tag}" if args.run_tag else "ppo"
    csv_path = str(Path(paths["raw_logs"]) / f"{prefix}_seed{args.seed}.csv")
    model_path = str(Path(paths["models"]) / f"{prefix}_seed{args.seed}")

    train_env = make_vec_env(args.env, n_envs=1, seed=args.seed)
    eval_env = make_vec_env(args.env, n_envs=1, seed=args.seed + 1000)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    callback = EvalCSVLoggerCallback(
        eval_env=eval_env,
        algo=prefix,
        seed=args.seed,
        csv_path=csv_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )

    model.learn(total_timesteps=args.total_steps, callback=callback)
    model.save(model_path)

    train_env.close()
    eval_env.close()
    print(f"Saved model: {model_path}.zip")
    print(f"Saved logs: {csv_path}")


if __name__ == "__main__":
    main()
