python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

# main experiment
bash scripts/run_all.sh . LunarLander-v2 300000

# PPO ablation (clip range)
bash scripts/run_ablation_ppo.sh . LunarLander-v2 200000 clip

# PPO ablation (learning rate)
bash scripts/run_ablation_ppo.sh . LunarLander-v2 200000 lr