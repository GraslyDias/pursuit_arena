from __future__ import annotations

"""
Train a PPO agent on the ChaseEscapeEnv.
"""

import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env

from .chase_escape_env import ChaseEscapeEnv, make_env


def main() -> None:
    log_dir = Path("runs/ppo_chase_escape")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Single environment wrapped for SB3
    def _env_fn():
        env = ChaseEscapeEnv(render_mode=None)
        env = Monitor(env, log_dir / "monitor.csv")
        return env

    env = DummyVecEnv([_env_fn])

    # Optional: run Gymnasium env checker once
    tmp_env = ChaseEscapeEnv(render_mode=None)
    check_env(tmp_env, warn=True)
    tmp_env.close()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir / "tb"),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_chase_escape",
    )

    timesteps = int(500_000)
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)

    model_path = log_dir / "ppo_chase_escape_final.zip"
    model.save(str(model_path))
    print(f"Saved final model to {model_path}")

    env.close()


if __name__ == "__main__":
    main()

