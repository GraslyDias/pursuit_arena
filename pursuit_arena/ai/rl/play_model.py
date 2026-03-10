from __future__ import annotations

"""
Load a trained model and watch it act in the environment with Pygame rendering.
"""

from pathlib import Path

from stable_baselines3 import PPO

from .chase_escape_env import ChaseEscapeEnv


def main() -> None:
    model_path = Path("runs/ppo_chase_escape/ppo_chase_escape_final.zip")
    if not model_path.exists():
        raise SystemExit(f"Model file not found at {model_path}. Train first with train_ppo.py.")

    env = ChaseEscapeEnv(render_mode="human")
    model = PPO.load(str(model_path), env=env)

    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
    env.close()


if __name__ == "__main__":
    main()

