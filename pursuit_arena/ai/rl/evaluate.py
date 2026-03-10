from __future__ import annotations

"""
Evaluate a trained PPO model on the ChaseEscapeEnv.
"""

from pathlib import Path

from stable_baselines3 import PPO

from .chase_escape_env import ChaseEscapeEnv


def main() -> None:
    model_path = Path("runs/ppo_chase_escape/ppo_chase_escape_final.zip")
    if not model_path.exists():
        raise SystemExit(f"Model file not found at {model_path}. Train first with train_ppo.py.")

    env = ChaseEscapeEnv(render_mode=None)
    model = PPO.load(str(model_path), env=env)

    episodes = 10
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
        print(f"Episode {ep + 1}: reward={ep_reward:.2f}, info={info}")

    env.close()


if __name__ == "__main__":
    main()

