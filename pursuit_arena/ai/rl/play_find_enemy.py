from pathlib import Path

from stable_baselines3 import PPO
from pursuit_arena.ai.rl.chase_escape_env import ChaseEscapeEnv, load_training_map


def main() -> None:
    # Use the same map you used for training
    map_path = Path("training_map.json")

    env = ChaseEscapeEnv(render_mode="human")
    env.static_enemy = True  # enemy does not move

    if map_path.exists():
        env.training_map_options = load_training_map(map_path)
        print("Using training_map.json for play")

    model_path = Path("runs/ppo_find_enemy/ppo_find_enemy_final.zip")
    model = PPO.load(str(model_path), env=env)

    episodes = 5
    for ep in range(episodes):
        obs, info = env.reset()
        done, truncated = False, False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

    env.close()


if __name__ == "__main__":
    main()