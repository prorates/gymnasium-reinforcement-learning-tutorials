import gymnasium as gym
from config import get_device


def train_model6(config: dict):
    device = get_device()

    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
    observation, info = env.reset(seed=42)

    num_episodes = config['episode_number']

    for _ in range(num_episodes):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
