import gymnasium as gym

from config import get_device


def train_model4(config: dict):
    device = get_device()

    env = gym.make("CartPole-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    num_episodes = config['episode_number']
    time_steps = config['time_steps']

    for _ in range(time_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
