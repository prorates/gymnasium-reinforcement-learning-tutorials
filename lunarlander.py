import gymnasium as gym

from config import get_device


def train_model5(config: dict):
    device = get_device()

    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)

    num_episodes = config['episode_number']
    time_steps = config['time_steps']

    for _ in range(time_steps):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
