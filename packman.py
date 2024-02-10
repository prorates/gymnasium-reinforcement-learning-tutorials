import time
import gymnasium as gym

from config import get_device


def train_model3(config: dict):
    device = get_device()

    env = gym.make("MsPacman-v0", render_mode="human")
    observation, info = env.reset(seed=42)

    # simulate the environment
    num_episodes = config['episode_number']
    time_steps = config['time_steps']

    for i in range(num_episodes):
        state = env.reset()
        totalReward = 0

        for _ in range(time_steps):
            env.render()

            # take a random action
            randomAction = env.action_space.sample()
            observation, reward, done, info = env.step(randomAction)

            time.sleep(0.1)
            totalReward += reward

        print('Episode', i, ', Total reward:', totalReward)

    env.close()
