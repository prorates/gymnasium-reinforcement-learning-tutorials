# -*- coding: utf-8 -*-
"""
Demonstration of the Cart Pole OpenAI Gym environment

Author: Aleksandar Haber
Date: February 2023
"""
import gymnasium as gym
import time

# create environment
env = gym.make('CartPole-v1', render_mode='human')
# reset the environment,
# returns an initial state
(state, _) = env.reset()
# states are
# cart position, cart velocity
# pole angle, pole angular velocity

# render the environment
env.render()
# close the environment
# env.close()

# push cart in one direction
env.step(0)

# observation space limits
env.observation_space

# upper limit
env.observation_space.high

# lower limit
env.observation_space.low

# action space
env.action_space

# all the specs
env.spec

# maximum number of steps per episode
env.spec.max_episode_steps

# reward threshold per episode
env.spec.reward_threshold

# simulate the environment
episodeNumber = 2
timeSteps = 100

for episodeIndex in range(episodeNumber):
    initial_state = env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations = []
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.1)
        if (terminated):
            time.sleep(1)
            break
env.close()
