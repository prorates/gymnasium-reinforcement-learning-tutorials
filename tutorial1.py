# -*- coding: utf-8 -*-
"""
Demonstration of the Cart Pole OpenAI Gym environment

Author: Aleksandar Haber
Date: February 2023
"""
import gymnasium as gym
import time

from config import get_device

def train_model1(config: dict):
    device = get_device()

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
    print(f"{'Observation Space : ':>25}{env.observation_space}")
    
    # upper limit
    print(f"{'Upper limit : ':>25}{env.observation_space.high}")
    
    # lower limit
    print(f"{'Lower limit : ':>25}{env.observation_space.low}")
    
    # action space
    print(f"{'Action Space : ':>25}{env.action_space}")
    
    # all the specs
    print(f"{'Spec : ':>25}{env.spec}")
    
    # maximum number of steps per episode
    print(f"{'Max Steps : ':>25}{env.spec.max_episode_steps}")
    
    # reward threshold per episode
    print(f"{'Reward Threshold : ':>25}{env.spec.reward_threshold}")
    
    # simulate the environment
    episodeNumber = config['episode_number']
    timeSteps = config['time_steps']
    
    for episodeIndex in range(episodeNumber):
        initial_state = env.reset()
        print(episodeIndex)
        env.render()
        appendedObservations = []
        for timeIndex in range(timeSteps):
            # print(timeIndex)
            random_action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(random_action)
            appendedObservations.append(observation)
            time.sleep(0.1)
            if (terminated):
                time.sleep(1)
                break
    env.close()
    