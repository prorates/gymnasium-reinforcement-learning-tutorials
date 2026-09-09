"""Demonstration of the Cart Pole environment: what the spaces and the step tuple look like.

Originally tutorial1.py, after Aleksandar Haber (February 2023).
"""

from __future__ import annotations

import time

import gymnasium as gym

from rl_tutorials.config import RunSettings

ENV_ID = "CartPole-v1"
DESCRIPTION = "Random actions on CartPole, printing the observation and action spaces"
SETTINGS_TYPE = RunSettings


def run(settings: RunSettings) -> None:
    env = gym.make(ENV_ID, render_mode=settings.render_mode)

    # Reset before touching env.spec or stepping: an env that has never been reset is not
    # required to have valid state. The original read spec after a bare step().
    env.reset(seed=settings.seed)

    # The observation is (cart position, cart velocity, pole angle, pole angular velocity).
    # CartPole's observation space is a Box, which is what carries the per-element bounds.
    box = env.observation_space
    assert isinstance(box, gym.spaces.Box)
    print(f"{'Observation Space : ':>25}{box}")
    print(f"{'Upper limit : ':>25}{box.high}")
    print(f"{'Lower limit : ':>25}{box.low}")
    print(f"{'Action Space : ':>25}{env.action_space}")
    print(f"{'Spec : ':>25}{env.spec}")
    if env.spec is not None:
        print(f"{'Max Steps : ':>25}{env.spec.max_episode_steps}")
        print(f"{'Reward Threshold : ':>25}{env.spec.reward_threshold}")

    for episode_index in range(settings.episodes):
        env.reset()
        print(f"Episode {episode_index + 1}/{settings.episodes}")
        for _ in range(settings.steps):
            _observation, _reward, terminated, truncated, _info = env.step(
                env.action_space.sample()
            )
            if settings.render:
                time.sleep(0.1)
            if terminated or truncated:
                break

    env.close()
