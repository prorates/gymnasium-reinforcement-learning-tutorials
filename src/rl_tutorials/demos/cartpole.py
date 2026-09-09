"""The smallest possible reset/step loop, on CartPole."""

from __future__ import annotations

import gymnasium as gym

from rl_tutorials.config import RunSettings

ENV_ID = "CartPole-v1"
DESCRIPTION = "A minimal reset/step loop with random actions"
SETTINGS_TYPE = RunSettings


def run(settings: RunSettings) -> None:
    env = gym.make(ENV_ID, render_mode=settings.render_mode)
    env.reset(seed=settings.seed)

    for _ in range(settings.episodes):
        env.reset()
        for _ in range(settings.steps):
            _observation, _reward, terminated, truncated, _info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                env.reset()

    env.close()
