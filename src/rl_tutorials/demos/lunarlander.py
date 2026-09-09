"""LunarLander under random actions — a Box2D environment.

This demo could not start before: Box2D was never in the pinned requirements, so
`gym.make` raised DependencyNotInstalled. `gymnasium[box2d]` fixes it with a wheel.
"""

from __future__ import annotations

import gymnasium as gym

from rl_tutorials.config import RunSettings

ENV_ID = "LunarLander-v3"
DESCRIPTION = "Random actions on LunarLander (Box2D); see ppo-lunarlander for a trained agent"
SETTINGS_TYPE = RunSettings


def run(settings: RunSettings) -> None:
    env = gym.make(ENV_ID, render_mode=settings.render_mode)
    env.reset(seed=settings.seed)

    for _ in range(settings.episodes):
        env.reset()
        for _ in range(settings.steps):
            # This is where a policy would go — see ppo_lunarlander.py.
            _observation, _reward, terminated, truncated, _info = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                env.reset()

    env.close()
