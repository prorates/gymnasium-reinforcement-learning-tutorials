"""BipedalWalker under random actions — a continuous-action Box2D environment.

Like lunarlander, this needed Box2D and therefore never started before.
"""

from __future__ import annotations

import gymnasium as gym

from rl_tutorials.config import RunSettings

ENV_ID = "BipedalWalker-v3"
DESCRIPTION = "Random actions on BipedalWalker (Box2D, continuous action space)"
SETTINGS_TYPE = RunSettings


def run(settings: RunSettings) -> None:
    env = gym.make(ENV_ID, hardcore=True, render_mode=settings.render_mode)
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
