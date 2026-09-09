"""Ms. Pac-Man under random actions — an image-observation Atari environment.

Two things were broken here, and they are worth knowing apart:

1. The id. `MsPacman-v0` was retired; the ALE environments now live under an `ALE/`
   namespace that has to be registered before `gym.make` can see it. Without the
   registration you get `NamespaceNotFound`, with the old id you get `NameNotFound`.
2. The step tuple. This demo unpacked `env.step()` into four values. Gym 0.26 split the
   old `done` into `terminated` and `truncated`, making it five — so this raised
   ValueError on the first step of the first episode, independently of the id problem.

Observations are left raw at (210, 160, 3) uint8. Grayscale, resize and frame-stacking are
what you add when you train on Atari; nothing here trains, and the point of this demo is to
show what an image observation looks like next to CartPole's four floats.
"""

from __future__ import annotations

import time

import ale_py
import gymnasium as gym

from rl_tutorials.config import RunSettings

ENV_ID = "ALE/MsPacman-v5"
DESCRIPTION = "Random actions on Ms. Pac-Man (Atari, 210x160x3 image observations)"
SETTINGS_TYPE = RunSettings

# ROMs ship inside the ale-py wheel, so this needs no download and no license step.
gym.register_envs(ale_py)


def run(settings: RunSettings) -> None:
    env = gym.make(ENV_ID, render_mode=settings.render_mode)
    env.reset(seed=settings.seed)

    for episode in range(settings.episodes):
        env.reset()
        total_reward = 0.0

        for _ in range(settings.steps):
            # Five values, not four.
            _observation, reward, terminated, truncated, _info = env.step(env.action_space.sample())
            total_reward += float(reward)
            if settings.render:
                time.sleep(0.01)
            if terminated or truncated:
                break

        print(f"Episode {episode + 1}/{settings.episodes}, total reward: {total_reward}")

    env.close()
