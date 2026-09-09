"""PPO on LunarLander, using Stable-Baselines3.

This is the short version of `dqn_cartpole.py`. That file spends ~200 lines on a replay
buffer, a target network, an epsilon schedule and a training loop; here the same category
of work is three statements, because the library owns all of it.

Worth reading them side by side. The library version is what you would actually reach for;
the from-scratch version is how you learn what the library is doing.

Note the device: SB3 deliberately does not auto-select MPS and recommends CPU for
MlpPolicy, which is the same conclusion `rl_tutorials.device.choose_device` reaches.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from rl_tutorials.config import PPOSettings
from rl_tutorials.device import choose_device

ENV_ID = "LunarLander-v3"
DESCRIPTION = "PPO on LunarLander via stable-baselines3 (the library version of dqn-cartpole)"
SETTINGS_TYPE = PPOSettings


def run(settings: PPOSettings) -> None:
    device = choose_device(settings.device)

    # Train headless — rendering every frame of training is the slowest way to watch a
    # policy that is not good yet.
    train_env = gym.make(ENV_ID)
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=settings.learning_rate,
        n_steps=settings.n_steps,
        batch_size=settings.batch_size,
        gae_lambda=settings.gae_lambda,
        gamma=settings.gamma,
        n_epochs=settings.n_epochs,
        ent_coef=settings.ent_coef,
        device=device,
        seed=settings.seed,
        verbose=1,
    )
    model.learn(total_timesteps=settings.total_timesteps)
    train_env.close()

    # Then watch it, if a display was asked for. Monitor records true episode returns and
    # lengths; without it SB3 warns that other wrappers may have altered what it reports.
    # LunarLander: an 8-float Box observation and a discrete action.
    eval_env: Monitor[np.ndarray, np.int64] = Monitor(
        gym.make(ENV_ID, render_mode=settings.render_mode)
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=settings.episodes, render=settings.render
    )
    eval_env.close()

    print(f"Mean reward over {settings.episodes} episodes: {mean_reward:.1f} +/- {std_reward:.1f}")
    print("(A random policy scores about -186; +200 is a reliable landing.)")
