"""Deep Q-Learning (DQN) on CartPole, implemented from scratch.

After the PyTorch tutorial by Adam Paszke and Mark Towers. This is the long way round —
see `ppo_lunarlander.py` for the same job done by a library in fifteen lines.

The idea: if we knew Q*(state, action) — the return from taking an action in a state — the
best policy would just be argmax over actions. We don't, so we approximate Q* with a small
network and train it against the Bellman equation

    Q(s, a) = r + gamma * max_a' Q(s', a')

using Huber loss, which behaves like squared error for small residuals and absolute error
for large ones, so a wildly wrong Q estimate cannot dominate a batch.

Two details make it stable, and both are why this is more than a regression problem:
  * a replay buffer, so consecutive (correlated) transitions do not form a batch;
  * a separate target network, updated slowly, so the regression target does not move
    every time the network being trained does.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from itertools import count

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn, optim

from rl_tutorials.config import DQNSettings
from rl_tutorials.device import choose_device

ENV_ID = "CartPole-v1"
DESCRIPTION = "Deep Q-Learning on CartPole, written from scratch (compare ppo-lunarlander)"
SETTINGS_TYPE = DQNSettings


@dataclass(frozen=True)
class Transition:
    """One (state, action) -> (next_state, reward) step. next_state is None if terminal."""

    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor | None
    reward: torch.Tensor


class ReplayMemory:
    """A bounded cyclic buffer of recent transitions, sampled uniformly at random."""

    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self.memory), batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    """Two hidden layers of 128 units. Input is the raw 4-element observation.

    Output is one expected return per action, so `forward` gives Q(s, left) and Q(s, right).
    """

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        out: torch.Tensor = self.layer3(x)
        return out


def select_action(
    state: torch.Tensor,
    steps_done: int,
    env: gym.Env,  # type: ignore[type-arg]
    policy_net: DQN,
    device: torch.device,
    settings: DQNSettings,
) -> torch.Tensor:
    """Epsilon-greedy: mostly the policy, sometimes a coin flip, decaying towards the policy."""
    eps_threshold = settings.eps_end + (settings.eps_start - settings.eps_end) * math.exp(
        -1.0 * steps_done / settings.eps_decay
    )
    if random.random() > eps_threshold:
        with torch.no_grad():
            # max(1).indices is the action with the larger expected return.
            q_values: torch.Tensor = policy_net(state)
            return q_values.max(1).indices.view(1, 1)
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model(
    memory: ReplayMemory,
    optimizer: optim.Optimizer,
    policy_net: DQN,
    target_net: DQN,
    device: torch.device,
    settings: DQNSettings,
) -> None:
    """One gradient step on a random batch from the replay buffer."""
    if len(memory) < settings.batch_size:
        return
    transitions = memory.sample(settings.batch_size)

    # A final state has no successor, so it contributes 0 to the expected return. Mask
    # those out rather than feeding the target network a placeholder.
    non_final_mask = torch.tensor(
        [t.next_state is not None for t in transitions], device=device, dtype=torch.bool
    )
    non_final_next_states = torch.cat(
        [t.next_state for t in transitions if t.next_state is not None]
    )
    state_batch = torch.cat([t.state for t in transitions])
    action_batch = torch.cat([t.action for t in transitions])
    reward_batch = torch.cat([t.reward for t in transitions])

    # Q(s_t, a_t): run the policy net, then pick the column of the action actually taken.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1}) from the *target* net — the slowly-moving copy.
    next_state_values = torch.zeros(settings.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * settings.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def plot_durations(
    episode_durations: list[int], *, show_result: bool = False, is_ipython: bool = False
) -> None:
    """Plot episode length over time, plus a 100-episode moving average once there is one.

    Durations are passed in rather than read from a module-level list. The original kept
    them at module scope, so a second run in the same process plotted the first run's data
    on top of its own — which the catalog test, running every demo in one process, hits.
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        # Imported here, at the one place it is used. The original imported it inside a
        # setup function and referenced it from this one, so it was a NameError waiting
        # for anyone who ran under an inline backend.
        from IPython import display

        display.display(plt.gcf())
        if not show_result:
            display.clear_output(wait=True)


def run(settings: DQNSettings) -> None:
    env = gym.make(ENV_ID, render_mode=settings.render_mode)
    device = choose_device(settings.device)

    # Headless implies no plot window: --no-render is what CI and SSH use, and
    # matplotlib warns and no-ops on show() under a non-interactive backend anyway.
    plotting = settings.plot and settings.render
    is_ipython = plotting and "inline" in matplotlib.get_backend()
    if plotting:
        plt.ion()

    n_actions = int(env.action_space.n)  # type: ignore[attr-defined]
    state_array, _info = env.reset(seed=settings.seed)
    n_observations = len(state_array)

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=settings.lr, amsgrad=True)
    memory = ReplayMemory(settings.memory_capacity)

    steps_done = 0
    episode_durations: list[int] = []

    for i_episode in range(settings.episodes):
        print(f"Episode {i_episode + 1}/{settings.episodes}")
        state_array, _info = env.reset()
        state = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            action = select_action(state, steps_done, env, policy_net, device, settings)
            steps_done += 1
            observation, reward, terminated, truncated, _info = env.step(int(action.item()))
            reward_t = torch.tensor([reward], device=device, dtype=torch.float32)

            next_state = (
                None
                if terminated
                else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            )
            memory.push(Transition(state, action, next_state, reward_t))

            optimize_model(memory, optimizer, policy_net, target_net, device, settings)

            # Soft update of the target network: theta' <- tau*theta + (1-tau)*theta'.
            # Nudging it every step is what keeps the regression target from lurching.
            target_state = target_net.state_dict()
            policy_state = policy_net.state_dict()
            for key in policy_state:
                target_state[key] = policy_state[key] * settings.tau + target_state[key] * (
                    1 - settings.tau
                )
            target_net.load_state_dict(target_state)

            if terminated or truncated:
                episode_durations.append(t + 1)
                if plotting:
                    plot_durations(episode_durations, is_ipython=is_ipython)
                break
            state = next_state if next_state is not None else state

    print("Complete")
    if plotting:
        plot_durations(episode_durations, show_result=True, is_ipython=is_ipython)
        plt.ioff()
        plt.show()
    env.close()
