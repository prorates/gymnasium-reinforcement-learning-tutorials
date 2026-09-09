"""Every advertised demo must actually start.

This is the test the collection did not have. Three of six demos had been dead — two on a
missing Box2D, one on a retired environment id — and nothing said so until you ran them by
hand, one at a time, with a display attached.
"""

from __future__ import annotations

import gymnasium as gym
import pytest

from rl_tutorials.demos import DEMOS, Demo


def test_registry_keys_match_demo_names() -> None:
    for key, demo in DEMOS.items():
        assert key == demo.name


def test_catalog_covers_the_expected_demos() -> None:
    assert set(DEMOS) == {
        "random-cartpole",
        "cartpole",
        "dqn-cartpole",
        "lunarlander",
        "bipedalwalker",
        "mspacman",
        "ppo-lunarlander",
    }


@pytest.mark.parametrize("demo", DEMOS.values(), ids=list(DEMOS))
def test_demo_environment_constructs_resets_and_steps(demo: Demo) -> None:
    """Construct, reset and step each demo's environment headless.

    A demo whose dependency is missing fails here, named, rather than at the moment a user
    tries to run it.
    """
    try:
        env = gym.make(demo.env_id)
    except gym.error.Error as exc:  # DependencyNotInstalled, NameNotFound, NamespaceNotFound
        pytest.fail(f"demo {demo.name!r} cannot construct {demo.env_id!r}: {exc}")

    try:
        observation, info = env.reset(seed=0)
        assert observation is not None
        assert isinstance(info, dict)

        result = env.step(env.action_space.sample())
        assert len(result) == 5, "gymnasium returns (obs, reward, terminated, truncated, info)"
        _obs, _reward, terminated, truncated, _info = result
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


@pytest.mark.parametrize("demo", DEMOS.values(), ids=list(DEMOS))
def test_demo_metadata_is_populated(demo: Demo) -> None:
    assert demo.name and demo.name == demo.name.lower()
    assert " " not in demo.name, "names are kebab-case, usable as a CLI argument"
    assert demo.env_id
    assert demo.description
    assert callable(demo.run)
