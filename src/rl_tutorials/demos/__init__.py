"""The demo catalog.

One dict is the single source of truth for three things: the names the CLI accepts, what
`--list` prints, and what the catalog test starts. They cannot drift apart, which is the
whole point — the previous arrangement had the list of valid names in one place (a `match`
on `alt_model`) and the demos in another, and three of the six had been dead for a while
without anything noticing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rl_tutorials.config import RunSettings

from . import (
    bipedalwalker,
    cartpole,
    dqn_cartpole,
    lunarlander,
    mspacman,
    ppo_lunarlander,
    random_cartpole,
)


@dataclass(frozen=True)
class Demo:
    """One runnable demo: what it is called, what it drives, and how to start it."""

    name: str
    env_id: str
    description: str
    run: Callable[[Any], None]
    settings_type: type[RunSettings]


def _demo(name: str, module: Any) -> Demo:
    return Demo(
        name=name,
        env_id=module.ENV_ID,
        description=module.DESCRIPTION,
        run=module.run,
        settings_type=module.SETTINGS_TYPE,
    )


DEMOS: dict[str, Demo] = {
    demo.name: demo
    for demo in (
        _demo("random-cartpole", random_cartpole),
        _demo("cartpole", cartpole),
        _demo("dqn-cartpole", dqn_cartpole),
        _demo("lunarlander", lunarlander),
        _demo("bipedalwalker", bipedalwalker),
        _demo("mspacman", mspacman),
        _demo("ppo-lunarlander", ppo_lunarlander),
    )
}

__all__ = ["DEMOS", "Demo"]
