"""Settings for a demo run, and the defaults -> file -> flags resolution.

Every setting a demo reads has a default here, so a run with no config file always
succeeds and no lookup can raise mid-run. That is the point of the dataclasses: the old
`config['model_basename']` dict lookups failed at the moment of use, sometimes minutes into
a run, and only for the code path that happened to need the key.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """A configuration file could not be read, parsed, or applied."""


@dataclass(frozen=True)
class RunSettings:
    """Settings every demo understands."""

    episodes: int = 5
    steps: int = 100
    render: bool = True
    device: str = "auto"
    seed: int | None = 42

    @property
    def render_mode(self) -> str | None:
        """Gymnasium's `render_mode`; None runs headless, which CI and SSH need."""
        return "human" if self.render else None


@dataclass(frozen=True)
class DQNSettings(RunSettings):
    """Hyperparameters for the from-scratch DQN, carried over from simple_model2."""

    episodes: int = 50
    batch_size: int = 128
    gamma: float = 0.99
    eps_start: float = 0.9
    eps_end: float = 0.05
    eps_decay: int = 1000
    tau: float = 0.005
    lr: float = 1e-4
    memory_capacity: int = 10_000
    plot: bool = True


@dataclass(frozen=True)
class PPOSettings(RunSettings):
    """Settings for the Stable-Baselines3 PPO demo.

    The defaults are rl-baselines3-zoo's tuned LunarLander values, and total_timesteps was
    chosen by measurement rather than taste. Mean return over 10 evaluation episodes on this
    machine, against a random policy's -186:

        100k -> -270   (worse than random; still learning to use the engines)
        200k ->   -8
        400k -> +177
        600k -> +240   <- default; 93s on CPU, and +200 is a reliable landing

    An earlier 60k default with untuned hyperparameters scored -308, i.e. worse than random.
    A demo advertised as "the library version" has to actually work, so it trains long
    enough to land. Lower total_timesteps if you only want to watch it try.
    """

    episodes: int = 5
    total_timesteps: int = 600_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    gae_lambda: float = 0.98
    gamma: float = 0.999
    n_epochs: int = 4
    ent_coef: float = 0.01


# `demo:` names the demo a config belongs to. It is checked, never used to choose the
# demo — the spec is explicit that a config supplies settings and cannot redirect a run.
DEMO_KEY = "demo"


def load_config_file(path: Path) -> dict[str, Any]:
    """Read one YAML config, raising ConfigError with the path on any problem."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(raw).__name__}")
    return raw


def resolve_settings[SettingsT: RunSettings](
    settings_type: type[SettingsT],
    demo_name: str,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SettingsT:
    """Merge defaults, then the config file, then command-line overrides.

    Later sources win. Unknown keys are an error rather than a silent no-op, so a typo
    cannot quietly leave a hyperparameter at its default.
    """
    data: dict[str, Any] = {}

    if config_path is not None:
        file_data = load_config_file(config_path)
        declared = file_data.pop(DEMO_KEY, None)
        if declared is not None and declared != demo_name:
            raise ConfigError(
                f"{config_path}: declares {DEMO_KEY} {declared!r} but {demo_name!r} was "
                "requested; a config supplies settings, it cannot select the demo"
            )
        known = {f.name for f in fields(settings_type)}
        unknown = sorted(set(file_data) - known)
        if unknown:
            raise ConfigError(
                f"{config_path}: unrecognized setting(s): {', '.join(unknown)} "
                f"(valid: {', '.join(sorted(known))})"
            )
        data.update(file_data)

    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    try:
        return settings_type(**data)
    except TypeError as exc:  # a value of the wrong shape for the dataclass
        raise ConfigError(f"invalid settings: {exc}") from exc
