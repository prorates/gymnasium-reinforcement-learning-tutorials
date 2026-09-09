"""Config resolution: defaults, then file, then flags — and loud failures in between."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rl_tutorials.config import (
    ConfigError,
    DQNSettings,
    RunSettings,
    load_config_file,
    resolve_settings,
)
from rl_tutorials.demos import DEMOS

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def write(tmp_path: Path, data: object, name: str = "c.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


# --- precedence -------------------------------------------------------------------


def test_defaults_apply_with_no_config_file() -> None:
    settings = resolve_settings(RunSettings, "cartpole")
    assert settings.episodes == RunSettings.episodes
    assert settings.steps == RunSettings.steps


def test_file_overrides_defaults_and_absent_keys_keep_defaults(tmp_path: Path) -> None:
    path = write(tmp_path, {"episodes": 11})
    settings = resolve_settings(RunSettings, "cartpole", path)
    assert settings.episodes == 11
    assert settings.steps == RunSettings.steps


def test_flag_overrides_file(tmp_path: Path) -> None:
    path = write(tmp_path, {"episodes": 11})
    settings = resolve_settings(RunSettings, "cartpole", path, {"episodes": 3})
    assert settings.episodes == 3


def test_none_overrides_do_not_clobber_the_file(tmp_path: Path) -> None:
    """An unset flag is None and must not overwrite what the file said."""
    path = write(tmp_path, {"episodes": 11})
    settings = resolve_settings(RunSettings, "cartpole", path, {"episodes": None, "steps": 7})
    assert settings.episodes == 11
    assert settings.steps == 7


def test_every_field_has_a_default() -> None:
    """No lookup can fail mid-run, because construction with no arguments must succeed."""
    for demo in DEMOS.values():
        demo.settings_type()


# --- error paths ------------------------------------------------------------------


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        resolve_settings(RunSettings, "cartpole", tmp_path / "nope.yaml")


def test_malformed_yaml_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path, "episodes: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        resolve_settings(RunSettings, "cartpole", path)


def test_non_mapping_top_level_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path, [1, 2, 3])
    with pytest.raises(ConfigError, match="must be a mapping"):
        resolve_settings(RunSettings, "cartpole", path)


def test_unknown_key_is_an_error(tmp_path: Path) -> None:
    path = write(tmp_path, {"epsiodes": 11})
    with pytest.raises(ConfigError, match="unrecognized setting"):
        resolve_settings(RunSettings, "cartpole", path)


def test_empty_file_is_fine(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert resolve_settings(RunSettings, "cartpole", path).episodes == RunSettings.episodes


def test_config_cannot_select_a_different_demo(tmp_path: Path) -> None:
    path = write(tmp_path, {"demo": "bipedalwalker", "episodes": 2})
    with pytest.raises(ConfigError, match="cannot select the demo"):
        resolve_settings(RunSettings, "cartpole", path)


def test_matching_demo_key_is_accepted(tmp_path: Path) -> None:
    path = write(tmp_path, {"demo": "cartpole", "episodes": 2})
    assert resolve_settings(RunSettings, "cartpole", path).episodes == 2


# --- the configs that ship --------------------------------------------------------


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_config_parses_names_its_demo_and_uses_known_keys(path: Path) -> None:
    data = load_config_file(path)
    demo_name = data.get("demo")
    assert demo_name in DEMOS, f"{path.name} must name a demo it belongs to"
    # Resolving is the real check: it rejects unknown keys and wrong-typed values.
    resolve_settings(DEMOS[demo_name].settings_type, demo_name, path)


def test_dqn_config_carries_the_hyperparameters() -> None:
    settings = resolve_settings(DQNSettings, "dqn-cartpole", CONFIG_DIR / "dqn-cartpole.yaml")
    assert settings.batch_size == 128
    assert settings.gamma == 0.99
    assert settings.lr == pytest.approx(1e-4)
