"""Device selection: CPU by default, explicit override, hard error when unavailable."""

from __future__ import annotations

import pytest
import torch

from rl_tutorials.device import DeviceError, choose_device


def test_auto_selects_cpu() -> None:
    """Deliberate: at two 128-unit layers the accelerator copy costs more than it saves."""
    assert choose_device("auto").type == "cpu"


def test_cpu_is_honoured() -> None:
    assert choose_device("cpu").type == "cpu"


def test_choice_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    choose_device("auto")
    assert "Using device: cpu" in capsys.readouterr().out


def test_unknown_device_raises() -> None:
    with pytest.raises(DeviceError, match="unknown device"):
        choose_device("tpu")


def test_unavailable_device_raises_rather_than_downgrading() -> None:
    """A silent fallback to CPU is how you spend an afternoon wondering why MPS is slow."""
    if torch.cuda.is_available():  # pragma: no cover - not on this machine
        pytest.skip("this machine has CUDA")
    with pytest.raises(DeviceError, match="cuda"):
        choose_device("cuda")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Metal device")
def test_mps_is_honoured_when_available() -> None:
    assert choose_device("mps").type == "mps"
