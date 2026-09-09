"""Compute-device selection.

The interesting part is the `auto` default, which is CPU — see `choose_device`.
"""

from __future__ import annotations

import torch

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


class DeviceError(Exception):
    """A device was requested that this machine cannot provide."""


def choose_device(requested: str = "auto") -> torch.device:
    """Return the torch device to train on, printing the choice.

    `auto` resolves to CPU on purpose. These demos train two 128-unit hidden layers on
    4-to-24-element observations; at that size the per-step host<->accelerator copy costs
    more than the matmul saves, so MPS is typically *slower* than CPU here.
    Stable-Baselines3 makes the same call — it does not auto-select MPS, and recommends CPU
    for MlpPolicy. Ask for `--device mps` explicitly if you want to measure it.

    An unavailable explicit request is an error, not a silent downgrade: quietly falling
    back to CPU is how you spend an afternoon wondering why "the GPU" is no faster.
    """
    if requested not in VALID_DEVICES:
        raise DeviceError(f"unknown device {requested!r}; choose from: {', '.join(VALID_DEVICES)}")

    if requested in ("auto", "cpu"):
        chosen = "cpu"
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            raise DeviceError("mps requested but this machine has no Metal device available")
        chosen = "mps"
    else:  # cuda
        if not torch.cuda.is_available():
            raise DeviceError("cuda requested but no CUDA device is available")
        chosen = "cuda"

    suffix = " (auto; see choose_device for why not mps)" if requested == "auto" else ""
    print(f"Using device: {chosen}{suffix}")
    return torch.device(chosen)
