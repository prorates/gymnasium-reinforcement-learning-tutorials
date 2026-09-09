"""The CLI contract: listing, selection, and failing before anything is constructed."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_tutorials.__main__ import EXIT_INTERRUPTED, EXIT_USAGE, main
from rl_tutorials.demos import DEMOS


def test_list_exits_zero_and_names_every_demo(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    for demo in DEMOS.values():
        assert demo.name in out
        assert demo.env_id in out


def test_help_enumerates_the_demos(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for name in DEMOS:
        assert name in out


def test_no_demo_named_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "no demo named" in err
    for name in DEMOS:
        assert name in err


def test_unknown_demo_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["not-a-demo"])
    assert exc.value.code == EXIT_USAGE
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--episodes", "--steps"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_counts_rejected(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["cartpole", flag, value])
    assert exc.value.code == EXIT_USAGE
    assert "positive integer" in capsys.readouterr().err


def test_missing_config_fails_before_the_environment_is_built(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cartpole", "-c", str(tmp_path / "absent.yaml")]) == EXIT_USAGE
    assert "not found" in capsys.readouterr().err


def test_unknown_device_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["cartpole", "--device", "tpu"])
    assert exc.value.code == EXIT_USAGE
    assert "invalid choice" in capsys.readouterr().err


def test_a_demo_actually_runs_headless(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["cartpole", "--no-render", "--episodes", "1", "--steps", "3"]) == 0
    assert "Running cartpole" in capsys.readouterr().out


def test_interrupt_exits_cleanly_without_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    """Ctrl-C is a deliberate act, so it gets a message and an exit code, not a stack trace."""
    import signal

    def interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.3)
    try:
        code = main(["cartpole", "--no-render", "--episodes", "100000", "--steps", "100000"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

    assert code == EXIT_INTERRUPTED
    err = capsys.readouterr().err
    assert "Interrupted" in err
    assert "Traceback" not in err
