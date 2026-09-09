"""Command line entry point: `rl-tutorials <demo>`.

A demo is chosen by name. It used to be chosen by an `alt_model: modelN` key inside a YAML
file, which meant you could not tell what would run without opening two files, and the
names carried no information.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rl_tutorials.config import ConfigError, resolve_settings
from rl_tutorials.demos import DEMOS
from rl_tutorials.device import VALID_DEVICES, DeviceError

EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    names = ", ".join(DEMOS)
    parser = argparse.ArgumentParser(
        prog="rl-tutorials",
        description="Run one Gymnasium reinforcement-learning tutorial demo.",
        epilog=f"demos: {names}\n\nUse --list for a description of each.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "demo", nargs="?", choices=list(DEMOS), metavar="DEMO", help=f"which demo to run ({names})"
    )
    parser.add_argument("--list", action="store_true", help="list the demos and exit")
    parser.add_argument("--episodes", type=positive_int, help="number of episodes to run")
    parser.add_argument("--steps", type=positive_int, help="maximum steps per episode")
    parser.add_argument(
        "--device", choices=VALID_DEVICES, help="compute device (default: auto, which is cpu)"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="run headless (no window; needed over SSH and CI)"
    )
    parser.add_argument("--seed", type=int, help="environment seed")
    parser.add_argument("-c", "--config", type=Path, help="YAML file of settings for this demo")
    return parser


def print_catalog() -> None:
    width = max(len(name) for name in DEMOS)
    env_width = max(len(demo.env_id) for demo in DEMOS.values())
    for demo in DEMOS.values():
        print(f"  {demo.name:<{width}}  {demo.env_id:<{env_width}}  {demo.description}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_catalog()
        return 0

    if args.demo is None:
        parser.print_usage(sys.stderr)
        print(
            f"{parser.prog}: error: no demo named. Choose one of: {', '.join(DEMOS)}",
            file=sys.stderr,
        )
        print(f"{parser.prog}: try --list for a description of each.", file=sys.stderr)
        return EXIT_USAGE

    demo = DEMOS[args.demo]

    overrides: dict[str, Any] = {
        "episodes": args.episodes,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        # store_true is False when absent, which would override the config file; only pass
        # it through when the flag was actually given.
        "render": False if args.no_render else None,
    }

    # Everything that can fail without touching an environment fails here.
    try:
        settings = resolve_settings(demo.settings_type, demo.name, args.config, overrides)
    except ConfigError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(f"Running {demo.name} ({demo.env_id})")
    try:
        demo.run(settings)
    except DeviceError as exc:
        print(f"{parser.prog}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        # gym.Env closes its window in __del__; the point here is to not dump a traceback
        # at someone who pressed Ctrl-C on purpose.
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    return 0


if __name__ == "__main__":
    sys.exit(main())
