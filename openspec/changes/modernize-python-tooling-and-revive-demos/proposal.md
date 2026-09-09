## Why

Half of this collection does not run. `LunarLander-v3` and `BipedalWalker-v3` raise
`DependencyNotInstalled` because the pinned `mps-requirements.txt` never included Box2D, and
`MsPacman-v0` raises `NameNotFound` because that id was removed from Gymnasium and the demo
still unpacks `env.step()` as the 4-tuple it stopped being in Gym 0.26. Verified on this
machine: of six demos, only the three CartPole ones start.

The tooling around them is equally stale — a `pip freeze` dump instead of a manifest, no lock
file, no linter, no type checker, no tests, and an implicit-namespace import graph that only
works when the interpreter is launched from the repo root. The stated goal is to read this
collection and run it on a Mac; both halves of that are currently blocked.

## What Changes

- **BREAKING** — `train.py -c <config> -m <modelfolder>` is replaced by a named-demo CLI:
  `uv run rl-tutorials <demo>`, with `--list`, `--episodes`, `--steps`, `--device` and an
  optional `-c/--config`. The `alt_model: model1..model6` indirection is deleted; a demo is
  selected by name, not by a key inside a YAML file.
- **BREAKING** — flat root modules move into `src/rl_tutorials/`, and the `tutorialN` /
  `modelN` names are replaced by names that say what the demo does.
- **BREAKING** — `simple_model1..6/config.yaml` are replaced by `configs/<demo>.yaml`. Only
  `simple_model2` held anything the code actually read; the other five were `alt_model`
  markers for the dispatch mechanism being deleted.
- Revive the three dead demos: `gymnasium[box2d]` brings back LunarLander and BipedalWalker
  (Box2D 2.3.10 publishes `macosx_11_0_arm64` wheels — no compiler, no swig), and `ale-py`
  0.12.1 brings back Ms. Pac-Man as `ALE/MsPacman-v5` with ROMs bundled in the wheel.
- Fix the 4-tuple `env.step()` unpack in the Ms. Pac-Man demo.
- Upgrade the stack: Gymnasium 1.0.0 → 1.3.0, torch 2.5.1 → 2.x current. No `step`/`reset`
  signature changed between those Gymnasium versions, so demo bodies stay valid.
- Adopt `uv` — `pyproject.toml` plus a committed `uv.lock` replaces `mps-requirements.txt`.
- Add one Stable-Baselines3 demo (PPO on LunarLander) beside the hand-rolled DQN, so the
  400-line from-scratch version and the 15-line library version sit side by side.
- Delete `utils.py`: nothing imports it, its metrics are NLP (CER/WER/BLEU) from an unrelated
  project, it calls `get_best_model_params_path(config, epoch)` against a one-argument
  definition, and it reads `config['model_basename']` and `config['preload']`, which no
  config defines.
- Make device selection honest. `get_device()` currently prefers MPS unconditionally; for the
  small MLPs here that is typically slower than CPU, and Stable-Baselines3 deliberately does
  not auto-select MPS for `MlpPolicy`. Default to CPU for these workloads, with `--device` to
  override.
- Add ruff, mypy (strict on the package), and a pytest suite that constructs and steps every
  registered demo's environment, so "which demos are broken" is answered by CI and not by
  running six things by hand.

## Capabilities

### New Capabilities

- `demo-catalog`: which demos exist, the name each answers to, the environment it drives, and
  the guarantee that each one constructs and steps on Apple Silicon.
- `demo-runner`: the command-line contract — demo selection by name, discovery via `--list`,
  the run-shaping flags, and exit behavior.
- `runtime-config`: how configuration is resolved (defaults, `configs/<demo>.yaml`, flag
  overrides, and their precedence) and how the compute device is chosen.

### Modified Capabilities

None — `openspec/specs/` is empty; this is the first change to declare specs.

## Impact

- **Code**: every `.py` file at the repo root. `config.py`, `train.py`, `tutorial1.py`,
  `tutorial2.py`, `cartpole.py`, `lunarlander.py`, `bipedalwalker.py`, `packman.py` move and
  are rewritten; `utils.py` is deleted.
- **Config**: `simple_model1..6/` deleted, `configs/` added.
- **Dependencies**: `mps-requirements.txt` deleted. New: `gymnasium[box2d,atari,classic-control,other]`,
  `ale-py`, `stable-baselines3`, `torch`, `pyyaml`; dev: `ruff`, `mypy`, `pytest`.
- **Python**: pinned to 3.13, capped below 3.14. All three of this machine's interpreters were
  tested: 3.12 and 3.13 resolve the full stack; **3.14 has no solution** — `gymnasium[box2d]`
  1.3.0 depends on `box2d==2.3.10`, which publishes wheels up to cp313 and none for cp314.
  3.13 is chosen as the newest that works, and it pulls the current everything
  (torch 2.14.0, stable-baselines3 2.9.0, numpy 2.5.3).
- **CI**: `.github/workflows/ci.yml` gates its `lint`, `type-check` and `test` jobs on
  `pyproject.toml`, which this change adds — those three jobs go from skipped to live.
  `type-check` runs `mypy src`, which the new layout satisfies.
- **Docs**: `README.md` is still the unfilled bootstrap template; its Install/Run/Development
  sections become real. `architecture.md` does not exist and is referenced by `CLAUDE.md` —
  this change creates it.
- **Not touched**: the meta-owned `bin/`, `.claude/`, and `.github/` machinery delivered by
  claude-meta, beyond the `ci.yml` jobs activating on their own.
