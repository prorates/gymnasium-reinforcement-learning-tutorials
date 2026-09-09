# gymnasium-reinforcement-learning-tutorials

> Gymnasium reinforcement-learning tutorial scripts and models, adapted for Apple Silicon.

A personal collection of reinforcement-learning demos gathered over several years, brought
up to a current Python toolchain and made to actually run on an Apple Silicon Mac. Each demo
is a single readable file driving one Gymnasium environment: five show the raw environment
API under random actions, and two train an agent — one DQN written from scratch, one PPO
from a library — so the two approaches can be read side by side. It is a leaf: it publishes
nothing and nothing depends on it.

## Install

    uv sync

Python 3.13 (see [Requirements](#requirements) — 3.14 does not work, and the reason is
worth knowing).

## Run

    uv run rl-tutorials --list          # every demo, with its environment
    uv run rl-tutorials dqn-cartpole    # run one

Useful flags: `--episodes N`, `--steps N`, `--no-render` (headless, for SSH or CI),
`--device {auto,cpu,mps,cuda}`, `--seed N`, `-c configs/<demo>.yaml`.

### The demos

| name | environment | what it shows |
| --- | --- | --- |
| `random-cartpole` | `CartPole-v1` | the raw API — observation and action spaces, printed |
| `cartpole` | `CartPole-v1` | the smallest possible reset/step loop |
| `dqn-cartpole` | `CartPole-v1` | **Deep Q-Learning written from scratch** — replay buffer, target network, epsilon schedule |
| `lunarlander` | `LunarLander-v3` | a Box2D environment under random actions |
| `bipedalwalker` | `BipedalWalker-v3` | a continuous action space |
| `mspacman` | `ALE/MsPacman-v5` | image observations (210×160×3) instead of four floats |
| `ppo-lunarlander` | `LunarLander-v3` | **the same job as `dqn-cartpole`, done by Stable-Baselines3** in a few lines |

Read `dqn-cartpole` and `ppo-lunarlander` together: the first is how it works, the second is
what you would actually write.

## Configuration

No environment variables. Settings resolve in one order — built-in defaults, then a YAML file
if you pass `-c`, then command-line flags — and every setting has a default, so a run with no
config file always works.

| file | belongs to |
| --- | --- |
| `configs/dqn-cartpole.yaml` | `dqn-cartpole` — replay buffer size, epsilon schedule, learning rate |
| `configs/ppo-lunarlander.yaml` | `ppo-lunarlander` — training length and PPO hyperparameters |

A config supplies settings only; it cannot select which demo runs. Its `demo:` key is checked
against the demo you asked for, and an unrecognized key is an error rather than a silent
no-op.

## Requirements

Python **3.13**, and not 3.14. `gymnasium[box2d]` depends on `box2d==2.3.10`, which publishes
wheels through cp313 and none for cp314, so a 3.14 environment fails to resolve at all:

    Because box2d==2.3.10 has no wheels with a matching Python ABI tag (e.g., `cp314`)
    and gymnasium[box2d]>=1.3.0 depends on box2d==2.3.10, ...requirements are unsatisfiable.

`.python-version` and the `requires-python` cap in `pyproject.toml` exist to keep `uv` off
3.14. Gymnasium's unreleased `main` routes ≥3.14 to `box2d-py` + `swig` (a source build), so
the cap can lift once a release carries that.

Atari ROMs ship inside the `ale-py` wheel — there is no download or license step.

## Development

    uv run pytest                 # 48 tests; every demo's environment is constructed and stepped
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src
    pre-commit run --all-files

The test suite starts every demo in the catalog, which is what stops a demo from silently
rotting — see [Migrating](#migrating-from-the-old-layout).

`bin/` and `.claude/` are excluded from ruff **linting**: they arrive by broadcast from
claude-meta, so a local fix there becomes a merge conflict on the next delivery. They are
still format-checked.

## Migrating from the old layout

Demos used to be selected indirectly — `train.py` read an `alt_model: modelN` key out of a
YAML file and dispatched on it, so you could not tell what would run without opening two
files. Now the demo is named directly:

| before | now |
| --- | --- |
| `python train.py -m simple_model1` | `uv run rl-tutorials random-cartpole` |
| `python train.py -m simple_model2` | `uv run rl-tutorials dqn-cartpole` |
| `python train.py -m simple_model3` | `uv run rl-tutorials mspacman` |
| `python train.py -m simple_model4` | `uv run rl-tutorials cartpole` |
| `python train.py -m simple_model5` | `uv run rl-tutorials lunarlander` |
| `python train.py -m simple_model6` | `uv run rl-tutorials bipedalwalker` |
| — | `uv run rl-tutorials ppo-lunarlander` (new) |

Three of those six could not run at all before this change: `lunarlander` and
`bipedalwalker` raised `DependencyNotInstalled` because Box2D was never in the requirements,
and `mspacman` used the retired `MsPacman-v0` id *and* unpacked `env.step()` as a 4-tuple,
which stopped being correct in Gym 0.26.

`simple_model1..6/` and `mps-requirements.txt` are gone; `utils.py` is gone too — nothing
imported it, and its metrics (character error rate, BLEU) belonged to an unrelated NLP
project. All of it remains in git history.

## Documentation

- [`architecture.md`](architecture.md) — how the code is organised; read it before changing it
- [`CLAUDE.md`](CLAUDE.md) — what a Claude session reads at start; a router, not a manual
- `openspec/` — `specs/` is what it must do; `changes/` is work in flight
