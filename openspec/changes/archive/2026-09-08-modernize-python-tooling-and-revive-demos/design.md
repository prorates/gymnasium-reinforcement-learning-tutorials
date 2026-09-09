## Context

See `proposal.md § Why` for motivation. The constraints that actually shape the approach:

- **Verified on this machine** (macOS arm64): `gymnasium[box2d,atari,classic-control]==1.3.0`
  installs from wheels only — no compiler, no swig, no ROM download step — and all four of
  `CartPole-v1`, `LunarLander-v3`, `BipedalWalker-v3`, `ALE/MsPacman-v5` construct, reset and
  step. This is the finding the whole change rests on; it was a dependency problem, not a
  code problem.
- **Python 3.13 is the target, and 3.14 is not merely untested — it has no solution.** All
  three interpreters on this machine were resolved against the full dependency set: 3.12 and
  3.13 succeed; 3.14 fails with `box2d==2.3.10 has no wheels with a matching Python ABI tag
  (e.g., cp314)`. Box2D publishes `macosx_11_0_arm64` wheels through cp313 only. Gymnasium's
  unreleased `main` branch does route Python ≥3.14 to `box2d-py` + `swig` instead, so this is
  expected to resolve itself in a later Gymnasium release — but that is a source build, and
  1.3.0 as published on PyPI carries no such marker. 3.13 is chosen as the newest that works
  today; it resolves torch 2.14.0, stable-baselines3 2.9.0 and numpy 2.5.3.
- **Gymnasium's `step`/`reset` signatures did not change between 1.0 and 1.3.** The demo
  bodies are portable as-is; only Ms. Pac-Man's 4-tuple unpack is genuinely broken, and that
  has been wrong since Gym 0.26 regardless of version.
- **The delivered `ci.yml` gates `lint`, `type-check` and `test` on `pyproject.toml`**, and
  `type-check` runs `mypy src`. Adding a `pyproject.toml` activates three CI jobs that have
  been dormant; the `src/` layout is what keeps `type-check` from failing on a missing path.
- These are tutorials. Readability is the product. A clever abstraction that saves twenty
  lines but costs a reader ten minutes is a net loss here.

## Goals / Non-Goals

**Goals:**

- A reader can open one demo file and understand that demo without reading the framework.
- `uv sync && uv run rl-tutorials --list` works from a fresh clone on an Apple Silicon Mac.
- CI proves every catalogued demo starts, so breakage is caught by a machine.

**Non-Goals:**

- Reproducing published results, or tuning any agent to a score target. `ppo-lunarlander`
  exists to be read and to demonstrably learn, not to be state of the art.
- A plugin system, entry-point discovery, or configuration schema language. Seven demos in
  one repository do not need extensibility machinery.
- Preserving the `-c` / `-m` command line or the `simple_modelN/` directories. Both exist to
  serve the `alt_model` dispatch being deleted.
- GPU/CUDA support beyond what torch gives for free.

## Decisions

### Registry as a dict of frozen dataclasses, not a plugin system

`demos/__init__.py` holds one `dict[str, Demo]`, where `Demo` is a frozen dataclass carrying
`name`, `env_id`, `description`, and `run`. The CLI's demo names, `--list` output, and the
parametrized test that starts every environment all read that one dict.

*Why:* the catalog requirement ("every advertised demo runs") is only enforceable if
advertising and running share a source of truth. A registry makes the test
`@pytest.mark.parametrize("demo", DEMOS.values())` — adding a demo automatically adds its
coverage, and a demo that cannot start cannot be quietly listed.

*Alternatives:* entry-point discovery (invisible indirection, and `--list` would depend on
install state); a `match` on the name in the CLI (the list of names and the list of
implementations drift apart — precisely today's `alt_model` bug).

### Settings as a per-demo dataclass, resolved once at startup

Each demo declares its settings as a dataclass with defaults. Resolution merges the YAML file
over the defaults and the CLI flags over that, then constructs the dataclass — so unknown
keys raise at construction, and every field is populated before the demo runs.

*Why:* `runtime-config` requires that no lookup can fail mid-run and that unknown keys are
rejected. Today's `config['model_basename']` KeyError-at-use-time is exactly the failure mode
this removes, and a dataclass gets mypy checking the demo's own field access for free.

*Alternatives:* keep passing `dict[str, Any]` (no type checking, no unknown-key detection,
the current bug survives); pydantic (a dependency and a DSL to learn, for seven flat
structs).

### Rendering is a resolved parameter, never a hard-coded `render_mode="human"`

Every demo takes its `render_mode` from resolved settings, defaulting to `"human"` for
interactive use and set to `None` by `--no-render`.

*Why:* `demo-runner` requires headless operation, and CI has no display. Today four demos
hard-code `render_mode="human"`, which is why none of them could ever run in CI.

### Device defaults to CPU, with an explicit override and a printed choice

`resolve_device()` returns CPU unless `--device` says otherwise; the selection is printed.

*Why:* researched and non-obvious. Stable-Baselines3 deliberately does not auto-select MPS
and recommends CPU for `MlpPolicy`; for networks this small (two 128-unit hidden layers) the
host↔accelerator transfer per step dominates the matmul. The current `get_device()` prefers
MPS unconditionally and is therefore likely making `dqn-cartpole` slower while looking
sophisticated. An unavailable explicit device is an error, not a silent downgrade — a silent
downgrade is how you spend an afternoon wondering why "the GPU" is slow.

*Alternative considered:* keep MPS-preferred and document the caveat. Rejected — the default
should be the fast path, not the impressive-looking one.

**Measured (task 6.1)**, `dqn-cartpole` on this machine, headless, no plotting:

| episodes | CPU | MPS |
| --- | --- | --- |
| 15 | 0.5 s | 2.2 s |
| 60 | 0.8 s | 4.3 s |

MPS is **~4-5x slower**, and the gap widens with episode length — exactly the shape you
expect when per-step transfer dominates. The old `get_device()` preferred MPS
unconditionally, so this collection has been paying that penalty by default. The decision
stands, now on a measurement rather than a citation.

### `tutorial2.py`'s module-level `episode_durations` becomes local state

The DQN demo currently appends to a module-level list, which makes two runs in one process
plot the first run's data. It becomes a local accumulated in the training loop and passed to
the plotting helper.

*Why:* the parametrized "every demo starts" test runs demos in one process; module-level
mutable state makes tests order-dependent. It is also simply a bug.

### Ms. Pac-Man observations are left raw

The demo keeps the `(210, 160, 3)` uint8 observation and takes random actions — no frame
stacking, no grayscale, no resize.

*Why:* its job in the catalog is to show what an image-observation environment looks like.
Preprocessing wrappers are what you add when you train on it, and nothing here trains on it.

## Risks / Trade-offs

- **Python capped below 3.14 by one transitive wheel, on a machine that has 3.14 installed** →
  `requires-python = ">=3.13,<3.14"` with the Box2D reason and the exact resolver error
  written in a comment beside it. This matters more than a normal cap: 3.14 is this machine's
  default `python3`, so without the pin `uv` would pick it and the failure would read as
  "gymnasium is broken" rather than "one transitive wheel lags a release". Revisit when a
  Gymnasium release routes ≥3.14 to `box2d-py` + `swig`.
- **`ppo-lunarlander` is the only demo whose runtime is minutes, not seconds** → the test
  suite constructs its environment but does not train it.

  *This assumption was wrong as first written and is corrected here.* "Small enough to
  finish quickly and learn visibly" turned out not to be one setting: at 60k timesteps with
  untuned hyperparameters the agent scored **-308 against a random policy's -186** — a demo
  advertised as "the library version" that performed worse than random. Measured, with
  rl-baselines3-zoo's tuned LunarLander values:

  | timesteps | elapsed (CPU) | mean return over 10 episodes |
  | --- | --- | --- |
  | 100k | 16 s | -270 |
  | 200k | 32 s | -9 |
  | 400k | 63 s | +177 |
  | **600k** | **93 s** | **+240** ← default |

  600k is the default: +200 is a reliable landing, and 93 s is still "quickly". The lesson
  is that the demo had to be *measured* against a baseline, not eyeballed — which is why
  the random-policy comparison is quoted in the demo's own output.
- **Stable-Baselines3 pins torch `>=2.8` and gymnasium `<2.0`** → an upper bound the lock file
  records. If SB3 lags a future Gymnasium 2.0, `ppo-lunarlander` is the one demo that blocks
  the upgrade; it is deliberately the most removable file in the catalog.
- **This is a near-total rewrite of a personal archive** → every original file is reachable in
  git history, the rename map is written into `README.md`, and the demos keep their original
  structure and comments rather than being re-expressed. The point is that the code still
  reads like the tutorials it came from.
- **ale-py bundles ROMs; that is a licensing posture, not a technical detail** → we depend on
  the published wheel as distributed and add no ROM-fetching step of our own.
- **Three dormant CI jobs go live at once** → they are expected to be green before the PR
  opens; if `mypy --strict` proves noisy on torch's untyped surface, the fallback is
  per-module `ignore_missing_imports` for third-party libraries only, never for our package.

## Migration Plan

No deployment and no consumers — this is a leaf repository. The migration is the reader's:

1. `README.md` carries an old → new table (`train.py -m simple_model2` → `rl-tutorials dqn-cartpole`)
   covering all six original entry points.
2. `mps-requirements.txt` and `simple_model1..6/` are deleted in the same commit that adds
   `pyproject.toml`, `uv.lock` and `configs/`, so no clone is ever half-migrated.
3. Rollback is `git revert`; nothing outside the repository is touched.

## Open Questions

None. The four scope decisions (layout, demo revival, CLI shape, `utils.py`) were settled with
the operator before this document, and the dependency question was settled by installing the
stack and running all four environments.
