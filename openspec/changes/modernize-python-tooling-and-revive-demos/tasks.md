## 1. Toolchain foundation

- [x] 1.1 Write `pyproject.toml`: hatchling backend, `requires-python = ">=3.13,<3.14"` with
      the Box2D-cp314-wheel reason and the resolver error quoted in a comment, runtime deps
      (`gymnasium[box2d,atari,classic-control,other]`, `ale-py`, `torch`, `pyyaml`,
      `stable-baselines3`), a `dev` group (`ruff`, `mypy`, `pytest`), and the
      `rl-tutorials` console script. Verify `uv sync` resolves and `uv run python -c "import
      rl_tutorials"` succeeds.
- [x] 1.2 Add `[tool.ruff]` with `line-length = 100` and `[tool.mypy]` strict for
      `src/rl_tutorials`, with `ignore_missing_imports` scoped to third-party modules only.
      Verify `uv run ruff check .` and `uv run mypy src` both run (findings expected until
      §3 lands) — and that `ruff format --check` no longer reformats the meta-owned
      `bin/*.py`, which today's missing config causes.
- [x] 1.3 Commit `uv.lock`; add `.python-version` pinning 3.13. Verify a clean
      `uv sync --locked` succeeds, and that it does so without picking up this machine's
      default `python3` (3.14), which cannot resolve `gymnasium[box2d]`.
- [x] 1.4 Delete `mps-requirements.txt`. Verify nothing references it (`grep -rn
      mps-requirements`).

## 2. Package skeleton

- [x] 2.1 Create `src/rl_tutorials/{__init__.py,__main__.py}` and `src/rl_tutorials/demos/`.
      Verify `uv run rl-tutorials --help` exits zero.
- [x] 2.2 Implement `Demo` (frozen dataclass: `name`, `env_id`, `description`, `run`) and the
      `DEMOS` registry dict in `demos/__init__.py`. Verify a unit test asserts each key
      equals its `Demo.name`.
- [x] 2.3 Implement `config.py`: per-demo settings dataclasses, YAML load, and the
      defaults → file → flags merge. Verify unit tests cover each precedence scenario in
      `specs/runtime-config/spec.md`, plus missing-file, malformed-YAML, and unknown-key
      errors.
- [x] 2.4 Implement `device.py` `resolve_device()` — CPU default, explicit override,
      hard error on an unavailable request, prints its choice. Verify unit tests cover the
      three device scenarios in the spec.

## 3. Port the demos

- [x] 3.1 Port `tutorial1.py` → `demos/random_cartpole.py`, keeping its space-introspection
      prints. Drop the unused `initial_state`/`appendedObservations` locals and read
      `env.spec` after a reset, not after a bare step. Verify it runs headless for 2 episodes.
- [x] 3.2 Port `cartpole.py` → `demos/cartpole.py`. Verify it runs headless and honors
      `--episodes`.
- [x] 3.3 Port `tutorial2.py` → `demos/dqn_cartpole.py`: keep the DQN, `ReplayMemory` and the
      explanatory comments; move `episode_durations` from module scope into the training
      loop (design § module-level state); fix the `display` NameError by importing IPython at
      module scope behind the backend check. Verify a 2-episode run completes and a second
      run in the same process plots only its own data.
- [x] 3.4 Port `lunarlander.py` and `bipedalwalker.py` → `demos/`. Verify both construct,
      reset and step — the Box2D dependency being the whole reason they were dead.
- [x] 3.5 Port `packman.py` → `demos/mspacman.py`: register the ALE namespace, use
      `ALE/MsPacman-v5`, and **fix the 4-tuple `env.step()` unpack to the 5-tuple**. Verify it
      steps and accumulates reward without raising.
- [x] 3.6 Write `demos/ppo_lunarlander.py` using Stable-Baselines3 PPO, short enough to read
      in one screen, with a comment pointing at `dqn_cartpole.py` as the from-scratch
      counterpart. Verify a short-timestep run trains and its mean episode return beats a
      random policy's.
- [x] 3.7 Delete `utils.py` and the six `simple_modelN/` directories; add `configs/` holding
      the DQN hyperparameters salvaged from `simple_model2/config.yaml`. Verify `grep -rn
      "model_basename\|alt_model\|preload"` returns nothing outside `openspec/`.

## 4. CLI

- [x] 4.1 Implement the argparse CLI in `__main__.py`: demo name positional, `--list`,
      `--episodes`, `--steps`, `--device`, `--no-render`, `-c/--config`. Verify
      `rl-tutorials --list` prints all seven demos with env ids.
- [x] 4.2 Implement the error paths — no demo named, unknown demo, non-positive counts,
      missing config file — each exiting non-zero before any environment is constructed.
      Verify tests assert exit codes and that the error names the valid demos.
- [x] 4.3 Handle `KeyboardInterrupt`: close the environment, print a short message, exit
      non-zero with no traceback. Verify by sending an interrupt to a running demo.
- [x] 4.4 Delete `train.py`. Verify `rl-tutorials` covers all six original entry points via
      the README rename table.

## 5. Tests and CI

- [x] 5.1 Write the parametrized catalog test: for every `Demo` in `DEMOS`, construct, reset
      and step its environment headless. Verify all seven pass and that removing a dependency
      makes exactly that demo fail with a message naming it.
- [x] 5.2 Write a test loading every file in `configs/`, asserting it parses, contains only
      recognized keys, and names an existing demo.
- [x] 5.3 Get `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src` and
      `uv run pytest` all green. Verify locally, then confirm the three previously-dormant
      CI jobs (`lint`, `type-check`, `test`) run and pass on the PR.

## 6. Benchmark the device decision

- [x] 6.1 Time `dqn-cartpole` for a fixed episode count on CPU and on MPS on this machine.
      Verify the measured numbers are recorded in `design.md § Decisions` beside the device
      rationale — if MPS wins, change the default and say so.

## 7. Documentation

- [x] 7.1 Fill in `README.md`: what the collection is, `uv sync`, `uv run rl-tutorials --list`,
      a table of the seven demos, the Development commands, and the old → new rename table
      for all six original entry points. Verify no `<placeholder>` text remains.
- [x] 7.2 Write `architecture.md` — registry, config resolution, demo contract, and the
      package boundary. Verify `uv run --script bin/check-doc-set.py` drops both of its
      current warnings.
- [x] 7.3 Update `CLAUDE.md` § 1 stack/run line and § 2 data table to match reality. Verify
      `uv run --script bin/claude-md-check.py` passes and the startup budget is unchanged.
- [ ] 7.4 Open the PR with CI green. Verify `openspec validate --changes
      modernize-python-tooling-and-revive-demos --strict` passes before archiving.
