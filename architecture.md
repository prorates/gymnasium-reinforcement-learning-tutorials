# architecture

How this collection is organised, and the three invariants worth knowing before you change
it. Motivation and history live in `README.md` and `openspec/`; this is the shape of the code.

## Layout

```
src/rl_tutorials/
  __main__.py        the CLI — argparse, error paths, the console entry point
  config.py          settings dataclasses + the defaults -> file -> flags merge
  device.py          choose_device(): which torch device, and why it is CPU
  demos/
    __init__.py      Demo dataclass + the DEMOS registry
    <demo>.py        one file per demo
configs/<demo>.yaml  optional settings, one file per demo that has tunables
tests/               catalog, config, device and CLI tests
bin/  .claude/       NOT OURS — see "What this repo does not own"
```

A demo module is deliberately readable end to end. These are tutorials: if a change makes a
demo shorter but means a reader has to hold two files in their head, it is the wrong change.

## The demo contract

A demo module exports exactly four names, and imports nothing from `demos/__init__.py`:

| name | type | what it is |
| --- | --- | --- |
| `ENV_ID` | `str` | the Gymnasium id it constructs |
| `DESCRIPTION` | `str` | one line, shown by `--list` |
| `SETTINGS_TYPE` | `type[RunSettings]` | the dataclass its settings resolve into |
| `run` | `(settings) -> None` | does the thing; owns its env from `gym.make` to `close` |

`demos/__init__.py` reads those four and assembles a `Demo` for each. The direction matters:
the registry imports the demos, never the reverse, so there is no import cycle and a demo
file can be read without the framework.

**To add a demo:** write the module with those four names, add one `_demo(...)` line to
`DEMOS`. Nothing else — the CLI, `--list`, `--help` and the catalog test all read that dict.

### Invariant 1 — the registry is the only catalog

`DEMOS` is the single source of truth for the names the CLI accepts, what `--list` prints,
and what the tests start. This is not tidiness. The previous arrangement kept the valid names
in a `match` statement and the implementations elsewhere, and **three of six demos were dead
for an unknown length of time** without anything noticing. Because the catalog test is
parametrized over `DEMOS`, a demo cannot be advertised without also being started by CI.

Never hard-code a demo name anywhere but `DEMOS`.

### Invariant 2 — every setting has a default, and unknown keys are errors

`RunSettings` and its per-demo subclasses give every field a default, so `settings_type()`
with no arguments always succeeds and no demo can raise `KeyError` mid-run. The old code
passed `dict[str, Any]` around and read `config['model_basename']` at the point of use — a
key no config defined, so it failed minutes into whichever run happened to reach that line.

`resolve_settings()` merges defaults, then the YAML file, then CLI overrides (an unset flag
is `None` and does not clobber the file). A key the dataclass does not declare is a
`ConfigError`, not a silent no-op, so a typo cannot leave a hyperparameter quietly at its
default. A config's `demo:` key is checked against the demo requested; it can never redirect
the run.

Everything that can fail without touching an environment — bad config, bad flag, unknown
device — fails in `main()` before `gym.make` is called.

### Invariant 3 — rendering is a parameter, never a constant

Every demo takes `render_mode` from `settings.render_mode`. Nothing hard-codes
`render_mode="human"`. That is what makes `--no-render` work, and it is why the catalog test
can run under CI with no display; four of the original demos hard-coded it and therefore
could never have been tested.

## Device selection

`choose_device("auto")` returns **CPU**, and that is a measured decision, not a fallback:

| `dqn-cartpole`, 60 episodes | CPU | MPS |
| --- | --- | --- |
| wall clock | 0.8 s | 4.3 s |

At two 128-unit hidden layers over a 4-element observation, the per-step host↔accelerator
copy costs more than the matmul saves. Stable-Baselines3 reaches the same conclusion — it
does not auto-select MPS and recommends CPU for `MlpPolicy`. An explicitly requested device
that is unavailable raises rather than silently downgrading, because a silent downgrade is
indistinguishable from a slow GPU.

## What this repo does not own

`bin/`, `.claude/`, `.github/` and `.pre-commit-config.yaml` arrive by broadcast from
claude-meta (class M). Do not edit them to fix a lint finding or a bug — the propagation path
is a 3-way merge, so a local edit becomes a conflict on the next delivery. Report through
`/alemax:feedback` instead, and let the fix arrive by broadcast.

This is why `[tool.ruff.lint] exclude` lists `bin/**` and `.claude/**`: 96 of the 100 findings
on the first lint run lived there. They are still format-checked, because `line-length = 100`
matches how meta formats them — without that setting ruff defaults to 88 and rewrites all 31
delivered files.
