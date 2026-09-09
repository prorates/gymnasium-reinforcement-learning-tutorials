# CLAUDE.md

<!-- Filling this in: CLAUDE-TEMPLATE-NOTES.md, beside this file. Delete the notes when done. -->

Orients a Claude session at the start of every task in this repo, and carries only what is
true for **every** task — depth lives in the skill named below and loads on demand. The other
two owners are [`README.md`](README.md) (a human at a shell: install, run, configuration,
what the tools do) and [`architecture.md`](architecture.md) (whoever is about to change the
code: modules, boundaries, invariants); what was decided and why lives in
[`openspec/ideas.md`](openspec/ideas.md). None of the three is restated here — read the one
you need when you need it.
Per-model advice, when the model changes: [`docs/MODEL-ADVICE.md`](docs/MODEL-ADVICE.md).

## 1. What this project is

Gymnasium reinforcement-learning tutorial scripts and models, adapted for Apple Silicon.

A personal collection of RL demos, one readable file per Gymnasium environment: five show the
raw environment API, two train an agent (a from-scratch DQN and a Stable-Baselines3 PPO) so the
two approaches can be read side by side. It is a leaf — it publishes nothing, and it is a
teaching artefact, not a benchmark: no demo is tuned to a score target.

- **Stack:** python 3.13 (**not 3.14** — `gymnasium[box2d]` has no cp314 wheel; README § Requirements) ·
  **Run:** `uv run rl-tutorials --list` ·
  **Layout and invariants:** `architecture.md` — read it before adding a demo or changing how
  settings resolve; do not re-derive it from the tree, and do not summarise it here.

## 2. Where the data lives — and who owns it

Resolve every path from its variable. Never hard-code one, never infer it from a default, and
if a variable is unset, **stop and ask** — do not guess a location and write there.

| tree | resolve it from | nature |
| --- | --- | --- |
| **code** | the session's repo root | private, on GitHub. **Sole owner** — refactor, rename, delete freely |
| **data** | none — this project declares no data tree and no env var. Model weights are not persisted; a demo trains in-process and exits | — |
| **not ours** | `bin/`, `.claude/`, `.github/`, `.pre-commit-config.yaml` | class M, arrives by broadcast. **Never edit to fix a lint or a bug** — the next delivery 3-way merges, so a local fix becomes a conflict. Report via `/alemax:feedback` |

## 3. The skills this project built

None. This project builds no skills of its own — everything runs through one command,
`uv run rl-tutorials`, documented for humans in `README.md`. The fleet skills (`/alemax:*`,
`/opsx:*`) apply as they do everywhere.

## 4. What this project produces for others

Nothing - a leaf. No package is published, no artefact leaves the repo.

## 5. What this project reads

Only its own tree, plus the class-M set claude-meta broadcasts (see section 2).

## 6. Task routing — everything else

| when you're working on… | invoke |
| --- | --- |
| adding or changing a demo | read `architecture.md` - The demo contract: four exports, one registry line |
| a delivery named in `.local/HANDOFF.md` | `/alemax:complete-update` |

## 7. How we code and spec here — with skills

- `/opsx:propose` → `/opsx:apply` → `/opsx:archive` for anything you would think about for
  more than five minutes before coding. Specs in `openspec/specs/`, in-flight work in
  `openspec/changes/`. This project is its own upstream: changes land here, by PR.
- `/alemax:front-burner` at session start, `/alemax:back-burner` at session end.
- `/alemax:feedback` the moment something bites — a gotcha goes to the skill of the thing that
  bit, or there; **never into this file.**
- A line stays here only while it is true for every task. When it stops being that, move it
  down one level — to the owning skill, `architecture.md`, `README.md` or `openspec/ideas.md` —
  do not delete it.
- Secrets, the dev loop, CI: README § Secrets, § Development.

## Rules of engagement

1. A path comes from its variable; unset means stop and ask.
2. Write only inside what this repo owns (§ 2); outside it, report.
3. Work lands by PR — never a direct push to `main`, even solo.
4. No secret in any tracked file; Keychain holds the values, `.env.example` names the keys.
5. Open questions go to `openspec/ideas.md`; gotchas to `/alemax:feedback`.

Standing constraints from the fleet (claude-meta specs `sibling-access-practice`, `project-environments`) — each names what enforces it:

- A session acts only inside this repo — its root, `.local/`, its worktrees, scratch and toolchain dirs; another repo's checkout or data is reached through that repo's own session, `/alemax:send-msg <drive> <repo>[@env]`, never read or written from here · enforced by: `.claude/hooks/scope-guard.py` (PreToolUse; exit 2 names the address)
- Work in place in your own repo; a write into a sibling repo happens in a worktree of it that you created, never in its primary checkout · enforced by: scope-guard — a sibling's checkout is outside scope; `.claude/worktrees/**` and the `<repo>-claude-meta` delivery worktree are inside
- `.local/env` names this clone's environment, `prod` or `dev`; absent, a clone under `Applications/` is `prod`, anything else `dev`; a `dev` session never writes under a prod clone or the data it declares · enforced by: scope-guard (write-shaped calls under `*/Applications/*` and declared `data:` roots refused while `dev`); `alemax_addr.py self` prints the label
- Two clones of one repo differ only in `.claude/settings.local.json` and `.local/`; everything tracked is identical and reaches both by `git pull`, never by a second delivery · enforced by: none — `bin/reconcile-settings.py check` reports floor drift per clone; a clones-match check must carve those two paths out
- `.local/scope-allow.txt` (one path per line, written by the operator) is the only way scope widens; never loosen a permission rule, a hook or the sandbox to get past a refusal · enforced by: scope-guard reads only that file; `.local/` is gitignored, so a widening never ships
- A corpus, vault or wiki is entered through its schema file, never its `index.md` — an index is a manifest for a tool, not a read path and never an `@import`; and the startup set (this file, every `@import`, every `.claude/rules/*.md` with no `paths:`) stays inside the budget, because a file over 5,000 tokens comes back from a compaction as a path with no content · enforced by: `bin/claude-md-check.py` (pre-commit; refuses the import, reports the budget)
- A prod↔dev channel (`tracking-NN.md`, an inbox) is this project's own file; claude-meta ships none and never writes into it; a brief carries the next action and a path, not the work · enforced by: none — `alemax_addr.py queue` writes only the sender's own `.local/outbox/`
- Data trees are declared in `data.yaml` (`uv run --script bin/data-check.py`); a `dev` session never writes a `prod` one · enforced by: `bin/data-check.py` (pre-commit, staged) · `.claude/hooks/scope-guard.py` (PreToolUse while `dev`)
