## Purpose

Defines the set of runnable tutorial demos this collection offers — the name each answers to,
the environment it drives, and the guarantee that every listed demo actually starts on Apple
Silicon rather than failing on a missing dependency or a retired environment id.

## ADDED Requirements

### Requirement: Every catalogued demo is runnable

Every demo the catalog advertises SHALL construct its environment, reset it, and take at
least one step without raising, on macOS/arm64, using only the project's declared
dependencies. A demo that cannot meet this MUST NOT appear in the catalog.

#### Scenario: Each advertised demo starts

- **WHEN** the test suite iterates every demo in the catalog and constructs, resets, and
  steps its environment
- **THEN** every one succeeds, and no demo is skipped for a missing optional dependency

#### Scenario: A demo whose dependency is absent

- **WHEN** a demo's environment cannot be constructed because a required dependency is not
  installed
- **THEN** the failure surfaces as a test failure naming the demo and the missing dependency,
  rather than at the moment a user tries to run it

### Requirement: Demo names describe behavior

Each demo SHALL be identified by a stable kebab-case name describing what it does. Names
SHALL NOT be positional or ordinal.

#### Scenario: Selecting a demo by name

- **WHEN** a user names a demo on the command line
- **THEN** that demo runs, and the name identifies the demo independently of any
  configuration file's contents

#### Scenario: Ordinal names are gone

- **WHEN** a user supplies a legacy ordinal identifier such as `model3` or `tutorial1`
- **THEN** the command fails with an error listing the valid demo names

### Requirement: The catalog covers the historical demos plus a library baseline

The catalog SHALL contain the six demos this collection has always carried, plus one demo
that trains an agent using an off-the-shelf RL library:

| name | environment | what it shows |
| --- | --- | --- |
| `random-cartpole` | `CartPole-v1` | the raw environment API, observation and action spaces |
| `cartpole` | `CartPole-v1` | a minimal reset/step loop |
| `dqn-cartpole` | `CartPole-v1` | a DQN implemented from scratch |
| `lunarlander` | `LunarLander-v3` | a Box2D environment under random actions |
| `bipedalwalker` | `BipedalWalker-v3` | a continuous-action Box2D environment |
| `mspacman` | `ALE/MsPacman-v5` | an image-observation Atari environment |
| `ppo-lunarlander` | `LunarLander-v3` | the same task as `lunarlander`, solved by a library agent |

#### Scenario: Listing the catalog

- **WHEN** a user asks the tool to list demos
- **THEN** all seven names are shown, each with its environment id and a one-line description

#### Scenario: The from-scratch and library agents are comparable

- **WHEN** a reader compares `dqn-cartpole` with `ppo-lunarlander`
- **THEN** both are present in the catalog as first-class demos, so the hand-written
  implementation and the library implementation can be read side by side

### Requirement: Retired environment ids are not used

The catalog SHALL reference only environment ids that the pinned Gymnasium version
registers. An environment requiring namespace registration SHALL be registered before use.

#### Scenario: Atari namespace registration

- **WHEN** the Ms. Pac-Man demo is run
- **THEN** the ALE namespace is registered first, and `ALE/MsPacman-v5` is constructed
  successfully

#### Scenario: Step returns the five-tuple

- **WHEN** any demo steps its environment
- **THEN** it unpacks the result as `(observation, reward, terminated, truncated, info)`, and
  treats termination and truncation as distinct conditions
