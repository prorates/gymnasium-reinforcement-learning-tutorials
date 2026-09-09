## Purpose

Defines how a demo's settings are resolved from defaults, an optional configuration file, and
command-line flags, and how the compute device is chosen — so that a run's behavior is
predictable and a missing setting fails loudly instead of at an arbitrary later moment.

## Requirements

### Requirement: Configuration precedence is defined and total

Settings SHALL resolve in a fixed order: built-in defaults, then values from a configuration
file when one is supplied, then command-line flags. Every setting a demo reads SHALL have a
built-in default, so that running with no configuration file always succeeds.

#### Scenario: No configuration file supplied

- **WHEN** a demo runs with no configuration file
- **THEN** every setting takes its built-in default and the run succeeds

#### Scenario: A configuration file overrides defaults

- **WHEN** a configuration file supplies a setting
- **THEN** that value replaces the built-in default, and settings absent from the file keep
  their defaults

#### Scenario: A flag overrides a configuration file

- **WHEN** both a configuration file and the corresponding command-line flag supply a setting
- **THEN** the flag's value wins

### Requirement: Configuration errors are reported at startup

An unreadable, malformed, or unknown-keyed configuration SHALL be reported before the
environment is constructed, naming the file and the problem.

#### Scenario: Configuration file does not exist

- **WHEN** a user points the tool at a path that does not exist
- **THEN** the tool reports the missing path and exits non-zero, rather than silently falling
  back to defaults

#### Scenario: Configuration file is malformed

- **WHEN** the configuration file is not valid YAML, or its top level is not a mapping
- **THEN** the tool reports the file and the parse problem and exits non-zero

#### Scenario: Unknown configuration key

- **WHEN** the configuration file contains a key the demo does not recognize
- **THEN** the tool reports the unrecognized key and exits non-zero, so a typo cannot be
  silently ignored

#### Scenario: A required setting is never missing at use time

- **WHEN** a demo reads any of its settings
- **THEN** the value is present, because resolution filled every setting from defaults —
  no lookup can raise for a missing key mid-run

### Requirement: Device selection defaults to the fastest device for the workload

The tool SHALL choose a compute device automatically, SHALL allow an explicit override, and
SHALL report which device it selected. The automatic choice SHALL favor CPU for the small
fully-connected networks these demos train, because accelerator transfer overhead dominates
at that size.

#### Scenario: Automatic selection on Apple Silicon

- **WHEN** a training demo runs on an Apple Silicon Mac with no device flag
- **THEN** CPU is selected, and the chosen device is printed

#### Scenario: Explicit override

- **WHEN** a user passes an explicit device
- **THEN** that device is used if available, and the choice is printed

#### Scenario: Requested device unavailable

- **WHEN** a user requests a device that this machine does not provide
- **THEN** the tool reports that the device is unavailable and exits non-zero, rather than
  silently substituting another device

### Requirement: Configuration files ship for the demos that need them

Each demo whose behavior is meaningfully tunable SHALL have a readable configuration file in
the repository demonstrating its settings. A configuration file SHALL NOT be needed to select
which demo runs.

#### Scenario: Shipped configuration is valid

- **WHEN** the test suite loads every configuration file in the repository
- **THEN** each parses, each contains only recognized keys, and each names the demo it
  belongs to

#### Scenario: Configuration does not select the demo

- **WHEN** a configuration file is supplied to a demo
- **THEN** it only supplies settings; it cannot redirect the run to a different demo
