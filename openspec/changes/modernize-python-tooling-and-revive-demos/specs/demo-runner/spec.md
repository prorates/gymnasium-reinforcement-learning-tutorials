## Purpose

Defines the command-line contract for running a tutorial demo — how a demo is chosen, how the
available demos are discovered, which flags shape a run, and what the tool does when asked for
something that does not exist.

## ADDED Requirements

### Requirement: A demo is selected by name as a positional argument

The tool SHALL accept the demo name as its first positional argument. Selecting a demo SHALL
NOT require a configuration file.

#### Scenario: Running a demo with no other arguments

- **WHEN** a user runs the tool with a valid demo name and no other arguments
- **THEN** that demo runs to completion using built-in defaults, and exits zero

#### Scenario: No demo named

- **WHEN** a user runs the tool with no positional argument
- **THEN** the tool prints usage listing the valid demo names and exits non-zero

#### Scenario: Unknown demo named

- **WHEN** a user names a demo that is not in the catalog
- **THEN** the tool prints an error naming the unknown demo, lists the valid names, and exits
  non-zero, without constructing any environment

### Requirement: The catalog is discoverable from the command line

The tool SHALL provide a way to list every available demo without running one.

#### Scenario: Listing demos

- **WHEN** a user passes the list flag
- **THEN** each demo's name, environment id, and one-line description are printed, and the
  tool exits zero without constructing an environment

#### Scenario: Help text names the demos

- **WHEN** a user asks for help
- **THEN** the help output enumerates the valid demo names

### Requirement: Run shape is overridable from the command line

The tool SHALL accept flags overriding episode count, step count per episode, the compute
device, and the rendering mode, without editing any file.

#### Scenario: Overriding episode count

- **WHEN** a user passes an episode-count flag
- **THEN** the demo runs that many episodes, regardless of what the defaults or a
  configuration file specify

#### Scenario: Running without a display

- **WHEN** a user requests no rendering
- **THEN** the demo runs without opening a window, so it can be used over SSH or in CI

#### Scenario: Invalid flag value

- **WHEN** a user passes a non-positive episode or step count
- **THEN** the tool reports the invalid value and exits non-zero before constructing an
  environment

### Requirement: Interrupting a run is clean

The tool SHALL close its environment and exit without a traceback when interrupted.

#### Scenario: User interrupts a running demo

- **WHEN** a user sends an interrupt while a demo is running
- **THEN** the environment is closed, a short message is printed, and the tool exits non-zero
  without printing a Python traceback

### Requirement: The tool is installed as a console entry point

The project SHALL expose the runner as a named console command, runnable without setting
`PYTHONPATH` and without the working directory being the repository root.

#### Scenario: Running from another directory

- **WHEN** a user invokes the console command from a directory other than the repository root
- **THEN** the demo runs normally, because module resolution does not depend on the working
  directory
