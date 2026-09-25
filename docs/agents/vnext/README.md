# vNext Agent Coordination

This directory defines the research-only specialist contract for SpectraTrack vNext.

Immutable integrated baseline:

`integration @ 3ebc4d50213593cac62b97399447fccf6bbc1755`

Coordination branch:

`vnext-base`

The exact `vnext-base` HEAD is supplied in the architect's `vNext Coordination Handoff`. Every specialist must branch from that exact SHA, not from a moving branch tip guessed later.

Branches:

- A1: `agent/vnext-detection`
- A2: `agent/vnext-tracking`
- A3: `agent/vnext-enhancement`
- A4: `agent/vnext-performance`
- A5: `agent/vnext-qa`

## Startup procedure

Before any implementation:

1. `git fetch origin`
2. verify the architect-supplied `vnext-base` SHA;
3. read:
   - `AGENTS.md`
   - `ARCHITECTURE.md`
   - `TASKS.md`
   - `DECISIONS.md`
   - `docs/VNEXT_RESEARCH_PLAN.md`
   - this file
   - your role file
4. inspect the existing implementation in your ownership area;
5. search for existing helpers/harnesses before creating new abstractions;
6. record the exact starting SHA in experiment metadata.

## Shared research rules

Research only. Do not change production behavior merely to test a hypothesis.

Each experiment must follow:

`baseline -> isolated candidate -> same data -> metrics -> cost -> decision`

Do not merge specialist branches together.

Do not modify:

- `integration`
- `main`

Do not release.

Do not commit large model/video binaries.

Prefer generated experiment material under:

`pc/benchmarks/vnext/<role>/`

Use the canonical replay schema in `docs/VNEXT_RESEARCH_PLAN.md` whenever tracker/detector decoupling is relevant.

## Required handoff content

Every specialist handoff must state:

- exact branch HEAD SHA;
- exact parent/start SHA;
- files changed;
- experiments run;
- input/corpus revision;
- model/backend IDs and hashes where applicable;
- CPU/DirectML provider;
- wall time;
- actual ONNX inference calls;
- quality metrics where GT exists;
- regressions / failures;
- licenses/dependencies introduced or evaluated;
- what remains unverified;
- whether evidence supports: reject / continue research / candidate for later integrator review.

"Looks better" is not a result.

## Synchronization

No one waits for A5 to begin.

Before real GT:

- A1: fusion + dump tooling;
- A2: replay + current tracker diagnostics;
- A3: operation-level enhancement profiling;
- A4: compute model + scheduler simulation;
- A5: corpus + annotation workflow.

After A5 freezes a usable corpus revision, all roles rerun on the same revision. A5 then assembles the shared comparison table.

Production integration is deferred to a later:

`agent/vnext-integrator`
