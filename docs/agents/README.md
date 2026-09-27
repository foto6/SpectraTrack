# Parallel agent handoff files

This directory is the per-role working memory for parallel SpectraTrack development.

## Rules

- `AGENTS.md` remains the canonical project-wide rulebook.
- `ARCHITECTURE.md` describes the shared architecture.
- `DECISIONS.md` records genuine shared architectural decisions.
- `TASKS.md` is the shared project ledger.
- Files in `docs/agents/` are role-specific handoff/state files.

Each agent owns exactly one role file:

- `detection.md` -> `agent/detection`
- `tracking.md` -> `agent/tracking`
- `enhancement.md` -> `agent/enhancement`
- `performance.md` -> `agent/performance`
- `qa.md` -> `agent/qa`

Do not edit another agent's role file.

Use the role file to record current implementation state, tests, measurements, limitations, branch/head SHA, cross-agent conflicts, and handoff notes.

Do not copy large logs into these files. Record facts and commands/results concisely.

Do not update shared root docs for routine progress. Update shared docs only when the shared architecture, contract, or task state genuinely changes.

## Required update before handoff

Before declaring a branch ready for integration, update your role file with:

1. current base branch and original base commit;
2. current branch and HEAD SHA;
3. implemented changes;
4. files changed;
5. tests actually run and their results;
6. benchmark before/after data, if real data exists;
7. known limitations and unverified claims;
8. overlaps/conflicts with other agent branches;
9. exact commits that belong to your work;
10. readiness for integration.

Never fabricate benchmark numbers or claim tests were run when they were not.
