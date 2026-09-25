# AGENTS.md

This repository is the shared codebase **and** shared project memory for parallel AI-agent development.

## Read first

Before editing code, read in this order:

1. `README.md`
2. `AGENTS.md`
3. `ARCHITECTURE.md`
4. `TASKS.md`
5. `DECISIONS.md`
6. the relevant tests and module files for your assigned area

Do not rely on chat history as the authoritative project state. The repository is authoritative.

## Branch discipline

Do not commit directly to `main`.

The intended topology after this project-setup branch is reviewed and merged is:

```text
main
 ↑
integration
 ↑
├── agent/detection
├── agent/tracking
├── agent/enhancement
├── agent/performance
└── agent/qa
```

Each agent should create one role branch from the current integration baseline:

- `agent/detection`
- `agent/tracking`
- `agent/enhancement`
- `agent/performance`
- `agent/qa`

If `integration` does not exist yet, coordinate with the owner before creating independent long-lived work from stale `main`.

## Coordination rules

- Work autonomously until there is a finished useful result in your area; do not stop after the first small fix.
- Keep scope narrow enough that another agent can review it.
- Before changing a shared type, CLI contract, serialized schema, session format, detector output contract, or shared config field, record the intended change in `DECISIONS.md`.
- Update `TASKS.md` when a task changes state or a new bug/debt item is discovered.
- Update `ARCHITECTURE.md` when data flow, module ownership, or public interfaces change.
- Commit code and its corresponding documentation/tests together when practical.
- Do not use documentation-only changes as an excuse to rewrite working code.
- Do not delete another agent's work merely because you would design it differently.
- Prefer additive/refactoring changes that preserve existing CLI and tests unless the task explicitly requires a breaking change.
- If your branch touches files another role is likely to edit, call that out in the commit/PR summary.

## Required validation

For PC changes, from `pc/`:

```powershell
ruff check spectratrack tests
python -m compileall -q spectratrack tests
pytest -q
python -m spectratrack.benchmark --frames 500 --targets 24
python -m spectratrack.diagnostics
python -m spectratrack.selfcheck
```

For packaging-sensitive PC changes, also verify the PyInstaller standalone build and both CLI smoke tests:

```text
SpectraTrack-PC.exe --help
SpectraTrack-PC.exe batch --help
```

For Android changes, build `:app:assembleDebug` with the repository's configured Java/Gradle/SDK versions.

Do not claim a test passed if you did not run it or CI did not run it.

## Scope ownership guide

### Detection agent

Primary files:

- `pc/spectratrack/detector.py`
- `pc/spectratrack/app.py` detector call path
- `pc/spectratrack/batch.py` batch detector call path
- `pc/spectratrack/enhance.py` only where analysis preprocessing is relevant
- `pc/tests/test_detector_decode.py`
- future person-recall benchmark/validation files

Current priority: small-person recall in poor high-angle/night/compressed video. Read `docs/detection.md` and `docs/PC_V03_PLAN.md` before changing thresholds, tiling, merge strategy, or model choice.

### Tracking agent

Primary files:

- `pc/spectratrack/tracker.py`
- `pc/spectratrack/types.py`
- `pc/spectratrack/motion.py`
- `pc/spectratrack/lock_refine.py`
- `pc/spectratrack/appearance.py`
- tracking/motion/lock-refine tests

Do not silently turn appearance cues into biometric/person identity logic.

### Enhancement agent

Primary files:

- `pc/spectratrack/enhance.py`
- `pc/spectratrack/stabilize.py`
- analysis/display split in `pc/spectratrack/app.py`
- relevant visual tests

Generative SR is not trusted as ground-truth evidence for detection. Pseudo-thermal is false-color only.

### Performance agent

Primary files:

- `pc/spectratrack/metrics.py`
- `pc/spectratrack/capture.py`
- `pc/spectratrack/runtime_config.py`
- `pc/presets/*.json`
- provider/model initialization in `pc/spectratrack/detector.py`
- CI/build workflows where needed

Preserve DirectML preference on Windows with CPU fallback.

### QA agent

Primary files:

- `pc/tests/`
- `docs/TESTING.md`
- benchmark/diagnostics/self-check modules
- release/build smoke checks

Highest-value missing QA asset: a representative annotated pedestrian-recall validation set and benchmark for poor CCTV footage.

## Non-negotiable project semantics

- Runtime is local-only; do not add telemetry, cloud inference, or hidden network access.
- No face recognition, biometric templates, named-person identification, or covert capture.
- Cross-video person links mean similar visible appearance in that batch only, not identity.
- Do not fabricate thermal measurements, target GPS, monocular metric range, or metric speed without suitable calibrated hardware.
- `pseudo-thermal` is only an RGB false-color display mode.
- AI super-resolution may invent detail and must remain explicitly marked as AI-enhanced.
- Neural weights are not silently auto-downloaded by the runtime; model provenance and SHA-256 support are intentional.
- Do not repurpose the project for weapons/fire-control integration.

## When changing shared interfaces

Examples of shared interfaces that require explicit coordination:

- `Detection` and `Track` in `pc/spectratrack/types.py`
- detector output conventions
- tracker `update()` / `predict_only()` contracts
- CMC affine representation
- JSONL session schema
- cross-video graph schema
- runtime-config keys
- CLI argument names/defaults
- Android/PC model compatibility assumptions

Record the decision in `DECISIONS.md`, add regression tests, then implement.
