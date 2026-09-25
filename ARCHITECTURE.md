# ARCHITECTURE.md

This is the canonical architecture map for the current repository state.

Baseline when this document was prepared:

- repository: `foto6/SpectraTrack`
- development baseline: `main` at `2012eaae2f4ffe820a66d12e40346d911616cd03`
- latest published release: `v0.3.0`, release target `141aa96595ce18a29fe4fbf665c4affd1dbab4de`
- active documentation/setup branch: `agent/project-setup`

## 1. Repository layout

```text
pc/             Windows/Python application, batch analysis, tests, presets
android/        Native Kotlin/CameraX Android client
models/         model format/provenance documentation; weights are not committed
tools/          model export and hashing helpers
docs/           detailed plans, testing notes, cross-video docs
.github/        PC CI, Android CI, versioned release workflows
```

The root files `AGENTS.md`, `ARCHITECTURE.md`, `TASKS.md`, and `DECISIONS.md` are the shared memory for parallel development.

## 2. PC runtime pipeline

The main entry point is `pc/run_spectratrack.py`.

- normal invocation dispatches to `spectratrack.app.main()`;
- first positional token `batch` dispatches to `spectratrack.batch.main()`.

The live/single-video pipeline in `pc/spectratrack/app.py::main()` is:

```text
CLI / runtime config
        ↓
model path + hash / manifest verification
        ↓
RobustCapture
        ↓
raw BGR frame
        ├── GlobalMotionEstimator (CMC, on raw frame)
        ├── optional VideoStabilizer
        ↓
working frame
        ├── optional enhance_visibility() -> analysis frame
        └── apply_display_mode()       -> display frame
        ↓
detector cadence gate
        ├── detector frame: YoloOnnxDetector.detect()
        │       ↓
        │   class filter
        │       ↓
        │   optional attach_appearance()
        │       ↓
        │   MultiObjectTracker.update()
        └── skipped detector frame
                ↓
            MultiObjectTracker.predict_only()
        ↓
selected-target LockRefiner
        ↓
compose_hud()
        ├── optional video writer
        ├── optional SessionRecorder JSONL
        └── OpenCV window / controls
```

### Important coupling rule

When optical stabilization is enabled, the tracker does **not** also apply CMC to the stabilized frame. `app.py` deliberately zeros `camera_shift` / `camera_transform` in that case to avoid double compensation.

## 3. Video input and decoding

File: `pc/spectratrack/capture.py`.

Key interfaces:

- `CaptureConfig`
- `CaptureRead`
- `RobustCapture`
- `RobustCapture._open()`
- `RobustCapture.read()`
- `RobustCapture.actual_properties()`
- `RobustCapture.stats()`

OpenCV `cv2.VideoCapture` performs decoding.

A source is considered a live camera only when the source is an integer index. Camera-only backends such as DirectShow/MSMF are not forced onto ordinary video files.

Camera reconnects are bounded. File EOF does not trigger reconnect loops.

Default capture config:

- backend: `auto`
- buffer size: `1`
- reconnect attempts: `3`
- reconnect delay: `250 ms`

## 4. Model loading and detector

Files:

- `pc/spectratrack/detector.py`
- `pc/spectratrack/model_manifest.py`
- `pc/spectratrack/integrity.py`
- `models/README.md`
- `tools/export_yolo.py`

Primary class: `YoloOnnxDetector`.

Important methods/functions:

- `YoloOnnxDetector.__init__()`
- `YoloOnnxDetector._letterbox()`
- `YoloOnnxDetector.detect()`
- `decode_end2end_predictions()`
- `_classwise_nms()`

### Runtime provider selection

On Windows the detector prefers `DmlExecutionProvider` when ONNX Runtime exposes it, then adds `CPUExecutionProvider` as fallback. `--cpu` disables GPU preference.

DirectML uses sequential ORT execution and disables memory patterns.

### Supported output conventions

The PC detector intentionally supports only known layouts:

- raw YOLO-style `[1,84,N]` or `[1,N,84]` style `xywh + class scores`;
- common end-to-end/post-NMS `xyxy + score + class` rows.

Segmentation, pose/keypoints, unknown objectness layouts, and arbitrary multi-output models are not silently guessed.

### Preprocessing

`YoloOnnxDetector._letterbox()`:

- resizes while preserving aspect ratio;
- fills unused area with value `114`;
- detector input is RGB, float32, normalized by `255`, CHW with batch dimension.

Default CLI detector settings in `app.py`:

- input size: `640`
- confidence: `0.35`
- IoU/NMS threshold: `0.45`

A fixed-size compatible YOLO ONNX model is expected. A model manifest can provide input size and SHA-256/provenance verification.

Model weights are not included in Git or silently downloaded by the runtime.

### Analysis enhancement vs display enhancement

`pc/spectratrack/enhance.py` intentionally separates detector analysis from operator display.

- `--enhance` runs `enhance_visibility()` on the detector analysis frame.
- `--view` runs `apply_display_mode()` on the display frame.
- Display mode changes do not automatically modify the detector input.

Available display modes:

- `normal`
- `clarity`
- `lowlight`
- `edges`
- `pseudo-thermal` — false-color luminance only

## 5. Detection cadence / performance profiles

Defined in `pc/spectratrack/app.py`:

- `quality`: detector every `1` frame
- `balanced`: detector every `2` frames
- `speed`: detector every `3` frames

`--detect-every N` overrides the profile cadence when `N > 0`.

Presets:

### `pc/presets/amd-quality.json`

- requested capture: 1920×1080 @ 30
- backend: DirectShow
- profile: quality
- detect_every: 1
- CMC on
- appearance on
- stabilization off
- analysis enhancement off

### `pc/presets/amd-balanced.json`

Same requested capture, but `balanced` / `detect_every: 2`.

### `pc/presets/cpu-safe.json`

- requested capture: 1280×720 @ 30
- CPU inference
- `speed`
- detect_every: 3

A manual stress configuration used during development was YOLO11x ONNX, input size 960, confidence 0.20, IoU 0.50, quality profile, detector every frame. This is **not** the repository default.

## 6. Tracking

Files:

- `pc/spectratrack/tracker.py`
- `pc/spectratrack/types.py`
- `pc/spectratrack/appearance.py`

Primary class: `MultiObjectTracker`.

Important methods:

- `MultiObjectTracker.update()`
- `MultiObjectTracker.predict_only()`
- `MultiObjectTracker._associate()`
- `MultiObjectTracker._candidate_score()`
- `MultiObjectTracker._apply_match()`
- `MultiObjectTracker._predicted_box()`

Current default tracker parameters:

- `max_missed = 14`
- `min_iou = 0.10`
- `max_center_ratio = 1.9`
- `high_conf = 0.45`
- `low_conf = 0.12`
- `min_hits = 3`

Behavior:

- same-class association only;
- predicted box uses CMC plus residual target velocity;
- first association stage prioritizes high-confidence detections;
- second looser stage can use lower-confidence detections to maintain tracks;
- only strong/high-confidence detections may create new IDs;
- detector-skipped frames call `predict_only()` and do not increment `missed`;
- confirmed tracks can survive short detector dropouts;
- track velocity is an exponentially smoothed residual image velocity, not metric world velocity;
- track history stores up to 64 center points.

Current association is greedy over candidate scores. It is **not** Hungarian assignment, Kalman-filter tracking, ByteTrack, or BoT-SORT.

### Appearance cue

`appearance.py::appearance_descriptor()` is a local HSV histogram cue used only to help same-class association. It is non-biometric.

## 7. Camera-motion compensation (CMC)

File: `pc/spectratrack/motion.py`.

Class: `GlobalMotionEstimator`.

Method: `GlobalMotionEstimator.update()`.

Pipeline:

1. grayscale conversion;
2. Shi-Tomasi corners via `goodFeaturesToTrack`;
3. pyramidal Lucas-Kanade optical flow;
4. `estimateAffinePartial2D` with RANSAC;
5. quality gates before passing affine motion to the tracker.

Current gates:

- at least 6 inliers after fitting;
- inlier ratio >= 0.35;
- absolute translation <= 75% of frame width/height;
- scale in `[0.78, 1.28]`;
- absolute rotation <= 35°.

The result is image-space global motion, not physical camera pose.

## 8. Selected-target lock refinement

File: `pc/spectratrack/lock_refine.py`.

Class: `LockRefiner`.

Methods:

- `initialize()`
- `update()`
- `reset()`

It tracks sparse points inside the explicitly selected target using LK optical flow and affine RANSAC.

Current refinement gates:

- >= 6 inliers;
- inlier ratio >= 0.45;
- scale in `[0.85, 1.18]`;
- absolute rotation <= 20°.

`app.py` re-anchors the refiner whenever a fresh detector observation exists. Flow refinement is used on detector-skipped frames or short detector misses.

Important current limitation: the refined box is copied into `display_tracks` for HUD/target-view rendering. It does **not** currently overwrite the underlying `MultiObjectTracker` state or the JSONL track bbox.

## 9. Stabilization

File: `pc/spectratrack/stabilize.py`.

Class: `VideoStabilizer`.

This is a separate operator/image stabilization path. Toggling stabilization resets stabilizer, CMC, tracker, lock refiner, and selection.

## 10. HUD / UI

File: `pc/spectratrack/hud.py`.

Main function: `compose_hud()`.

Other relevant functions:

- `draw_tracks()`
- `draw_corner_box()`
- `draw_center_reticle()`
- `_calibrated_lines()`

The UI is currently OpenCV-window based on PC.

HUD includes:

- track IDs and boxes;
- lifecycle/quality/recovery state;
- target crop panel;
- detector/provider/FPS info;
- per-stage timings;
- CMC image motion;
- capture reconnect/failure stats;
- selected lock-flow status;
- calibrated angular data when calibration exists.

Controls in `app.py`:

- left click: select/unselect target
- `E`: detector analysis enhancement
- `M`: display mode
- `Z`: stabilization
- `R`: tracker reset
- `H`: HUD
- `S`: source crop snapshot
- `U`: optional external Real-ESRGAN x4 snapshot
- `Q` / Esc: quit

## 11. Logging and provenance

File: `pc/spectratrack/session.py`.

Class: `SessionRecorder`.

Methods:

- `frame()`
- `event()`
- `close()`

JSONL stores:

- header metadata including model hash/providers/source/profile/config;
- per-frame track state;
- selected ID;
- CMC motion;
- selected lock refinement status;
- stage timing;
- lifecycle/recovery events;
- footer duration.

Related utilities:

- `session_report.py`
- `session_csv.py`
- `session_overlay.py`
- `snapshot_meta.py`

Snapshots store provenance sidecars. AI-upscaled outputs are classified separately as `AI_ENHANCED`.

### Recording

`app.py` records the rendered output through OpenCV `VideoWriter` using `mp4v`.

For file inputs, output FPS is taken from source capture FPS when available. Audio is not preserved by this OpenCV recording path.

## 12. Cross-video batch analysis

Files:

- `pc/spectratrack/batch.py`
- `pc/spectratrack/crossvideo.py`
- `pc/spectratrack/crossvideo_report.py`
- `pc/spectratrack/appearance.py`

Entry point: `SpectraTrack-PC.exe batch ...` → `batch.main()`.

Important functions:

- `discover_videos()`
- `video_identifier()`
- `analyze_video()`
- `crossvideo_descriptor()`
- `build_cross_video_graph()`
- `tracklet_similarity()`
- `write_html_report()`

Batch defaults:

- input size: 640
- confidence: 0.35
- IoU: 0.45
- detector every frame
- candidate similarity threshold: 0.86
- strong threshold: 0.94
- minimum retained observations: 3
- progress print every 120 frames

Batch tracking uses CMC. Generated SpectraTrack outputs are skipped by default to avoid recursive self-analysis.

The current cross-video descriptor is model-free:

- 2×2 spatial HSV histograms;
- grayscale histogram;
- edge-orientation histogram;
- up to six diverse tracklet gallery entries.

Grouping is conservative complete-link clustering. Manual `SAME`, `DIFFERENT`, and `UNSURE` decisions can be exported from the HTML report and re-applied.

For `person`, graph semantics are explicitly `same_appearance_candidate`, not biometric identity.

## 13. Calibration

Files:

- `pc/spectratrack/calibration.py`
- `pc/spectratrack/calibrate_fov.py`

Calibration can provide camera dimensions and FOV for angular offset/size display.

SpectraTrack does not fabricate monocular metric target range.

## 14. Metrics / benchmark / diagnostics

Files:

- `pc/spectratrack/metrics.py`
- `pc/spectratrack/benchmark.py`
- `pc/spectratrack/qa_benchmark.py`
- `pc/spectratrack/diagnostics.py`
- `pc/spectratrack/selfcheck.py`
- `pc/benchmarks/README.md`

`StageTimer` records recent stage timings.

A synthetic tracker benchmark exists and CI runs:

```text
python -m spectratrack.benchmark --frames 500 --targets 24
```

`qa_benchmark.py` is the detector/tracker quality harness. It reads lightweight JSONL ground truth, runs the current detector + affine CMC + appearance cue + tracker on local video, stores model/ground-truth hashes and settings, and reports person recall/precision, false positives/negatives, apparent-size/frame-tag/object-attribute breakdowns, ID switches, fragmentation, FPS, and optional externally measured VRAM. Its comparator rejects mismatched evaluation inputs and explicitly lists `NEW FALSE NEGATIVE` regressions.

Important remaining gap: the harness exists, but there is currently **no collected and validated representative real CCTV corpus** for small-person recall in poor high-angle/night/compressed footage. Synthetic fixtures test the evaluator only and are not quality evidence. Collecting the real annotated/held-out set remains a top-priority task in `docs/PC_V03_PLAN.md`.

## 15. Android architecture

Primary files:

- `android/app/src/main/java/com/foto6/spectratrack/MainActivity.kt`
- `OnnxDetector.kt`
- `LiteTracker.kt`
- `HudOverlayView.kt`
- `Models.kt`

Android uses:

- CameraX preview and ImageAnalysis;
- ONNX Runtime Android 1.30.0;
- NNAPI attempt with CPU fallback;
- local lightweight tracking;
- tap-to-lock HUD.

Android version metadata is `versionName 0.3.0`, `versionCode 3`.

The PC v0.3 cross-video graph, PC CMC implementation, and PC lock-refiner are not shared implementations with Android.

## 16. Important dependency relationships

- detector output must remain compatible with `types.Detection` and tracker input;
- tracker expects CMC affine in a 6-float 2×3 representation;
- `appearance_descriptor()` populates `Detection.appearance`;
- `SessionRecorder` depends on `Track` fields and lifecycle semantics;
- HUD consumes `Track`, calibration, timings, CMC and lock-refine summaries;
- batch reuses detector, tracker, CMC and appearance infrastructure but has a separate output graph/report path;
- model manifest/input size must agree with the actual fixed-size ONNX model;
- stabilization and CMC must not both compensate the same motion in the tracking path.

Any change to these contracts should update `DECISIONS.md`, tests, and this document.
