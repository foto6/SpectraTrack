# SpectraTrack v0.2 roadmap

## Android / Samsung sensor fusion

1. Camera2 metadata bridge: focal length, zoom ratio, OIS state, ISO/exposure, focus distance/range, lens calibration, camera intrinsics.
2. IMU fusion: gyroscope + accelerometer for roll/pitch/horizon and camera-motion prediction.
3. Geomagnetic sensor: heading/azimuth with calibration quality shown in HUD.
4. GNSS: phone position, speed and course; never fabricate target GPS from a 2D image.
5. Barometer: relative altitude/pressure trend.
6. Laser-AF/focus experiment: expose only Android-reported focus distance/range and calibration quality. Label as FOCUS RANGE unless a true ranging API is available.
7. UWB ranging mode: distance/azimuth only to a paired compatible UWB peer or anchor.
8. Physical camera selector: expose available physical lenses and per-lens intrinsics.
9. On-demand target enhancement: non-generative denoise/deblur first; neural SR only as an explicitly marked enhanced view.
10. Device capability screen: enumerate Camera2 characteristics and Android sensors so unsupported features stay disabled instead of faked.

## PC

Highest-priority next milestone: **high-recall small-pedestrian detection for high-angle/night/compressed footage**.
See [PC_V03_PLAN.md](PC_V03_PLAN.md). The implementation path is person-only high-resolution/tiled inference,
class-specific thresholds, temporal confirmation/recovery, benchmark-driven model selection, and explicit
offline quality-max vs live performance modes.

1. Upgrade tracker to BoT-SORT/ByteTrack-style association with optional ReID.
2. Camera-motion compensation before association.
3. Optional SAM-style target segmentation for precise mask lock.
4. Vulkan/DirectML enhancement pipeline: denoise, deblur, low-light, dehaze, SR snapshots.
5. Multi-camera mode with calibrated stereo depth when two cameras are available.
6. Calibration module: intrinsics, distortion, FOV, ground-plane calibration.
7. Honest kinematics: image velocity always available; metric speed/range only when calibration/depth supports it.
8. Record/replay sessions with JSON metadata sidecar.
9. Model manager with SHA-256 verification and explicit model provenance.
10. Benchmark overlay: inference ms, tracker ms, dropped frames and active provider.

## HUD data labels

- MEASURED: direct hardware sensor data.
- CALIBRATED: derived from calibrated sensor/camera metadata.
- ESTIMATED: geometry/model estimate.
- AI-ENHANCED: neural reconstruction that may invent fine detail.

Never display fabricated range, thermal readings, target GPS coordinates or metric speed.
