from __future__ import annotations

import numpy as np

from .calibration import CameraCalibration
from .enhance import DISPLAY_MODES, apply_display_mode
from .metrics import StageTimer
from .tracker import MultiObjectTracker
from .types import Detection


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    print("SpectraTrack self-check")

    tracker = MultiObjectTracker(min_hits=1)
    first = Detection((100, 100, 200, 220), 0.95, 0, "person")
    tr = tracker.update([first])[0]
    check(tr.track_id == 1, "tracker failed initial identity")

    affine = (1.0, 0.0, 25.0, 0.0, 1.0, -10.0)
    moved = Detection((125, 90, 225, 210), 0.92, 0, "person")
    tr = tracker.update([moved], camera_transform=affine)[0]
    check(abs(tr.vx) < 1e-6 and abs(tr.vy) < 1e-6, "affine CMC residual is wrong")

    tr = tracker.predict_only(camera_motion=(2.0, 0.0))[0]
    check(tr.missed == 0, "scheduled prediction incorrectly counted as missed")

    cal = CameraCalibration(1920, 1080, 80.0)
    yaw, pitch = cal.angular_offset_deg(960, 540)
    check(abs(yaw) < 1e-9 and abs(pitch) < 1e-9, "calibration center is not zero")

    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    frame[:, 80:] = 160
    for mode in DISPLAY_MODES:
        out = apply_display_mode(frame, mode)
        check(out.shape == frame.shape and out.dtype == np.uint8, f"display mode failed: {mode}")

    timer = StageTimer()
    timer.add("x", 1.0)
    timer.add("x", 3.0)
    check(timer.average("x") == 2.0, "metrics average failed")

    print("tracker=ok")
    print("affine_cmc=ok")
    print("prediction_only=ok")
    print("calibration=ok")
    print("display_modes=ok")
    print("metrics=ok")
    print("SELF_CHECK=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
