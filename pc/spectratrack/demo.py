from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .calibration import CameraGeometry
from .cmc import MotionEstimate
from .hud import compose_hud
from .tracker import MultiObjectTracker
from .types import Detection


def main() -> int:
    tracker = MultiObjectTracker(max_missed=8)
    selected = None
    fps = 60.0
    t0 = time.perf_counter()
    frame_i = 0
    cv2.namedWindow("SpectraTrack demo", cv2.WINDOW_NORMAL)
    while True:
        frame = np.zeros((720, 1080, 3), dtype=np.uint8)
        t = time.perf_counter() - t0
        detections = []
        specs = [
            ("car", 2, 0.93, 220 + 120 * math.sin(t * 0.9), 240 + 50 * math.cos(t * 0.6), 170, 90),
            ("person", 0, 0.88, 650 + 160 * math.sin(t * 0.5 + 1.2), 380 + 110 * math.cos(t * 0.8), 75, 170),
            ("airplane", 4, 0.84, 510 + 300 * math.sin(t * 0.32), 115 + 38 * math.cos(t * 0.4), 210, 70),
        ]
        for label, cid, score, cx, cy, bw, bh in specs:
            if not (label == "person" and frame_i % 120 in range(78, 86)):
                detections.append(Detection((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), score, cid, label))

        tracks = tracker.update(detections)
        confirmed = [tr for tr in tracks if tr.confirmed]
        if selected is None and confirmed:
            selected = confirmed[0].track_id
        output = compose_hud(
            frame,
            tracks,
            selected,
            fps,
            "DEMO/SYNTHETIC",
            "off",
            metrics={"detect_avg_ms": 0.0, "track_avg_ms": 0.2},
            motion=MotionEstimate.identity(),
            geometry=CameraGeometry(70.0),
            cmc_enabled=False,
            stabilization_enabled=False,
        )
        cv2.imshow("SpectraTrack demo", output)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord("q")):
            break
        frame_i += 1
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
