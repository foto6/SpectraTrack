from __future__ import annotations

import argparse
import cv2
import numpy as np

from .stereo import StereoDepthEstimator, StereoRigCalibration


def colorize_disparity(disparity: np.ndarray) -> np.ndarray:
    valid = disparity[np.isfinite(disparity) & (disparity > 0)]
    if valid.size == 0:
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [5, 95])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((disparity - lo) / (hi - lo), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0)
    image = (scaled * 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect calibrated rectified stereo video/image pairs")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", default="", help="Optional disparity visualization video/image")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args()

    cal = StereoRigCalibration.from_json(args.calibration)
    est = StereoDepthEstimator(cal)

    left_img = cv2.imread(args.left)
    right_img = cv2.imread(args.right)
    if left_img is not None and right_img is not None:
        disparity = est.compute_disparity(left_img, right_img)
        valid = disparity[np.isfinite(disparity)]
        fraction = valid.size / max(disparity.size, 1)
        print(f"valid_fraction={fraction:.4f}")
        if valid.size:
            print(f"median_disparity_px={float(np.median(valid)):.3f}")
            print(f"median_depth_m={cal.distance_from_disparity(float(np.median(valid))):.3f}")
        if args.output:
            cv2.imwrite(args.output, colorize_disparity(disparity))
        return 0

    left_cap = cv2.VideoCapture(args.left)
    right_cap = cv2.VideoCapture(args.right)
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise SystemExit("Could not open left/right as image pair or video pair")

    writer = None
    frame_i = 0
    valid_fractions: list[float] = []
    medians: list[float] = []
    try:
        while True:
            ok_l, left = left_cap.read()
            ok_r, right = right_cap.read()
            if not ok_l or not ok_r:
                break
            frame_i += 1
            disparity = est.compute_disparity(left, right)
            valid = disparity[np.isfinite(disparity)]
            valid_fractions.append(valid.size / max(disparity.size, 1))
            if valid.size:
                medians.append(float(np.median(valid)))

            if args.output:
                vis = colorize_disparity(disparity)
                if writer is None:
                    fps = left_cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (vis.shape[1], vis.shape[0])
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create output: {args.output}")
                writer.write(vis)

            if args.max_frames > 0 and frame_i >= args.max_frames:
                break
    finally:
        left_cap.release()
        right_cap.release()
        if writer is not None:
            writer.release()

    if frame_i == 0:
        raise SystemExit("No paired frames read")
    print(f"frames={frame_i}")
    print(f"avg_valid_fraction={float(np.mean(valid_fractions)):.4f}")
    if medians:
        median_disp = float(np.median(medians))
        print(f"median_disparity_px={median_disp:.3f}")
        print(f"median_depth_m={cal.distance_from_disparity(median_disp):.3f}")
    else:
        print("median_depth_m=unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
