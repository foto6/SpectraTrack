from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .camera_calibration import calibrate_checkerboard


def main() -> int:
    p = argparse.ArgumentParser(description="Calibrate a camera from checkerboard photos")
    p.add_argument("--images", required=True, help="Glob such as calib/*.jpg")
    p.add_argument("--cols", type=int, required=True, help="Checkerboard inner corners across")
    p.add_argument("--rows", type=int, required=True, help="Checkerboard inner corners down")
    p.add_argument("--square", type=float, default=1.0, help="Square size in any consistent unit")
    p.add_argument("--min-images", type=int, default=5)
    p.add_argument("--output", default="camera-calibration.json")
    args = p.parse_args()

    paths = sorted(glob.glob(args.images, recursive=True))
    if not paths:
        raise SystemExit(f"No images matched: {args.images}")
    try:
        calibration = calibrate_checkerboard(paths, args.cols, args.rows, args.square, args.min_images)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    calibration.save(args.output)
    print(f"saved: {Path(args.output).resolve()}")
    print(f"used images: {calibration.used_images}")
    print(f"RMS: {calibration.rms:.5f}")
    print(f"mean reprojection error: {calibration.mean_reprojection_error_px:.5f} px")
    print(f"HFOV: {calibration.hfov_deg:.3f} deg")
    print(f"VFOV: {calibration.vfov_deg:.3f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
