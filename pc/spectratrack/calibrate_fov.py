from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def estimate_hfov_deg(image_width_px: float, object_width_px: float, object_width_m: float, distance_m: float) -> float:
    if min(image_width_px, object_width_px, object_width_m, distance_m) <= 0:
        raise ValueError("all inputs must be positive")
    fx_px = object_width_px * distance_m / object_width_m
    return math.degrees(2.0 * math.atan(image_width_px / (2.0 * fx_px)))


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate horizontal FOV from a known-size object at a measured distance")
    p.add_argument("--image-width", type=float, required=True, help="Full image width in pixels")
    p.add_argument("--image-height", type=int, required=True)
    p.add_argument("--object-px", type=float, required=True, help="Known object's horizontal width in pixels")
    p.add_argument("--object-m", type=float, required=True, help="Known object's real horizontal width in metres")
    p.add_argument("--distance-m", type=float, required=True, help="Measured camera-to-object distance in metres")
    p.add_argument("--output", default="calibration.json")
    p.add_argument("--name", default="calibrated-camera")
    args = p.parse_args()

    hfov = estimate_hfov_deg(args.image_width, args.object_px, args.object_m, args.distance_m)
    payload = {
        "width": int(args.image_width),
        "height": int(args.image_height),
        "hfov_deg": round(hfov, 6),
        "name": args.name,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"HFOV={hfov:.3f} deg")
    print(f"saved={args.output}")
    print("classification=CALIBRATED (quality depends on your distance/size measurements and lens zoom state)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
