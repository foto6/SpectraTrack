from __future__ import annotations

import argparse
import platform
import sys

import cv2
import numpy as np
import onnxruntime as ort


def probe_cameras(max_index: int) -> list[int]:
    found = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
        if cap.isOpened():
            found.append(idx)
        cap.release()
    return found


def main() -> int:
    p = argparse.ArgumentParser(description="SpectraTrack environment diagnostics")
    p.add_argument("--probe-cameras", action="store_true", help="Actively open camera indices 0..N")
    p.add_argument("--max-camera", type=int, default=6)
    args = p.parse_args()

    print("SpectraTrack PC diagnostics")
    print(f"python={sys.version.split()[0]}")
    print(f"os={platform.platform()}")
    print(f"opencv={cv2.__version__}")
    print(f"numpy={np.__version__}")
    print(f"onnxruntime={ort.__version__}")
    print(f"ort_available={','.join(ort.get_available_providers())}")
    print("directml=" + ("yes" if "DmlExecutionProvider" in ort.get_available_providers() else "no"))
    if args.probe_cameras:
        print("cameras=" + ",".join(map(str, probe_cameras(args.max_camera))))
    else:
        print("cameras=not-probed (use --probe-cameras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
