from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .capture import open_capture


def main() -> int:
    p = argparse.ArgumentParser(description="Capture checkerboard calibration frames")
    p.add_argument("--source", default="0")
    p.add_argument("--output", default="calibration-images")
    p.add_argument("--cols", type=int, default=0, help="Optional checkerboard inner corners across")
    p.add_argument("--rows", type=int, default=0, help="Optional checkerboard inner corners down")
    args = p.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cap = open_capture(args.source)
    saved = len(list(output.glob("calib-*.png")))
    window = "SpectraTrack calibration capture"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            display = frame.copy()
            found = False
            corners = None
            if args.cols >= 3 and args.rows >= 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows))
                if found:
                    cv2.drawChessboardCorners(display, (args.cols, args.rows), corners, found)
            status = f"SAVED {saved} | SPACE capture | Q quit"
            if args.cols >= 3 and args.rows >= 3:
                status += f" | BOARD {'OK' if found else 'NO'}"
            cv2.putText(display, status, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                saved += 1
                path = output / f"calib-{saved:04d}.png"
                cv2.imwrite(str(path), frame)
                print(path)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
