from __future__ import annotations

import sys
from pathlib import Path


def _choose_model() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"GUI file picker unavailable: {exc}")
        return ""

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askopenfilename(
            title="Select SpectraTrack ONNX detector",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path


def main() -> int:
    if len(sys.argv) == 1:
        model = _choose_model()
        if not model:
            return 0
        sys.argv.extend(["--model", model, "--source", "0", "--adaptive-detect", "--target-fps", "30"])
        sidecar = Path(model + ".manifest.json")
        if sidecar.is_file():
            sys.argv.extend(["--model-manifest", str(sidecar)])

    from spectratrack.app import main as app_main
    return app_main()


if __name__ == "__main__":
    raise SystemExit(main())
