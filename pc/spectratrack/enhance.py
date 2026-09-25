from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

ENHANCE_MODES = ("off", "visibility", "lowlight", "detail")


def enhance_visibility(frame: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Non-generative local contrast + mild denoise + unsharp masking."""
    strength = float(max(0.0, min(1.0, strength)))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0 + strength * 1.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    if strength > 0.25:
        enhanced = cv2.bilateralFilter(enhanced, 5, 28, 28)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.05)
    return cv2.addWeighted(enhanced, 1.0 + 0.65 * strength, blurred, -0.65 * strength, 0)


def enhance_lowlight(frame: np.ndarray) -> np.ndarray:
    """Adaptive gamma lift followed by CLAHE.

    This brightens information already present in the image; it is not a
    generative low-light model and does not invent missing scene detail.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean >= 125.0:
        gamma = 0.92
    elif mean >= 75.0:
        gamma = 0.76
    else:
        gamma = 0.58
    lut = np.array([min(255, round(((i / 255.0) ** gamma) * 255.0)) for i in range(256)], dtype=np.uint8)
    lifted = cv2.LUT(frame, lut)
    return enhance_visibility(lifted, strength=0.52)


def enhance_detail(frame: np.ndarray) -> np.ndarray:
    """Edge-preserving detail mode for already visible scenes."""
    denoised = cv2.bilateralFilter(frame, 5, 22, 22)
    blur = cv2.GaussianBlur(denoised, (0, 0), 0.85)
    return cv2.addWeighted(denoised, 1.72, blur, -0.72, 0)


def apply_enhancement(frame: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode == "off":
        return frame
    if mode == "visibility":
        return enhance_visibility(frame)
    if mode == "lowlight":
        return enhance_lowlight(frame)
    if mode == "detail":
        return enhance_detail(frame)
    raise ValueError(f"Unknown enhancement mode: {mode}")


def next_enhancement_mode(mode: str) -> str:
    try:
        idx = ENHANCE_MODES.index(mode.lower())
    except ValueError:
        return ENHANCE_MODES[0]
    return ENHANCE_MODES[(idx + 1) % len(ENHANCE_MODES)]


def crop_with_margin(frame: np.ndarray, bbox: tuple[float, float, float, float], margin: float = 0.22) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * margin))
    y1 = int(max(0, y1 - bh * margin))
    x2 = int(min(w, x2 + bw * margin))
    y2 = int(min(h, y2 + bh * margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def upscale_preview(crop: np.ndarray | None, width: int = 420, height: int = 300) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    ch, cw = crop.shape[:2]
    scale = min(width / max(cw, 1), height / max(ch, 1))
    out = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    oy = (height - out.shape[0]) // 2
    ox = (width - out.shape[1]) // 2
    canvas[oy:oy + out.shape[0], ox:ox + out.shape[1]] = out
    return canvas


def run_realesrgan_snapshot(executable: str | Path, input_path: str | Path, output_path: str | Path, scale: int = 4) -> None:
    """Run an explicitly supplied local Real-ESRGAN ncnn/Vulkan executable."""
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(executable)
    cmd = [str(executable), "-i", str(input_path), "-o", str(output_path), "-s", str(scale)]
    subprocess.run(cmd, check=True, shell=False)
