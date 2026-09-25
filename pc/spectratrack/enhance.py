from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

DISPLAY_MODES = ("normal", "clarity", "lowlight", "edges", "pseudo-thermal")


def enhance_visibility(frame: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Non-generative visibility enhancement for live video.

    Uses local contrast + mild denoise + unsharp masking. This cannot recover
    information that is not present in the source frame.
    """
    strength = float(max(0.0, min(1.0, strength)))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0 + strength * 1.5, tileGridSize=(8, 8))
    lightness2 = clahe.apply(lightness)
    enhanced = cv2.cvtColor(cv2.merge([lightness2, a, b]), cv2.COLOR_LAB2BGR)

    if strength > 0.25:
        enhanced = cv2.bilateralFilter(enhanced, 5, 28, 28)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.05)
    sharpened = cv2.addWeighted(enhanced, 1.0 + 0.65 * strength, blurred, -0.65 * strength, 0)
    return sharpened


def assess_frame_quality(frame: np.ndarray) -> dict[str, float]:
    """Estimate issue severities for adaptive detector preprocessing.

    Values are normalized to 0..1 where a larger value means a stronger issue.
    These are routing heuristics, not calibrated image-quality scores.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a non-empty BGR image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)
    height, width = gray.shape

    mean_luma = float(np.mean(gray_f)) / 255.0
    darkness = float(np.clip((0.42 - mean_luma) / 0.42, 0.0, 1.0))

    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    blur = float(np.clip((120.0 - lap_var) / 120.0, 0.0, 1.0))

    smooth = cv2.GaussianBlur(gray, (3, 3), 0.0)
    residual = cv2.absdiff(gray, smooth)
    noise_level = float(np.median(residual))
    noise = float(np.clip((noise_level - 1.5) / 16.0, 0.0, 1.0))

    block_samples: list[float] = []
    inner_samples: list[float] = []
    if width > 16:
        boundaries = np.arange(8, width, 8)
        if len(boundaries):
            block_samples.append(float(np.mean(np.abs(gray_f[:, boundaries] - gray_f[:, boundaries - 1]))))
        inner = np.arange(4, width, 8)
        inner = inner[inner > 0]
        if len(inner):
            inner_samples.append(float(np.mean(np.abs(gray_f[:, inner] - gray_f[:, inner - 1]))))
    if height > 16:
        boundaries = np.arange(8, height, 8)
        if len(boundaries):
            block_samples.append(float(np.mean(np.abs(gray_f[boundaries, :] - gray_f[boundaries - 1, :]))))
        inner = np.arange(4, height, 8)
        inner = inner[inner > 0]
        if len(inner):
            inner_samples.append(float(np.mean(np.abs(gray_f[inner, :] - gray_f[inner - 1, :]))))
    block_edge = float(np.mean(block_samples)) if block_samples else 0.0
    inner_edge = float(np.mean(inner_samples)) if inner_samples else block_edge
    compression = float(np.clip((block_edge - inner_edge) / max(inner_edge + 4.0, 1.0), 0.0, 1.0))

    short_side = float(min(height, width))
    low_resolution = float(np.clip((720.0 - short_side) / 720.0, 0.0, 1.0))

    return {
        "blur": blur,
        "darkness": darkness,
        "compression": compression,
        "low_resolution": low_resolution,
        "noise": noise,
    }


def adaptive_analysis_frame(frame: np.ndarray) -> tuple[np.ndarray, dict[str, float], tuple[str, ...]]:
    """Apply bounded non-generative preprocessing selected from frame quality.

    This is intended for an optional detector analysis branch. It does not
    upscale the whole frame; tiny-object scaling is handled by tiled inference.
    """
    quality = assess_frame_quality(frame)
    out = frame
    operations: list[str] = []

    if quality["darkness"] >= 0.22:
        gamma = 1.0 - min(0.32, quality["darkness"] * 0.28)
        lut = np.clip(
            np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        out = cv2.LUT(out, lut)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8 + quality["darkness"] * 0.8, tileGridSize=(8, 8))
        out = cv2.cvtColor(cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2BGR)
        operations.append("low_light")

    if quality["noise"] >= 0.24 or quality["compression"] >= 0.20:
        diameter = 5 if max(quality["noise"], quality["compression"]) < 0.65 else 7
        out = cv2.bilateralFilter(out, diameter, 24, 24)
        operations.append("denoise_deblock")

    has_structure = float(np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) >= 8.0
    if has_structure and quality["blur"] >= 0.18 and quality["noise"] < 0.62 and quality["compression"] < 0.70:
        amount = min(0.38, 0.12 + quality["blur"] * 0.28)
        blurred = cv2.GaussianBlur(out, (0, 0), 0.9)
        out = cv2.addWeighted(out, 1.0 + amount, blurred, -amount, 0)
        operations.append("mild_sharpen")

    return out, quality, tuple(operations)


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


def upscale_preview(crop: np.ndarray, width: int = 420, height: int = 300) -> np.ndarray:
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
    """Run the user's locally installed official ncnn-vulkan binary.

    No downloader is included intentionally. The caller controls exactly which
    executable is run.
    """
    executable = Path(executable)
    if not executable.exists():
        raise FileNotFoundError(executable)
    cmd = [str(executable), "-i", str(input_path), "-o", str(output_path), "-s", str(scale)]
    subprocess.run(cmd, check=True, shell=False)


def apply_display_mode(frame: np.ndarray, mode: str) -> np.ndarray:
    """Operator display transform. Detection should run on a separate analysis frame.

    pseudo-thermal is only a false-color luminance visualization; it is not
    thermal sensing and must never be interpreted as temperature.
    """
    mode = mode.lower()
    if mode not in DISPLAY_MODES:
        raise ValueError(f"Unknown display mode: {mode}")
    if mode == "normal":
        return frame
    if mode == "clarity":
        return enhance_visibility(frame, 0.72)
    if mode == "lowlight":
        lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, 0.58) * 255.0, 0, 255).astype(np.uint8)
        lifted = cv2.LUT(frame, lut)
        return enhance_visibility(lifted, 0.48)
    if mode == "edges":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 55, 145)
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 0.78, edge_bgr, 0.70, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
