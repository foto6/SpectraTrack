from __future__ import annotations

import os

import cv2


def parse_source(text: str):
    return int(text) if text.isdigit() else text


def open_capture(source_text: str) -> cv2.VideoCapture:
    source = parse_source(source_text)
    if os.name == "nt" and isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source_text}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
