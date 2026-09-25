SPECTRATRACK PC v0.2 - STANDALONE QUICK START
=================================================

1. Copy a compatible fixed-size YOLO ONNX model into this folder.
2. Rename it to:
     model.onnx
3. Recommended: copy its SpectraTrack manifest here as:
     model.manifest.json
4. Double-click:
     START_SPECTRATRACK.bat

START_SPECTRATRACK.bat uses:
- camera 0
- balanced profile (detector every 2nd frame)
- local JSONL logging at sessions\latest.jsonl
- DirectML when ONNX Runtime exposes DmlExecutionProvider, otherwise CPU

Useful direct commands:

  SpectraTrack-PC.exe --help

  SpectraTrack-PC.exe --model model.onnx --source 0 --profile quality

  SpectraTrack-PC.exe --model model.onnx --source video.mp4 --headless --record analyzed.mp4 --session-log analyzed.jsonl

Controls:
  Left click target = lock/unlock
  E = detector analysis clarity enhancement
  M = cycle operator view
  Z = stabilization
  R = tracker reset
  H = HUD
  S = snapshot
  U = optional Real-ESRGAN snapshot upscale
  Q / Esc = quit

Notes:
- PSEUDO-THERMAL is false-color luminance, not thermal sensing.
- Angular data requires a valid calibration file.
- AI super-resolution can invent fine detail.
- SpectraTrack does not auto-download models or executables.
