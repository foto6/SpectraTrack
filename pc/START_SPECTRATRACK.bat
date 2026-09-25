@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MODEL="
if exist "model.onnx" set "MODEL=model.onnx"
if not defined MODEL if exist "yolo11x.onnx" set "MODEL=yolo11x.onnx"
if not defined MODEL for %%F in (*.onnx) do if not defined MODEL set "MODEL=%%F"

if not defined MODEL (
  echo.
  echo [SpectraTrack] No .onnx model found in this folder.
  echo Copy a compatible fixed-size YOLO ONNX model here.
  echo Preferred names: model.onnx or yolo11x.onnx.
  echo.
  pause
  exit /b 1
)

set "MANIFEST="
if exist "model.manifest.json" set "MANIFEST=--model-manifest model.manifest.json"

if not exist sessions mkdir sessions

echo Starting SpectraTrack PC...
echo Model: %MODEL%
SpectraTrack-PC.exe --model "%MODEL%" %MANIFEST% --source 0 --profile balanced --session-log "sessions\latest.jsonl"

echo.
echo SpectraTrack exited with code %ERRORLEVEL%.
pause
