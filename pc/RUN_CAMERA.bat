@echo off
setlocal
cd /d "%~dp0"

if not exist model.onnx (
  echo.
  echo [SpectraTrack] model.onnx not found.
  echo Copy your verified ONNX model into this folder and name it model.onnx.
  echo Optional: also place model.manifest.json next to it for SHA-256 verification.
  echo.
  pause
  exit /b 1
)

set MANIFEST=
if exist model.manifest.json set MANIFEST=--model-manifest model.manifest.json

if not exist sessions mkdir sessions

echo Starting SpectraTrack PC...
SpectraTrack-PC.exe --model model.onnx %MANIFEST% --source 0 --profile balanced --session-log "sessions\latest.jsonl"

echo.
echo SpectraTrack exited with code %ERRORLEVEL%.
pause
