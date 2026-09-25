@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "MODEL="
if exist "model.onnx" set "MODEL=model.onnx"
if not defined MODEL if exist "yolo11x.onnx" set "MODEL=yolo11x.onnx"
if not defined MODEL for %%F in (*.onnx) do if not defined MODEL set "MODEL=%%F"

if not defined MODEL (
  echo [SpectraTrack] No .onnx model found in %CD%
  echo Put a compatible fixed-size YOLO ONNX model next to this BAT file.
  pause
  exit /b 1
)

set "INPUT=%~1"
if not defined INPUT (
  echo Drag a folder containing videos onto this BAT, or paste its path below.
  set /p "INPUT=Folder path: "
)
if not exist "%INPUT%\" (
  echo [SpectraTrack] Folder not found: %INPUT%
  pause
  exit /b 2
)

set "OUT=%INPUT%\spectratrack_cross_video.json"
set "REVIEW="
if exist "%INPUT%\spectratrack_review.json" set "REVIEW=--review "%INPUT%\spectratrack_review.json""

echo.
echo Model : %MODEL%
echo Folder: %INPUT%
echo Graph : %OUT%
echo.

SpectraTrack-PC.exe batch --model "%MODEL%" --input-dir "%INPUT%" --output "%OUT%" --detect-every 1 --conf 0.20 --iou 0.50 --recursive %REVIEW%
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo [SpectraTrack] Batch analysis complete.
  echo JSON  : %OUT%
  echo Report: %INPUT%\spectratrack_cross_video.html
  start "" "%INPUT%\spectratrack_cross_video.html"
) else (
  echo [SpectraTrack] Failed with code %CODE%.
)
pause
exit /b %CODE%
