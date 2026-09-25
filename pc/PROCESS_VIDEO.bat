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

set "SOURCE=%~1"
if not defined SOURCE (
  echo Drag a video file onto this BAT, or paste its full path below.
  set /p "SOURCE=Video path: "
)
if not exist "%SOURCE%" (
  echo [SpectraTrack] Video not found: %SOURCE%
  pause
  exit /b 2
)

set "OUT=%~dpn1_SpectraTrack.mp4"
set "LOG=%~dpn1_SpectraTrack.jsonl"
if "%~1"=="" (
  set "OUT=%CD%\RESULT_MAX.mp4"
  set "LOG=%CD%\RESULT_MAX.jsonl"
)

echo.
echo Model : %MODEL%
echo Video : %SOURCE%
echo Output: %OUT%
echo.

SpectraTrack-PC.exe --model "%MODEL%" --source "%SOURCE%" --profile quality --detect-every 1 --conf 0.20 --iou 0.50 --view clarity --record "%OUT%" --session-log "%LOG%"
set "CODE=%ERRORLEVEL%"

echo.
if "%CODE%"=="0" (
  echo [SpectraTrack] Done.
  echo Video: %OUT%
  echo Log  : %LOG%
) else (
  echo [SpectraTrack] Failed with code %CODE%.
)
pause
exit /b %CODE%
