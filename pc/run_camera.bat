@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements-win.txt
if not exist ..\models\yolo11n.onnx (
  echo Missing ..\models\yolo11n.onnx
  echo Read ..\models\README.md first.
  pause
  exit /b 1
)
python -m spectratrack.app --model ..\models\yolo11n.onnx --source 0
