@echo off
setlocal
if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -U pip
pip install -r requirements-win.txt
python -m spectratrack.demo
