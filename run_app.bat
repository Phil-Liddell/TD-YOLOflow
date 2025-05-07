@echo off
setlocal
if not exist ".venv\Scripts\activate.bat" (
    echo Run  setup_venv.bat  first.
    pause & exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m yoloflow %*
