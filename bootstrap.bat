REM bootstrap.bat – run from YOLOflow repo root
@echo off
set "PYTHON=python"

if not exist .venv (
    echo Creating virtual-env …
    %PYTHON% -m venv .venv || exit /b 1
    call .venv\Scripts\activate
    python -m pip install --upgrade pip
    REM ↓ install with GUI requirements (PyQt5 etc.)
    python -m pip install -e ".[gui]"
    echo Venv ready.
) else (
    echo Venv already exists.
)
