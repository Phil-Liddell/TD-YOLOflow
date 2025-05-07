@echo off
setlocal enabledelayedexpansion
title YOLOflow – first-time environment bootstrap

rem ─── configurable bits ────────────────────────────────────────────────
set "PYTHON=py -3.11"                                ^
&& set "EXTRA_INDEX=https://download.pytorch.org/whl/cu118"
rem ──────────────────────────────────────────────────────────────────────

echo Removing old virtual environment …
if exist ".venv" rmdir /s /q ".venv"

%PYTHON% -c "import sys,platform;print('Using Python',platform.python_version())" >nul 2>&1 || (
    echo ERROR: Python 3.11 not found.  Install it and re-run.
    pause & exit /b 1
)

echo Creating new virtual environment …
%PYTHON% -m venv .venv  || goto :fail
call ".venv\Scripts\activate.bat"

rem ── locate most-recent wheel in dist\  ────────────────────────────────
for /f "delims=" %%W in ('dir /b /o:-n ".\dist\yoloflow-*.whl" 2^>nul') do (
    set "WHEEL=dist\%%W"
    goto :have_wheel
)
echo ERROR: No wheel found in dist\.  Run  py -3.11 -m poetry build  first.
goto :fail

:have_wheel
echo Installing %WHEEL% …
pip install --extra-index-url %EXTRA_INDEX% "%WHEEL%" || goto :fail

rem ── sanity check (CUDA available?) ────────────────────────────────────
python -c "import torch, sys; print('✓ PyTorch', torch.__version__, 'CUDA OK' if torch.cuda.is_available() else '⚠ CPU-only build')"

echo(
echo =============================================================
echo   Environment ready – double-click  run_app.bat  to launch
echo =============================================================
pause
exit /b 0

:fail
echo(
echo !!!!!  Environment setup FAILED  !!!!!
echo        See messages above.
pause
exit /b 1
