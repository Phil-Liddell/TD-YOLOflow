@echo off
chcp 65001 >nul
title YOLOflow - Object Detection Workflow Tool
color 0A
cls

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo ERROR: Python not found\!
    echo Please install Python 3.11 or later from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if requirements are installed
python -c "import PyQt5, cv2, numpy" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing required packages...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        color 0C
        echo ERROR: Failed to install dependencies.
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM Create data directory if it doesn't exist
if not exist data\nul mkdir data
if not exist runs\nul mkdir runs

REM Download default model if needed
if not exist models\yolo11n.pt if not exist models\yolo11s.pt (
    echo Downloading default model...
    python download_models.py
)

REM Launch the application
python main.py

REM If the application exits with an error, keep the window open
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo ERROR: YOLOflow exited with an error.
    pause
)
