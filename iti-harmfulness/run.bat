@echo off
REM Quick start script for ITI Harmfulness Steering System (Windows)
REM This activates the virtual environment and runs the main script

echo.
echo ========================================
echo ITI Harmfulness Steering System
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Activate venv
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat

REM Test environment
echo [2/3] Testing environment...
python test_env.py

REM Run main script
echo.
echo [3/3] Running ITI implementation...
echo.
python iti_harmfulness_main.py

pause
