@echo off
REM Setup script for eRisk 2026 Task 1 (Windows)

echo ==========================================
echo eRisk 2026 Task 1 - Setup Script
echo ==========================================

REM Check Python version
echo Checking Python version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
echo.
echo Creating directories...
if not exist submissions mkdir submissions
if not exist personas mkdir personas

REM Copy .env.example to .env if .env doesn't exist
if not exist .env (
    echo.
    echo Creating .env file from .env.example...
    copy .env.example .env
    echo Please edit .env and add your Hugging Face token!
)

REM Run tests
echo.
echo Running setup tests...
python scripts\test_setup.py

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your HF_TOKEN
echo 2. Run: python scripts\test_setup.py
echo 3. Run: python quick_start.py
echo 4. Run: streamlit run demo\app.py
echo.

pause
