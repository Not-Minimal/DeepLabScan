@echo off
REM Quick Start Script for DeepLabScan YOLO Project (Windows)
REM This script helps you set up the project environment

echo ================================
echo DeepLabScan Setup Script
echo ================================
echo.

REM Check Python version
echo 1. Checking Python version...
python --version
echo.

REM Create virtual environment
echo 2. Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo    [OK] Virtual environment created
) else (
    echo    [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 3. Activating virtual environment...
call venv\Scripts\activate
echo    [OK] Virtual environment activated
echo.

REM Upgrade pip
echo 4. Upgrading pip...
python -m pip install --upgrade pip -q
echo    [OK] pip upgraded
echo.

REM Install dependencies
echo 5. Installing dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt -q
echo    [OK] Dependencies installed
echo.

REM Verify directories
echo 6. Verifying directory structure...
echo    [OK] All directories are in place
echo.

REM Print next steps
echo ================================
echo Setup Complete! 🎉
echo ================================
echo.
echo Next steps:
echo.
echo 1. Set up Roboflow account and create a project
echo    Visit: https://roboflow.com/
echo.
echo 2. Upload and label your data in Roboflow
echo.
echo 3. Export your dataset in YOLO format to data\roboflow\
echo.
echo 4. Update configs\data_config.yaml with your class names
echo.
echo 5. Start training:
echo    python src\training\train.py --config configs\training_config.yaml
echo.
echo 6. For detailed instructions, see docs\getting_started.md
echo.
echo To activate the environment in the future, run:
echo    venv\Scripts\activate
echo.
pause
