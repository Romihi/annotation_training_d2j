@echo off

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM ========================================
REM CPU Fallback Mode
REM Use this if CUDA kernel errors occur
REM ========================================
echo WARNING: Running in CPU fallback mode
echo CUDA will be disabled for this session
echo.

REM Disable CUDA
set CUDA_VISIBLE_DEVICES=-1

REM PyTorch optimization for CPU
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8

REM Check if virtual environment exists
if not exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first to create the virtual environment.
    echo.
    pause
    exit /b 1
)

REM Activate the virtual environment
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

REM Check if dependencies are installed
python -c "import matplotlib" >nul 2>&1
if errorlevel 1 (
    echo Error: Dependencies not installed!
    echo Please run setup.bat first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Run main.py in CPU mode
echo Starting application in CPU mode...
echo (YOLO training will be slower but stable)
echo.
python "%SCRIPT_DIR%main.py"

pause
