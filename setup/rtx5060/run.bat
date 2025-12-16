@echo off

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM ========================================
REM RTX 5060 Laptop GPU Optimization
REM ========================================
REM CRITICAL: RTX 5060 (sm_120) compatibility fix
REM Force PyTorch to use compatible CUDA architecture
set TORCH_CUDA_ARCH_LIST=9.0
set CUDA_VISIBLE_DEVICES=0

REM PyTorch CUDA optimization
set CUDA_LAUNCH_BLOCKING=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

REM cuDNN optimization
set CUDNN_BENCHMARK=1

REM PyTorch optimization
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4

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

REM Run main.py
echo Starting application...
python "%SCRIPT_DIR%main.py"

pause
