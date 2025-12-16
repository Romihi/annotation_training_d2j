@echo off
echo ========================================
echo PyTorch Nightly with CUDA 12.8
echo RTX 5060 sm_120 Support Test
echo ========================================
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Check if virtual environment exists
if not exist "%SCRIPT_DIR%venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run setup.bat first to create the virtual environment.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

echo.
echo Step 1: Checking current PyTorch installation...
python -c "import torch; print(f'Current PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul

echo.
echo Step 2: Uninstalling existing PyTorch packages...
pip uninstall -y torch torchvision torchaudio

echo.
echo Step 3: Installing PyTorch Nightly with CUDA 12.8 support...
echo This may take several minutes...
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

echo.
echo Step 4: Verifying CUDA installation...
python -c "import torch; print('='*50); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print('='*50)"

echo.
echo Step 5: Testing GPU tensor operations...
python -c "import torch; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); x = torch.rand(5, 3).to(device); print(f'Tensor device: {x.device}'); print('GPU test successful!' if x.device.type == 'cuda' else 'GPU test failed')"

echo.
echo Installation complete!
echo.
echo Next step: Run check_yolo_ready.bat to verify YOLO readiness
echo Then run your application and test YOLO training
echo.
pause
