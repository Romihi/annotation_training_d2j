@echo off
echo ========================================
echo YOLO Training Readiness Check
echo ========================================
echo.

set SCRIPT_DIR=%~dp0
call "%SCRIPT_DIR%venv\Scripts\activate.bat"

echo Checking PyTorch...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo Checking torchvision...
python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"

echo.
echo Checking Ultralytics...
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"

echo.
echo Testing YOLO model loading...
python -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); print('YOLO model loading: OK')"

echo.
echo GPU Test...
python -c "import torch; x=torch.rand(3,3).cuda(); print(f'GPU Tensor: {x.device}'); print('Ready for training!')"

echo.
pause
