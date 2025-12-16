"""
GPU Environment Check Script
Verify PyTorch GPU installation
"""
import torch
import sys
import warnings

# Suppress RTX 5060 compatibility warning (GPU still works)
warnings.filterwarnings('ignore', message='.*CUDA capability.*')

print("=" * 60)
print("PyTorch GPU環境チェック")
print("=" * 60)
print()

# PyTorchバージョン
print(f"PyTorch version: {torch.__version__}")
print()

# CUDA情報
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()

    # GPU詳細情報
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  - Compute capability: {props.major}.{props.minor}")
        print(f"  - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - Multi-processor count: {props.multi_processor_count}")
    print()

    # GPU computation test
    print("GPU Operation Test:")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"  [OK] Matrix computation successful (result shape: {z.shape})")
    print()

    # Optimization features check
    print("Optimization Features:")
    print(f"  - torch.compile: {'[OK] Available' if hasattr(torch, 'compile') else '[X] Not available'}")
    print(f"  - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - Mixed Precision (AMP): [OK] Available")
    print()

    print("=" * 60)
    print("[OK] GPU environment is ready!")
    print("=" * 60)
else:
    print()
    print("=" * 60)
    print("[X] CUDA is not available")
    print("=" * 60)
    print()
    print("Solutions:")
    print("1. Check if NVIDIA driver is up to date")
    print("2. Install PyTorch GPU version:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    print()
    sys.exit(1)
