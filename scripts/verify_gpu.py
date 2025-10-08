"""
GPU Availability Verification Script for GTA V ALPR Project

This script checks the availability of GPU acceleration for:
- PyTorch (used by Ultralytics YOLOv8)
- PaddlePaddle (used by PaddleOCR)

Usage:
    python scripts/verify_gpu.py
"""

import sys
import importlib

import torch
import paddle


def check_pytorch_gpu():
    """Check PyTorch CUDA availability."""
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  - CUDA device count: {torch.cuda.device_count()}")
        print(f"  - CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    else:
        print("  - Running on CPU (GPU not available)")
    
    return cuda_available


def check_paddle_gpu():
    """Check PaddlePaddle CUDA availability."""
    cuda_available = paddle.device.is_compiled_with_cuda()
    print(f"\nPaddlePaddle CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_count = paddle.device.cuda.device_count()
        print(f"  - GPU device count: {gpu_count}")
        if gpu_count > 0:
            print(f"  - Using GPU: {paddle.device.get_device()}")
    else:
        print("  - Running on CPU (GPU not available)")
    
    return cuda_available


def verify_imports():
    """Verify all required packages can be imported."""
    print("\n" + "="*60)
    print("Verifying Package Imports")
    print("="*60)
    
    packages = {
        'ultralytics': 'Ultralytics (YOLOv8)',
        'cv2': 'OpenCV',
        'paddleocr': 'PaddleOCR',
        'albumentations': 'Albumentations',
        'yaml': 'PyYAML',
        'pytest': 'Pytest'
    }
    
    all_success = True
    for package_name, display_name in packages.items():
        if importlib.util.find_spec(package_name) is None:
            print(f"✗ {display_name} ({package_name}) - NOT INSTALLED")
            all_success = False
            continue

        importlib.import_module(package_name)
        print(f"✓ {display_name} ({package_name})")
    
    return all_success


def get_package_versions():
    """Get versions of installed packages."""
    print("\n" + "="*60)
    print("Package Versions")
    print("="*60)
    
    packages = ['torch', 'ultralytics', 'cv2', 'paddle', 'paddleocr', 
                'albumentations', 'yaml', 'pytest']
    
    for package_name in packages:
        if importlib.util.find_spec(package_name) is None:
            print(f"{package_name}: Not installed")
            continue

        module = importlib.import_module(package_name)
        if package_name == 'cv2':
            print(f"{package_name}: {module.__version__}")
            continue

        version = getattr(module, '__version__', 'Unknown')
        print(f"{package_name}: {version}")


def main():
    """Main verification function."""
    print("="*60)
    print("GTA V ALPR Project - Environment Verification")
    print("="*60)
    
    # Check GPU availability
    pytorch_gpu = check_pytorch_gpu()
    paddle_gpu = check_paddle_gpu()
    
    # Verify imports
    imports_ok = verify_imports()
    
    # Get package versions
    get_package_versions()
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    print(f"PyTorch GPU: {'✓ Available' if pytorch_gpu else '✗ Not Available (CPU mode)'}")
    print(f"PaddlePaddle GPU: {'✓ Available' if paddle_gpu else '✗ Not Available (CPU mode)'}")
    print(f"All packages installed: {'✓ Yes' if imports_ok else '✗ No (see errors above)'}")
    
    if not (pytorch_gpu or paddle_gpu):
        print("\n⚠ Warning: No GPU detected. Models will run on CPU (slower performance).")
    
    if imports_ok and (pytorch_gpu or paddle_gpu):
        print("\n✓ Environment setup complete and GPU-accelerated!")
        return 0
    elif imports_ok:
        print("\n✓ Environment setup complete (CPU mode).")
        return 0
    else:
        print("\n✗ Environment setup incomplete. Install missing packages.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
