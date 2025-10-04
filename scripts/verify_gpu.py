"""
GPU Availability Verification Script for GTA V ALPR Project

This script checks the availability of GPU acceleration for:
- PyTorch (used by Ultralytics YOLOv8)
- PaddlePaddle (used by PaddleOCR)

Usage:
    python scripts/verify_gpu.py
"""

import sys

def check_pytorch_gpu():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"  - CUDA device count: {torch.cuda.device_count()}")
            print(f"  - CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
        else:
            print("  - Running on CPU (GPU not available)")
        
        return cuda_available
    except ImportError:
        print("PyTorch not installed!")
        return False
    except Exception as e:
        print(f"Error checking PyTorch: {e}")
        return False


def check_paddle_gpu():
    """Check PaddlePaddle CUDA availability."""
    try:
        import paddle
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
    except ImportError:
        print("\nPaddlePaddle not installed!")
        return False
    except Exception as e:
        print(f"\nError checking PaddlePaddle: {e}")
        return False


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
        try:
            __import__(package_name)
            print(f"✓ {display_name} ({package_name})")
        except ImportError:
            print(f"✗ {display_name} ({package_name}) - NOT INSTALLED")
            all_success = False
    
    return all_success


def get_package_versions():
    """Get versions of installed packages."""
    print("\n" + "="*60)
    print("Package Versions")
    print("="*60)
    
    packages = ['torch', 'ultralytics', 'cv2', 'paddle', 'paddleocr', 
                'albumentations', 'yaml', 'pytest']
    
    for package_name in packages:
        try:
            if package_name == 'cv2':
                import cv2
                print(f"{package_name}: {cv2.__version__}")
            elif package_name == 'yaml':
                import yaml
                print(f"{package_name}: {yaml.__version__}")
            else:
                module = __import__(package_name)
                version = getattr(module, '__version__', 'Unknown')
                print(f"{package_name}: {version}")
        except ImportError:
            print(f"{package_name}: Not installed")
        except Exception as e:
            print(f"{package_name}: Error - {e}")


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
