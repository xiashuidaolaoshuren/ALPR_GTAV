"""
GPU Environment Verification Script

Confirm GPU acceleration support for core libraries used in the GTA V ALPR
pipeline, including PyTorch (YOLOv8) and PaddlePaddle (PaddleOCR).

Usage:
    python scripts/diagnostics/verify_gpu.py
"""

from __future__ import annotations

import importlib
import sys

import paddle
import torch


def check_pytorch_gpu() -> bool:
    """Check PyTorch CUDA availability."""
    available = torch.cuda.is_available()
    print(f"PyTorch CUDA available: {available}")

    if available:
        print(f"  - CUDA device count: {torch.cuda.device_count()}")
        print(f"  - CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    else:
        print("  - Running on CPU (GPU not available)")

    return available


def check_paddle_gpu() -> bool:
    """Check PaddlePaddle CUDA availability."""
    available = paddle.device.is_compiled_with_cuda()
    print(f"\nPaddlePaddle CUDA available: {available}")

    if available:
        device_count = paddle.device.cuda.device_count()
        print(f"  - GPU device count: {device_count}")
        if device_count:
            print(f"  - Using GPU: {paddle.device.get_device()}")
    else:
        print("  - Running on CPU (GPU not available)")

    return available


def verify_imports() -> bool:
    """Verify all required packages can be imported."""
    print("\n" + "=" * 60)
    print("Verifying Package Imports")
    print("=" * 60)

    packages = {
        "ultralytics": "Ultralytics (YOLOv8)",
        "cv2": "OpenCV",
        "paddleocr": "PaddleOCR",
        "albumentations": "Albumentations",
        "yaml": "PyYAML",
        "pytest": "Pytest",
    }

    all_present = True
    for package_name, display in packages.items():
        if importlib.util.find_spec(package_name) is None:
            print(f"✗ {display} ({package_name}) - NOT INSTALLED")
            all_present = False
            continue

        importlib.import_module(package_name)
        print(f"✓ {display} ({package_name})")

    return all_present


def get_package_versions() -> None:
    """Print versions of installed packages."""
    print("\n" + "=" * 60)
    print("Package Versions")
    print("=" * 60)

    packages = [
        "torch",
        "ultralytics",
        "cv2",
        "paddle",
        "paddleocr",
        "albumentations",
        "yaml",
        "pytest",
    ]

    for package_name in packages:
        if importlib.util.find_spec(package_name) is None:
            print(f"{package_name}: Not installed")
            continue

        module = importlib.import_module(package_name)
        if package_name == "cv2":
            print(f"{package_name}: {module.__version__}")
            continue

        version = getattr(module, "__version__", "Unknown")
        print(f"{package_name}: {version}")


def main() -> int:
    print("=" * 60)
    print("GTA V ALPR Project - Environment Verification")
    print("=" * 60)

    pytorch_gpu = check_pytorch_gpu()
    paddle_gpu = check_paddle_gpu()
    imports_ok = verify_imports()
    get_package_versions()

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"PyTorch GPU: {'✓ Available' if pytorch_gpu else '✗ Not Available (CPU mode)'}")
    print(f"PaddlePaddle GPU: {'✓ Available' if paddle_gpu else '✗ Not Available (CPU mode)'}")
    print(f"All packages installed: {'✓ Yes' if imports_ok else '✗ No (see errors above)'}")

    if not (pytorch_gpu or paddle_gpu):
        print("\n⚠ Warning: No GPU detected. Models will run on CPU (slower performance).")

    if imports_ok and (pytorch_gpu or paddle_gpu):
        print("\n✓ Environment setup complete and GPU-accelerated!")
        return 0

    if imports_ok:
        print("\n✓ Environment setup complete (CPU mode).")
        return 0

    print("\n✗ Environment setup incomplete. Install missing packages.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
