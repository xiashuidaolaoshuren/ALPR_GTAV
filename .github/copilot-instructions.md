# Copilot Instructions for GTA V ALPR Project

## Project Architecture
- **Pipeline:** Two-stage: (1) License Plate Detection (YOLOv8 via Ultralytics), (2) License Plate Recognition (OCR via PaddleOCR).
- **Data Flow:** Video frame → YOLOv8 detects plates → Crop plate → PaddleOCR recognizes text → Tracking algorithm maintains plate identity across frames.
- **Tracking:** Use ByteTrack (integrated with YOLOv8) or IOU-based tracker to avoid redundant OCR.

## Key Files & Directories
- Project plan: `GTA_V_ALPR_Project_Plan.md` (contains technical stack, methodology, and dataset details)
- Datasets: Organize as `datasets/lpr/train/images`, `datasets/lpr/valid/images`, etc. Use YOLO format for detection and PaddleOCR format for recognition.
- Environment: Use Python 3.9+, recommend `venv` or `conda`. Required packages: `ultralytics`, `opencv-python`, `paddlepaddle-gpu`, `paddleocr`, `albumentations`.

## Development Guidelines
- **Coding Standards:** Follow PEP 8. Use meaningful variable names and modular functions.
- **MCP Uses:** Use the following MCP servers when developing:
  - `context7`: get the latest information and avoid deprecated methods
  - `sequential-thinking` & `shrimp-task-manager`: for structured problem solving
  - `mcp-feedback-enhanced`: for interactive feedback, follows the mcp instructions before completing any task.
---
For more details, see `GTA_V_ALPR_Project_Plan.md`.
