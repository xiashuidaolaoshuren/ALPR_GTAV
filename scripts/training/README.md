# Training Scripts

Tools for training and fine-tuning the YOLOv8 detection model on the GTA V ALPR dataset.

## Overview

This directory contains utilities for fine-tuning the YOLOv8 model to detect license plates in GTA V gameplay footage. The training pipeline handles data preparation, hyperparameter configuration, and model evaluation.

## Quick Start

### Train Detection Model

```powershell
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 100 `
  --batch-size 16 `
  --device cuda
```

### Resume Training from Checkpoint

```powershell
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 100 `
  --batch-size 16 `
  --resume models/detection/yolov8_finetuned_v2_best.pt
```

### Train with Custom Hyperparameters

```powershell
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 100 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --weight-decay 0.0005 `
  --augmentation-intensity high
```

## Key Scripts

### train_detection.py
**Fine-tune YOLOv8 on the GTA V ALPR dataset**

Implements the full training pipeline:
- Data loading and validation
- Model initialization (YOLOv8n as default)
- Training with configurable hyperparameters
- Automatic checkpoint saving
- Validation on test set
- Metrics logging and visualization

**Arguments:**
```
--data                Path to data.yaml (required)
--epochs              Number of training epochs (default: 100)
--batch-size          Batch size for training (default: 16)
--device              cuda or cpu (default: cuda)
--learning-rate       Initial learning rate (default: 0.001)
--weight-decay        L2 regularization coefficient (default: 0.0005)
--momentum            SGD momentum (default: 0.937)
--warmup-epochs       Number of warmup epochs (default: 5)
--patience            Early stopping patience (default: 20)
--augmentation-intensity  low/medium/high (default: medium)
--model-size          n/s/m/l/x for YOLOv8 (default: n)
--resume              Path to checkpoint for resuming training
--output-dir          Output directory for results (default: runs/detect)
--seed                Random seed for reproducibility (default: 42)
--verbose             Enable verbose logging (default: False)
```

**Output:**
- `runs/detect/train<N>/weights/best.pt` - Best model weights
- `runs/detect/train<N>/weights/last.pt` - Last epoch weights
- `runs/detect/train<N>/results.csv` - Training metrics
- `runs/detect/train<N>/` - Tensorboard logs (if enabled)

### generate_training_config.py
**Create and validate training configuration files**

Generates customized training config with:
- Dataset-specific parameters
- Optimal hyperparameter suggestions
- Data augmentation settings
- Hardware-specific configurations

**Usage:**
```powershell
python scripts/training/generate_training_config.py `
  --output configs/training_config.yaml `
  --data-path datasets/lpr `
  --batch-size 16 `
  --target-fps 15
```

## Command-Line Arguments

### Common Flags

```
--help, -h              Show help message
--verbose, -v           Enable verbose logging
--device                cuda or cpu
--seed                  Random seed for reproducibility
--output-dir, -o        Output directory for results
```

### Data Arguments

```
--data, -d              Path to data.yaml file (required)
--split-ratio           Train/val/test split ratio
--shuffle               Shuffle training data (default: True)
--num-workers           DataLoader workers (default: 4)
```

### Model Arguments

```
--model-size            YOLOv8 model size: n/s/m/l/x (default: n)
--pretrained            Use pretrained weights (default: True)
--freeze-backbone       Freeze backbone layers (default: False)
--freeze-neck           Freeze neck layers (default: False)
```

### Training Arguments

```
--epochs, -e            Number of epochs (default: 100)
--batch-size, -b        Batch size (default: 16)
--learning-rate, -lr    Initial learning rate (default: 0.001)
--weight-decay, -wd     L2 regularization (default: 0.0005)
--momentum              SGD momentum (default: 0.937)
--warmup-epochs         Warmup epochs (default: 5)
--warmup-momentum       Warmup momentum (default: 0.8)
```

### Regularization Arguments

```
--dropout               Dropout rate (default: 0.0)
--label-smoothing       Label smoothing factor (default: 0.0)
--hsv-h                 HSV hue augmentation (default: 0.015)
--hsv-s                 HSV saturation augmentation (default: 0.7)
--hsv-v                 HSV value augmentation (default: 0.4)
--degrees               Rotation augmentation (default: 0.0)
--translate             Translation augmentation (default: 0.1)
--scale                 Scale augmentation (default: 0.5)
--flipud                Flip upside down probability (default: 0.0)
--fliplr                Flip left right probability (default: 0.5)
--mosaic                Mosaic augmentation (default: 1.0)
--mixup                 Mixup augmentation (default: 0.0)
```

### Optimization Arguments

```
--optimizer             Optimizer: SGD/Adam/AdamW (default: SGD)
--patience              Early stopping patience (default: 20)
--resume                Path to checkpoint for resuming
--lr-scheduler          Scheduler: linear/cosine/poly (default: cosine)
```

## Training Workflows

### 1. Initial Training from Scratch

```powershell
# First time training on GTA V dataset
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 100 `
  --batch-size 16 `
  --model-size n `
  --warmup-epochs 10 `
  --seed 42
```

### 2. Resume Training from Checkpoint

```powershell
# Continue from best model
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 150 `
  --batch-size 16 `
  --resume models/detection/yolov8_finetuned_v2_best.pt
```

### 3. Fine-tune with Stronger Augmentation

```powershell
# Improve robustness to light/angle variations
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 100 `
  --batch-size 16 `
  --augmentation-intensity high `
  --degrees 15 `
  --scale 0.7 `
  --hsv-h 0.02 `
  --mosaic 1.0 `
  --mixup 0.1
```

### 4. Freeze Backbone for Transfer Learning

```powershell
# Fast training with pretrained backbone
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 50 `
  --batch-size 32 `
  --freeze-backbone `
  --learning-rate 0.005
```

### 5. Multi-Step Training Strategy

```powershell
# Phase 1: Conservative training with pretrained backbone frozen
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 30 `
  --batch-size 32 `
  --freeze-backbone `
  --learning-rate 0.01 `
  --warmup-epochs 5

# Phase 2: Full model training with lower learning rate
python scripts/training/train_detection.py `
  --data datasets/lpr/data.yaml `
  --epochs 50 `
  --batch-size 16 `
  --resume runs/detect/train<N>/weights/best.pt `
  --learning-rate 0.0001 `
  --warmup-epochs 0
```

## Recommended Hyperparameters

### For GTA V Gameplay Footage

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model Size | `n` or `s` | Balance accuracy/speed for real-time processing |
| Batch Size | 16 | GPU memory efficiency on RTX 3070Ti |
| Epochs | 100 | Typically converges by 80-100 epochs |
| Learning Rate | 0.001 | Standard SGD learning rate |
| Augmentation | Medium | GTA V has synthetic variation already |
| Warmup Epochs | 5-10 | Stabilize training in early phases |

### For Different Hardware

**RTX 3070Ti:**
```
--batch-size 32
--model-size m
```

**RTX 3060:**
```
--batch-size 16
--model-size n
```

**CPU Only:**
```
--batch-size 8
--model-size n
--device cpu
```

## Output Directory Structure

```
runs/detect/train<N>/
├── weights/
│   ├── best.pt           # Best model (lowest validation loss)
│   └── last.pt           # Last epoch model
├── results.csv           # Epoch-wise metrics
├── confusion_matrix.png  # Validation confusion matrix
├── F1_curve.png         # F1 vs confidence curve
├── P_curve.png          # Precision vs confidence curve
├── R_curve.png          # Recall vs confidence curve
└── ...                   # Other visualization outputs
```

## Best Practices

1. **Validate dataset before training**
   ```powershell
   python scripts/diagnostics/validate_config.py datasets/lpr/data.yaml
   ```

2. **Monitor training progress**
   - Watch `results.csv` for convergence
   - Check validation metrics don't degrade (overfitting)
   - Use early stopping patience to avoid wasted epochs

3. **Use appropriate augmentation**
   - Light augmentation for clean synthetic data (GTA V)
   - Heavy augmentation for real-world data
   - Validate augmentation doesn't hurt accuracy

4. **Save intermediate checkpoints**
   - Model automatically saves best weights
   - Also saves last epoch for resuming
   - Archive important models separately

5. **Validate on held-out test set**
   ```powershell
   python scripts/evaluation/evaluate_detection.py `
     --model runs/detect/train<N>/weights/best.pt `
     --data datasets/lpr/data.yaml
   ```

## Troubleshooting

### Training too slow

- **Solution:** Reduce batch size or use smaller model
- **Alternative:** Reduce `num-workers` if dataloader is bottleneck
- **Check:** GPU utilization with `nvidia-smi`

### Model not converging

- **Check learning rate:** Too high causes instability, too low causes slow convergence
- **Check data quality:** Ensure dataset has good variety
- **Try warmup:** Increase `warmup-epochs` to stabilize early training

### Out of memory

- **Reduce batch size:** `--batch-size 8`
- **Use smaller model:** `--model-size n` instead of `m`
- **Clear cache:** Restart Python interpreter between runs

### Validation loss increasing (overfitting)

- **Reduce epochs:** Enable early stopping with `--patience 10`
- **Add augmentation:** `--augmentation-intensity high`
- **Reduce model size:** Use `n` or `s` variant
- **Collect more data:** Dataset may be too small

### GPU out of memory during validation

- **Reduce image size:** In `data.yaml`, reduce `imgsz` value
- **Reduce batch size:** Validation batch size scales with training batch

## Advanced Topics

### Custom Data Augmentation

Modify `get_augmentation_transforms()` in the training script to add custom augmentations specific to GTA V's characteristics.

### Learning Rate Scheduling

The script supports multiple schedulers:
- `linear`: Linearly decrease LR
- `cosine`: Cosine annealing (recommended)
- `poly`: Polynomial decay

### Mixed Precision Training

Enable with:
```python
# In train_detection.py
trainer = YOLO(...)
trainer.train(..., device=device, amp=True)
```

## Performance Benchmarks

### Training Speed (RTX 3070Ti)

- **YOLOv8n:** ~5 hours for 100 epochs
- **YOLOv8s:** ~8 hours for 100 epochs
- **YOLOv8m:** ~15 hours for 100 epochs

### Model Size & Speed

| Model | Size | FPS (GPU) | FPS (CPU) |
|-------|------|-----------|-----------|
| YOLOv8n | 3.2 MB | 45+ | 12 |
| YOLOv8s | 11.2 MB | 30+ | 6 |
| YOLOv8m | 49.7 MB | 15+ | 2 |

## References

- **[YOLOv8 Documentation](https://docs.ultralytics.com/)**
- **[Ultralytics Training Guide](https://docs.ultralytics.com/modes/train/)**
- **[Configuration Guide](../../configs/pipeline_config.yaml)**
- **[Evaluation Scripts](../evaluation/README.md)**

---

*Last updated: November 15, 2025*
