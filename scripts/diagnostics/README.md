# Diagnostics Scripts

Environment verification and health checks for the ALPR pipeline.

## Quick Start

### Verify GPU Setup
```powershell
python scripts/diagnostics/verify_gpu.py
```

### Check All Dependencies
```powershell
python scripts/diagnostics/check_dependencies.py
```

### Validate Configuration
```powershell
python scripts/diagnostics/validate_config.py configs/pipeline_config.yaml
```

## Key Scripts

**verify_gpu.py** - Check CUDA/GPU availability and key package versions.

**check_dependencies.py** - Verify all required packages are installed with correct versions.

**test_model_loading.py** - Validate detection and recognition models load correctly.

**validate_config.py** - Check pipeline configuration YAML syntax and parameters.

**check_dataset_quality.py** - Verify dataset completeness and image quality.

**system_info.py** - Print system information and environment details.

## Recommended Workflow

1. Check GPU: `python scripts/diagnostics/verify_gpu.py`
2. Check dependencies: `python scripts/diagnostics/check_dependencies.py`
3. Validate config: `python scripts/diagnostics/validate_config.py configs/pipeline_config.yaml`
4. Test models: `python scripts/diagnostics/test_model_loading.py`
5. Check dataset: `python scripts/diagnostics/check_dataset_quality.py --data datasets/lpr/data.yaml`

## Common Solutions

**GPU not detected:** Check NVIDIA driver with `nvidia-smi`, verify CUDA Toolkit installation

**Missing packages:** Run `pip install -r requirements.txt`

**Model loading fails:** Verify model paths exist, check file permissions

**Config errors:** Validate YAML syntax, check file paths in configuration

## References

- **[Configuration Guide](../../docs/configuration_guide.md)**
- **[Troubleshooting](../../docs/troubleshooting.md)**

---

*Last updated: November 15, 2025*
