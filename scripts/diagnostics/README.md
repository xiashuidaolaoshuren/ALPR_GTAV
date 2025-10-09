# Diagnostics Scripts

Health checks for the development environment and dataset quality to
ensure the ALPR pipeline runs reliably.

- `verify_gpu.py`: Confirms CUDA availability for PyTorch and PaddlePaddle
	while checking that key packages are installed with expected versions.
- `check_dataset_quality.py`: Validates dataset completeness, diversity,
	and basic image quality prior to labeling.
- `test_model_loading.py`: Sanity-checks that the YOLOv8 and PaddleOCR models
	load correctly with the configured weights and catch missing checkpoint issues
	before running inference jobs.
