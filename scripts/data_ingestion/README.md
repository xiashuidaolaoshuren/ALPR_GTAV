# Data Ingestion Scripts

Utilities for collecting raw footage, extracting frames, and preparing
metadata for the GTA V ALPR dataset.

- `extract_frames.py`: Core frame extractor used across the project.
- `batch_process_footage.py`: Opinionated wrapper that walks
	`outputs/raw_footage/` and extracts frames into `outputs/test_images/`.
- `generate_metadata.py`: Builds a starter `metadata.txt` file for the
	curated image set.
- `clean_test_images.py`: Removes duplicates/low quality artefacts from the
	curated test images.
- `data_collection_guide.md`: Cookbook outlining the end-to-end capture
	workflow.
