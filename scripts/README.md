# Scripts Directory Overview

The `scripts/` directory now contains **only** the componentized CLI tools.
Every runnable entrypoint lives inside its functional subfolder—there are no
root-level wrapper shims anymore.

```
scripts/
├── annotation/      # Label Studio helpers, conversion utilities
├── data_ingestion/  # Frame extraction, dataset curation
├── diagnostics/     # Environment and health checks
├── evaluation/      # Metrics, reporting, visualization
└── inference/       # Image and video inference pipelines
```

### How to Run Scripts

1. Activate your project virtual environment.
2. Change into the relevant subdirectory, or reference the module path
	directly when invoking Python.

Examples:

```powershell
# Extract frames from a raw video capture
python scripts/data_ingestion/extract_frames.py --input raw.mp4 --output outputs/frames

# Run single-image detection
python scripts/inference/detect_image.py --image inputs/sample.jpg --output outputs/annotated.jpg

# Generate an evaluation report
python scripts/evaluation/generate_evaluation_report.py --results outputs/pipeline_results.csv
```

Each subdirectory provides a dedicated `README.md` covering the available
commands, required arguments, and expected outputs. Add new tooling inside the
appropriate folder so that usage remains discoverable and consistent.
