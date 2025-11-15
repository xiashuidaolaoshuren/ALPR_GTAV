---
marp: true
theme: default
---

# ALPR_GTAV

**Automatic License Plate Recognition in Grand Theft Auto V**

Chun Ning SO
2025-11-30

---

## Problem & Motivation

**Why ALPR in a video game?**

- **The Challenge:** Can we build a system to read license plates in a complex, simulated world with variable lighting, weather, and motion blur?
- **The Opportunity:** High-fidelity simulators like GTA V are powerful tools for generating vast amounts of training data for real-world AI, especially for autonomous vehicles.
- **This Project:** A case study on bridging the "sim-to-real" gap by training and validating a perception system in a synthetic environment.

---

## Technical Approach

A modular, three-stage pipeline for real-time processing:

![Pipeline](https://i.imgur.com/your-pipeline-diagram.png) 

**1. Detection (YOLOv8):** A fine-tuned model detects license plates in each frame.
**2. Recognition (PaddleOCR):** A robust OCR engine reads the characters from the detected plate.
**3. Tracking (ByteTrack):** An efficient algorithm tracks plates across frames to maintain identity and improve stability.

---

## Key Results: Detection Performance

Fine-tuning on in-game data was critical for success.

| Model     | Detection Rate | Avg. Confidence |
| :-------- | :------------- | :-------------- |
| Baseline  | 77.5%          | 0.63            |
| **Fine-Tuned** | **93.8%**      | **0.87**        |

- **Massive Improvement:** The fine-tuned model dramatically outperformed the baseline model trained on real-world data.
- **Robustness:** High performance was maintained across all in-game conditions (day/night, clear/rain).

---

## GUI Demonstration

An interactive GUI was developed using Streamlit for easy demonstration and testing.

![GUI Screenshot](https://i.imgur.com/your-gui-screenshot.png)

**Features:**
-   Video File Upload
-   Real-time video display with overlays
-   Interactive controls for pipeline parameters
-   Status and logging panels

*(A 30-second video demonstrating the GUI in action would be presented here.)*

---

## Key Achievements

- **High-Performance ALPR System:** Successfully built and validated a system that accurately detects and recognizes license plates in GTA V.
- **Bridged the Sim-to-Real Gap:** Proved that fine-tuning on a relatively small, custom-annotated synthetic dataset is highly effective.
- **Interactive GUI:** Developed a user-friendly Streamlit application for easy demonstration and interaction with the system.
- **Real-Time Processing:** The pipeline is optimized for real-time performance.

---

## Conclusion & Future Work

**Conclusion:**
This project successfully demonstrates a high-performance ALPR system for a complex synthetic environment, validating the use of simulators like GTA V for developing and testing real-world perception systems.

**Future Work:**
-   **Quantitative OCR Evaluation:** Measure Character Error Rate (CER) for a more rigorous assessment.
-   **Tracking Optimization:** Further analyze and optimize the tracking component for efficiency and stability.
-   **Expand Dataset:** Continue to expand the dataset with more edge cases.

---

## Q&A

**Thank you!**

**GitHub Repository:**
[github.com/xiashuidaolaoshuren/ALPR_GTAV](https://github.com/xiashuidaolaoshuren/ALPR_GTAV)
