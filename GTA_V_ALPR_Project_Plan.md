# GTA V Automatic License Plate Recognition (ALPR) System

## 1. Project Overview

This document outlines the project plan for developing an Automatic License Plate Recognition (ALPR) system for the video game Grand Theft Auto V (GTA V). The project aims to create a proof-of-concept system that can detect and recognize license plate characters from in-game video footage or real-time gameplay. This project will be developed over a 1.5-month (6-week) period, leveraging pre-trained deep learning models for rapid prototyping and development.

**Primary Objectives:**
- Detect license plates in various in-game scenarios (day/night, different weather conditions).
- Recognize the characters on the detected license plates.
- Build a complete pipeline that takes a video frame as input and outputs the recognized license plate text.
- Evaluate the performance of the system.

## 2. Timeline

A high-level 6-week timeline for the project:

- **Week 1:** Project Setup, Environment Configuration, and Data Acquisition Strategy.
- **Week 2:** Implement and evaluate the License Plate Detection module.
- **Week 3:** Implement and evaluate the License Plate Recognition (OCR) module.
- **Week 4:** Integrate Detection and OCR modules, and implement a tracking algorithm.
- **Week 5:** Model Fine-tuning, Performance Evaluation, and Experimentation.
- **Week 6:** Finalize the project, code cleanup, and prepare the final report/presentation.

## 3. Technical Stack

### 3.1. Programming Language
- **Python 3.9+**

### 3.2. Core Libraries

- **Ultralytics (`ultralytics`):** A powerful, user-friendly library for training and deploying YOLO (You Only Look Once) models. We will use it for the license plate detection task. It provides easy access to pre-trained YOLOv8 models and a simple API for inference and training.
- **OpenCV (`opencv-python`):** The de-facto standard library for computer vision tasks. It will be used for image and video processing, such as reading frames from a video, cropping detected license plates, and image pre-processing before feeding them to the OCR model.
- **PaddlePaddle (`paddlepaddle`) & PaddleOCR:** An open-source deep learning platform. We will specifically use **PaddleOCR**, a part of the PaddlePaddle ecosystem, which provides a rich set of pre-trained models for text detection and recognition. It offers excellent performance for OCR tasks out-of-the-box.
- **Albumentations (`albumentations`):** A fast and flexible library for image augmentation. If we decide to fine-tune our models, this library will be used to create a more robust training dataset by applying various augmentations like brightness changes, motion blur, and rotations to simulate different in-game conditions.

### 3.3. Pre-trained Models

- **License Plate Detection:** We will start with a pre-trained YOLOv8 model fine-tuned for license plate detection. A good candidate from the Hugging Face Hub is **`yasirfaizahmed/license-plate-object-detection`**. This model provides a strong baseline for detecting license plates in images.
- **License Plate Recognition:** We will use the general-purpose OCR model provided by **PaddleOCR**. It is highly effective at recognizing text in a variety of fonts and conditions, making it suitable for the stylized license plates in GTA V.

### 3.4. Development Environment

- **Hardware:** An NVIDIA RTX 3070Ti GPU or Google Colab Pro will be sufficient for running and fine-tuning the models.
- **IDE:** Visual Studio Code.
- **Version Control:** Git and GitHub.

## 4. Methodology

The ALPR system will be built as a two-stage pipeline:

1.  **Detection:** A video frame is passed to the YOLOv8 model to detect the bounding box of any license plates.
2.  **Recognition:** The detected bounding box area is cropped from the frame, pre-processed, and then passed to the PaddleOCR engine to recognize the characters.

A tracking algorithm will be implemented to maintain the identity of a license plate across multiple frames, improving stability and reducing redundant processing.

## 5. Detailed Project Plan

### Week 1: Setup and Data Acquisition
- **Tasks:**
    - Set up the Python environment (`conda` or `venv`).
    - Install all required libraries: `ultralytics`, `opencv-python`, `paddlepaddle-gpu`, `paddleocr`, `albumentations`.
    - Set up a Git repository for version control.
    - Develop a strategy for data acquisition. This could involve:
        - Recording gameplay footage in GTA V under various conditions (day, night, rain).
        - Writing a script using `ScriptHookV` to automate data collection (e.g., spawning different vehicles, changing weather, and capturing screenshots).
- **Deliverable:** A configured development environment and a small dataset of initial test images/videos.

### Week 2: License Plate Detection
- **Tasks:**
    - Load the pre-trained YOLOv8 license plate detection model.
    - Write a script to run inference on the collected test images and videos.
    - Evaluate the initial performance of the detector. Identify its strengths and weaknesses (e.g., does it fail at night? At sharp angles?).
    - If necessary, start annotating a small custom dataset for future fine-tuning using **Label Studio**.
- **Deliverable:** A Python script that can detect and draw bounding boxes around license plates in images and videos.

### Week 3: License Plate Recognition (OCR)
- **Tasks:**
    - Set up the PaddleOCR engine.
    - Write a script that takes a cropped image of a license plate and outputs the recognized text.
    - Pre-process the cropped images to improve OCR accuracy (e.g., grayscale conversion, resizing, perspective correction).
    - Test the OCR module on manually cropped license plates from the dataset.
- **Deliverable:** A Python script that can perform OCR on license plate images.

### Week 4: Pipeline Integration and Tracking
- **Tasks:**
    - Combine the detection and recognition scripts into a single end-to-end pipeline.
    - Implement a simple object tracking algorithm (e.g., ByteTrack, which is integrated with YOLOv8, or a simple IOU-based tracker) to associate detections across frames. This prevents running the OCR model on the same plate in every single frame.
    - The pipeline should take a video frame and output the recognized text for each tracked plate.
- **Deliverable:** A complete ALPR pipeline that processes video frames in near real-time.

### Week 5: Fine-tuning and Evaluation
- **Tasks:**
    - Based on the performance evaluation, decide if fine-tuning is necessary.
    - If the detector is weak, fine-tune the YOLOv8 model on the custom annotated dataset. Use `albumentations` to augment the data.
    - If the OCR is weak, investigate fine-tuning options for PaddleOCR or experiment with more advanced image pre-processing techniques.
    - Systematically evaluate the entire pipeline's performance on a held-out test set. Measure metrics like detection accuracy (mAP) and recognition accuracy (Character Error Rate).
- **Deliverable:** An improved model (if fine-tuned) and a performance evaluation report.

### Week 6: Finalization and Report
- **Tasks:**
    - Refactor and clean up the codebase.
    - Add comments and documentation.
    - Prepare the final project report, detailing the architecture, methodology, results, and challenges.
    - Prepare a presentation and/or a video demonstration of the system in action.
- **Deliverable:** Final source code, project report, and presentation materials.

## 6. Deliverables
- A functional ALPR system capable of processing GTA V video footage.
- The complete, documented source code.
- A final project report detailing the project's design, implementation, and performance.
- A presentation/demo of the final system.

## 7. Risks and Mitigation
- **Risk:** Poor model performance in certain in-game conditions (e.g., heavy rain, low light).
  - **Mitigation:** Collect a diverse dataset that includes these challenging conditions. Use data augmentation to simulate them. Fine-tune the models on this augmented data.
- **Risk:** The stylized fonts on GTA V license plates are difficult for the OCR model to recognize.
  - **Mitigation:** Extensive pre-processing of the cropped license plate image. If issues persist, fine-tuning the OCR model on a synthetic dataset of GTA V-style license plates may be necessary.
- **Risk:** Real-time performance is not achieved.
  - **Mitigation:** Use a smaller, faster YOLO model (e.g., YOLOv8n). Optimize the pipeline by only running OCR when a new plate is detected or after a certain number of frames.

## 8. Datasets

The performance of the ALPR system is highly dependent on the quality and diversity of the data used for training and evaluation. We will require three main types of datasets:

### 8.1. Raw In-Game Footage
This dataset will consist of video clips recorded directly from GTA V gameplay. It is crucial to capture a wide variety of scenarios to ensure the final model is robust.

- **Content:** Gameplay recordings of vehicles from different angles, distances, and speeds.
- **Conditions to Capture:**
    - **Time of Day:** Day, night, dawn, and dusk.
    - **Weather:** Clear, rainy, foggy, and overcast.
    - **Lighting:** Direct sunlight, shadows, and artificial light from street lamps or headlights.
    - **Occlusion:** Partially obscured license plates (e.g., by other cars, objects, or dirt on the plate).
- **Source:** Automated data collection using scripts with **ScriptHookV**.

### 8.2. License Plate Detection Dataset (for YOLOv8)
This dataset will be used to fine-tune the YOLOv8 model for detecting license plates. It will be created by annotating frames extracted from the raw in-game footage.

- **Format:** Each image will have a corresponding label file containing the bounding box coordinates of the license plate(s). The format will be compatible with YOLO, where each line in the label file represents one object: `<class_id> <x_center> <y_center> <width> <height>`.
- **Annotation Tools:** We will use **Label Studio** to label the license plates.
- **Structure:** The dataset will be split into `train`, `validation`, and `test` sets. A `data.yaml` file will be created to define the dataset paths and class names, as required by YOLOv8.
    ```yaml
    train: ../datasets/lpr/train/images
    val: ../datasets/lpr/valid/images
    
    # Number of classes
    nc: 1
    
    # Class names
    names: ['license_plate']
    ```

### 8.3. License Plate Recognition Dataset (for OCR)
This dataset is for training or evaluating the PaddleOCR model. It will consist of cropped images of license plates.

- **Content:** Tightly cropped images of license plates, extracted from the detection dataset.
- **Preprocessing:** These images may be pre-processed (e.g., de-skewed, converted to grayscale, resized) to improve OCR performance.
- **Labeling:** A simple text file mapping image filenames to the corresponding license plate text will be created. For example:
    ```
    image_001.png	SA8821A
    image_002.png	46EEK827
    ```
This format is compatible with PaddleOCR's training pipeline if fine-tuning becomes necessary.
