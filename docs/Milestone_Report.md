# Project Milestone Report

**Project Title:** Automatic License Plate Recognition for Grand Theft Auto V (GTA V)
**Author:** So Chun Ning
**Date:** November 3, 2025


---

## 1. Introduction

Automatic License Plate Recognition (ALPR) is a technology that uses optical character recognition (OCR) on images to read vehicle registration plates. It is a mature field with widespread applications in law enforcement, electronic toll collection, and traffic management. However, applying ALPR systems to synthetic environments, such as video games, presents unique challenges and opportunities. The stylized graphics, variable lighting, motion blur, and non-standard fonts found in games like Grand Theft Auto V (GTA V) create a difficult test case for traditional ALPR models.

This project aims to develop a proof-of-concept ALPR system specifically tailored for the GTA V environment. The problem is interesting because it pushes the boundaries of computer vision models, forcing them to generalize to synthetic data that often differs significantly from real-world training sets. Previous work in this space has largely focused on real-world imagery. While many open-source models exist for object detection and OCR, few are fine-tuned for the specific domain of video game graphics.

Our overall plan is to construct a modular, two-stage pipeline. The first stage will use a state-of-the-art object detection model, YOLOv8, to locate license plates within a video frame. The second stage will employ the PaddleOCR engine to recognize the characters on the cropped plate. A tracking algorithm will be integrated between these stages to maintain plate identity across frames, optimizing performance by avoiding redundant OCR calls.

We expect our primary contribution to be a complete, end-to-end system demonstrating the feasibility of high-performance ALPR in a complex synthetic world. This work is interesting as it not only provides a novel tool for the gaming/modding community but also serves as a case study on adapting real-world computer vision systems to simulated environments, which is increasingly relevant for training and testing autonomous systems.

## 2. Related Work

Our project builds upon decades of research in object detection and optical character recognition. The works cited below represent the foundational technologies and state-of-the-art approaches that inform our system design.

1.  **YOLO (You Only Look Once)**: The YOLO family of models, first introduced by Redmon et al. (2016), revolutionized real-time object detection. Unlike two-stage detectors, YOLO treats object detection as a single regression problem, making it exceptionally fast. Our project uses **YOLOv8**, the latest iteration from Ultralytics, which offers a balance of high accuracy and performance.

2.  **PaddleOCR**: Developed by Baidu, PaddleOCR is a comprehensive OCR toolkit that includes a variety of text detection and recognition models. The recognition model is often based on the CRNN (Convolutional Recurrent Neural Network) architecture, as proposed by Shi et al. (2017). This architecture combines a CNN for feature extraction, an RNN for sequence modeling, and a CTC loss function for transcription, making it highly effective for text of varying lengths.

3.  **ByteTrack**: Tracking-by-detection is a common paradigm for multi-object tracking. Zhang et al. (2022) introduced ByteTrack, a simple yet effective algorithm that addresses the issue of objects being occluded. It retains low-confidence detections and uses their similarity to re-associate them with existing tracks, reducing fragmentation and improving tracking continuity. We plan to use ByteTrack, which is conveniently integrated with YOLOv8.

4.  **Deep Learning for License Plate Recognition**: Many works have specifically addressed ALPR using deep learning. For instance, Laroca et al. (2018) presented a comprehensive study comparing different deep learning approaches for ALPR, demonstrating the effectiveness of two-stage systems (detection then recognition). Their work validates our architectural choice.

5.  **CR-NET**: Zherzdev and Gruzdev (2019) proposed CR-NET, a compact and efficient model for license plate recognition that performs recognition directly on the full frame without prior detection. While interesting, this end-to-end approach is often less flexible than a modular two-stage pipeline, which allows for independent optimization of detection and recognition.

6.  **Synthetic Data for Training**: The use of synthetic data to train deep learning models is a well-established practice. Papers by Shrivastava et al. (2017) on "SimGAN" and others have shown that synthetic data, when refined, can significantly improve model robustness. Our project, while not generating synthetic data, applies models to a purely synthetic *environment*, providing insights into this domain gap.

7.  **Albumentations**: Buslaev et al. (2020) created Albumentations, a fast and flexible library for image augmentation. Data augmentation is critical for training robust models, especially when the initial dataset is small. We plan to use this library to simulate various in-game conditions (e.g., motion blur, different lighting) if we decide to fine-tune our models.

8.  **Label Studio**: Label Studio is an open-source data labeling tool that supports a wide variety of data types. Its flexibility in defining labeling configurations is crucial for our project, as it allows us to capture not only bounding boxes but also custom metadata like the `readability` attribute for each plate.

9.  **Attention-based OCR**: More recent OCR models have incorporated attention mechanisms. For example, work by Li et al. (2019) on "SAR" (Show, Attend and Read) demonstrates how attention can improve recognition of irregular or distorted text, which may be relevant for handling the stylized fonts in GTA V.

10. **Open Datasets for ALPR**: The availability of public datasets like the one from Hsu et al. (2013) has been crucial for benchmarking ALPR systems. While these datasets are based on real-world images, they provide a baseline for understanding the complexity of the ALPR task. Our work involves creating a new, domain-specific dataset from the GTA V environment.

### References

-   Buslaev, A., et al. (2020). "Albumentations: Fast and Flexible Image Augmentations." *Information*, 11(2), 125.
-   Hsu, G.-S., et al. (2013). "Application-Oriented License Plate Recognition." *IEEE Transactions on Intelligent Transportation Systems*, 14(3), 1410-1419.
-   Laroca, R., et al. (2018). "A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector." *2018 International Joint Conference on Neural Networks (IJCNN)*.
-   Li, H., et al. (2019). "Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition." *AAAI Conference on Artificial Intelligence*.
-   Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
-   Shi, B., et al. (2017). "An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(11), 2298-2304.
-   Shrivastava, A., et al. (2017). "Learning from Simulated and Unsupervised Images through Adversarial Training." *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
-   Zhang, Y., et al. (2022). "ByteTrack: A Simple and Effective Data Association Method for Multi-Object Tracking." *arXiv preprint arXiv:2110.06864*.
-   Zherzdev, P., & Gruzdev, A. (2019). "CR-NET: A Simple Approach to Vehicle License Plate Recognition." *arXiv preprint arXiv:1910.05956*.
-   PaddleOCR, Baidu. [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 3. Data

The performance of our system is critically dependent on the quality and diversity of the data collected from GTA V. Our data strategy is multi-faceted, involving the collection of raw footage and the creation of two specialized datasets for detection and recognition.

**Data Source**: All data is sourced directly from Grand Theft Auto V gameplay. We have developed a data collection strategy that involves recording footage under a wide variety of conditions to ensure our models are robust. This includes:
-   **Time of Day**: Day, night.
-   **Weather**: Clear, rainy.
-   **Location**: Various locations within the game.
-   **Vehicle Types**: Different types of vehicles (e.g., sedans, SUVs, trucks).
-   **Angle**: Various angles relative to the camera.

**1. Raw In-Game Footage**: This is our primary source material, consisting of video clips recorded during gameplay. A full dataset of near 900 images and clips has already been collected to facilitate development and testing.

**2. License Plate Detection Dataset (YOLO Format)**: This dataset is being created to fine-tune the YOLOv8 detection model.
-   **Annotation**: We are using Label Studio to manually annotate frames extracted from the raw footage.
-   **Format**: Annotations are saved in the YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
-   **Key Innovation**: We have customized the labeling interface to include a mandatory **`readability`** attribute for each bounding box. Annotators must classify each plate as `clear`, `blurred`, or `occluded`. This allows us to train the detection model on all visible plates (improving its robustness) while filtering for only `clear` plates to create a high-quality dataset for the subsequent OCR task.

**3. License Plate Recognition Dataset (OCR Format)**: This dataset will be used to evaluate and potentially fine-tune the OCR model.
-   **Content**: It will consist of tightly cropped images of license plates.
-   **Generation**: This dataset will be generated from the detection dataset by extracting the bounding box regions of plates marked as `clear`.
-   **Format**: Labels will be stored in a simple text file mapping each image filename to the ground-truth plate text (e.g., `image_001.png\t12ABC345`), which is the standard format for PaddleOCR. We have identified that GTA V license plates follow a consistent `00AAA000` format (`\d{2}[A-Z]{3}\d{3}$`), which will be crucial for validation.

## 4. Approach

Our methodology is centered around a modular, two-stage pipeline that can be developed and evaluated independently before final integration.

**Baseline**: Our primary baseline for detection is a pre-trained YOLOv8 model (`yasirfaizahmed/license-plate-object-detection`) fine-tuned on real-world license plates. We will evaluate its zero-shot performance on GTA V data to gauge the domain gap between real and synthetic imagery. For recognition, the baseline is the general-purpose English recognition model provided by PaddleOCR.

**Stage 1: License Plate Detection**
-   **Model**: We will use the **YOLOv8** object detection model provided by the `ultralytics` library.
-   **Process**: Each video frame is passed to the model, which returns a set of bounding boxes corresponding to potential license plates. We will use a confidence threshold to filter out weak detections.

**Stage 2: License Plate Recognition**
-   **Model**: We will use the **PaddleOCR** engine.
-   **Preprocessing**: The cropped plate image from Stage 1 will be preprocessed. This may include grayscale conversion, resizing to an optimal dimension, and contrast enhancement (e.g., using CLAHE) to improve OCR accuracy.
-   **Post-processing**: Since PaddleOCR may detect multiple lines of text on a plate (e.g., the state name and the plate number), we will implement a rule-based filter to select the correct text. This filter will:
    1.  Validate candidates against the known GTA V plate regex (`^\d{2}[A-Z]{3}\d{3}$`).
    2.  Score the remaining candidates using a formula that considers OCR confidence, the text's bounding box height, and its length: $score = p \cdot h \cdot \min(\frac{L}{8}, 1)$.
    3.  Select the candidate with the highest score as the final result.

**Stage 3: Tracking**
-   **Algorithm**: We will use **ByteTrack**, which is integrated into the YOLOv8 framework.
-   **Purpose**: Tracking associates detections across frames, which is crucial for two reasons:
    1.  **Efficiency**: It prevents the computationally expensive OCR model from running on the same plate in every single frame.
    2.  **Stability**: It allows for temporal smoothing of results, as a plate's text can be aggregated over several frames to improve confidence.

All parameters for these stages (e.g., thresholds, model paths, regex patterns) will be managed centrally in an independent config file `pipeline_config.yaml` to ensure modularity and ease of experimentation.

## 5. Preliminary Results

As of this milestone, we have not yet trained or fine-tuned any deep learning models. Our efforts have been focused on establishing a robust project foundation, developing the core architecture, and implementing the initial stages of the pipeline.

**Accomplishments:**

1.  **Project Scaffolding and Environment Setup**: We have successfully set up the complete project directory structure, a Python virtual environment with all necessary dependencies (`ultralytics`, `paddleocr`, `opencv-python`, etc.), and a version-controlled Git repository.

2.  **Data Acquisition and Annotation Infrastructure**:
    -   A detailed data acquisition strategy has been documented.
    -   An initial test dataset of over 50 images and video clips has been collected.
    -   Label Studio has been configured with a custom labeling interface, including the critical `readability` attribute. This infrastructure is ready for large-scale annotation.

3.  **Detection Module Implementation**:
    -   The core logic for the detection module has been implemented. This includes scripts to load the pre-trained YOLOv8 model and run inference on both single images and video files.
    -   Utility functions for visualizing bounding boxes on output media have been created.

**Anticipated Next Steps:**
1.  **Systematic Evaluation**: We will continue to evaluate the end-to-end performance of the pipeline on our test dataset, measuring detection accuracy (mAP) and recognition accuracy (Character Error Rate).
2.  **Annotation and Fine-Tuning**: We may proceed with annotating a larger dataset. Based on the evaluation results, we will decide whether to continue fine-tuning the YOLOv8 detector or the PaddleOCR recognizer.
3. **GUI Development**: We plan to develop a simple graphical user interface (GUI) to facilitate easier interaction with the ALPR system, allowing users to upload videos and view results in real-time.

**Obstacles Encountered**:
The primary obstacle so far has been theoretical rather than technical: anticipating the challenges of applying models trained on real-world data to a synthetic environment. This led us to proactively design robust data collection and annotation strategies (like the `readability` attribute) and a sophisticated OCR post-processing filter. We anticipate that the domain gap will still be a challenge, but our current approach is designed to mitigate it effectively.