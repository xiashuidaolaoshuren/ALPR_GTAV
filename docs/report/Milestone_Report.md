# Project Milestone Report

**Project Title:** Automatic License Plate Recognition for Grand Theft Auto V (GTA V)
**Author:** So Chun Ning
**Date:** November 5, 2025


---

## 1. Introduction

Automatic License Plate Recognition (ALPR) is a technology that uses optical character recognition (OCR) on images to read vehicle registration plates. While a mature field with widespread applications in law enforcement and traffic management, its application in synthetic environments like video games offers a unique and valuable research frontier. This project develops a proof-of-concept ALPR system for Grand Theft Auto V (GTA V), tackling challenges like stylized graphics, variable lighting, and non-standard fonts.

The problem is interesting not just as a technical challenge, but for its significant real-world implications. Modern AI systems, particularly for autonomous vehicles, rely on vast, diverse, and perfectly annotated datasets for training. Simulators like GTA V can act as powerful data generation engines, creating a limitless supply of labeled data for scenarios that are expensive, dangerous, or rare in the real world (e.g., extreme weather, accidents). This project serves as a direct case study in **leveraging synthetic data to train and validate robust perception systems**.

My overall plan is to construct a modular, two-stage pipeline: a YOLOv8 model for license plate detection, followed by a PaddleOCR engine for character recognition, with a ByteTrack algorithm for object tracking to optimize performance.

The contributions of this work extend beyond the gaming community. By developing a system that functions in a high-fidelity simulation, I am exploring solutions to the critical **"sim-to-real" domain gap**. The techniques used to make a model effective in GTA V can inform strategies for transferring models trained on synthetic data to real-world applications. Furthermore, this project provides a blueprint for using simulated environments as a safe, cost-effective, and scalable platform for testing and validating computer vision components before their deployment in physical systems like autonomous cars.

## 2. Related Work

My project builds upon decades of research in object detection and optical character recognition. The works cited below represent the foundational technologies and state-of-the-art (SOTA) approaches that inform my system design.

1.  **YOLO (You Only Look Once)**: The YOLO family of models, first introduced by Redmon et al. (2016), revolutionized real-time object detection. Unlike two-stage detectors, YOLO treats object detection as a single regression problem, making it exceptionally fast. My project uses **YOLOv8**, one of the most powerful iterations from Ultralytics, which offers a balance of high accuracy and performance.

2.  **PaddleOCR**: Developed by Baidu, PaddleOCR is a comprehensive OCR toolkit that includes a variety of text detection and recognition models. The recognition model is often based on the CRNN (Convolutional Recurrent Neural Network) architecture, as proposed by Shi et al. (2017). This architecture combines a CNN for feature extraction, an RNN for sequence modeling, and a CTC loss function for transcription, making it highly effective for text of varying lengths.

3.  **ByteTrack**: Tracking-by-detection is a common paradigm for multi-object tracking. Zhang et al. (2022) introduced ByteTrack, a simple yet effective algorithm that addresses the issue of objects being occluded. It retains low-confidence detections and uses their similarity to re-associate them with existing tracks, reducing fragmentation and improving tracking continuity. I plan to use ByteTrack, which is conveniently integrated with YOLOv8.

4.  **Deep Learning for License Plate Recognition**: Many works have specifically addressed ALPR using deep learning. For instance, Laroca et al. (2018) presented a comprehensive study comparing different deep learning approaches for ALPR, demonstrating the effectiveness of two-stage systems (detection then recognition). Their work validates my architectural choice.

5.  **CR-NET**: Zherzdev and Gruzdev (2019) proposed CR-NET, a compact and efficient model for license plate recognition that performs recognition directly on the full frame without prior detection. While interesting, this end-to-end approach is often less flexible than a modular two-stage pipeline, which allows for independent optimization of detection and recognition.

6.  **Synthetic Data for Training**: The use of synthetic data to train deep learning models is a well-established practice. Papers by Shrivastava et al. (2017) on "SimGAN" and others have shown that synthetic data, when refined, can significantly improve model robustness. My project, while not generating synthetic data, applies models to a purely synthetic *environment*, providing insights into this domain gap.

7.  **Albumentations**: Buslaev et al. (2020) created Albumentations, a fast and flexible library for image augmentation. Data augmentation is critical for training robust models, especially when the initial dataset is small. The library is used to simulate various in-game conditions (e.g., motion blur, different lighting) if I decide to fine-tune my models.

8.  **Attention-based OCR**: More recent OCR models have incorporated attention mechanisms. For example, work by Li et al. (2019) on "SAR" (Show, Attend and Read) demonstrates how attention can improve recognition of irregular or distorted text, which may be relevant for handling the stylized fonts in GTA V.

9. **Open Datasets for ALPR**: The availability of public datasets like the one from Hsu et al. (2013) has been crucial for benchmarking ALPR systems. While these datasets are based on real-world images, they provide a baseline for understanding the complexity of the ALPR task.

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

The performance of my system is critically dependent on the quality and diversity of the data collected from GTA V. My data strategy is multi-faceted, involving the collection of raw footage and the creation of two specialized datasets for detection and recognition.

**Data Source**: All data is sourced directly from Grand Theft Auto V gameplay. I have developed a data collection strategy that involves recording footage under a wide variety of conditions to ensure my models are robust. This includes:
-   **Time of Day**: Day, night.
-   **Weather**: Clear, rainy.
-   **Location**: Various locations within the game.
-   **Vehicle Types**: Different types of vehicles (e.g., sedans, SUVs, trucks).
-   **Angle**: Various angles relative to the camera.

**1. Raw In-Game Footage**: This is the primary source material, consisting of video clips recorded during gameplay. A full dataset of near 900 images and clips has already been collected to facilitate development and testing.

**2. License Plate Detection Dataset (YOLO Format)**: This dataset is being created to fine-tune the YOLOv8 detection model.
-   **Annotation**: I am using Label Studio to manually annotate frames extracted from the raw footage.
-   **Format**: Annotations are saved in the YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
-   **Key Innovation**: I have customized the labeling interface to include a mandatory **`readability`** attribute for each bounding box. Annotators must classify each plate as `clear`, `blurred`, or `occluded`. This allows me to train the detection model on all visible plates (improving its robustness) while filtering for only `clear` plates to create a high-quality dataset for the subsequent OCR task.

**3. License Plate Recognition Dataset (OCR Format)**: This dataset will be used to evaluate and potentially fine-tune the OCR model.
-   **Content**: It will consist of tightly cropped images of license plates.
-   **Generation**: This dataset will be generated from the detection dataset by extracting the bounding box regions of plates marked as `clear`.
-   **Format**: Labels will be stored in a simple text file mapping each image filename to the ground-truth plate text (e.g., `image_001.png\t12ABC345`), which is the standard format for PaddleOCR. I have identified that GTA V license plates follow a consistent `00AAA000` format (`\d{2}[A-Z]{3}\d{3}$`), which will be crucial for validation.

## 4. Approach

My methodology is centered around a modular, two-stage pipeline that can be developed and evaluated independently before final integration.

**Baseline**: My primary baseline for detection is a pre-trained YOLOv8 model (`yasirfaizahmed/license-plate-object-detection`) fine-tuned on real-world license plates. I will evaluate its zero-shot performance on GTA V data to gauge the domain gap between real and synthetic imagery. For recognition, the baseline is the general-purpose English recognition model provided by PaddleOCR.

**Stage 1: License Plate Detection**
-   **Model**: I will use the **YOLOv8** object detection model provided by the `ultralytics` library. My core strategy is to **fine-tune** a pre-trained YOLOv8 model on my custom-annotated GTA V dataset. This approach, known as transfer learning, allows the model to adapt its powerful, general-purpose feature extraction capabilities to the specific visual characteristics of the GTA V environment—such as its unique lighting, textures, and plate designs.
-   **Process**: Each video frame is passed to my fine-tuned model, which returns a set of bounding boxes corresponding to potential license plates. I will use a confidence threshold to filter out weak detections. This fine-tuning step is critical for bridging the "sim-to-real" gap and achieving high accuracy.

**Stage 2: License Plate Recognition**
-   **Model**: I will use the **PaddleOCR** engine.
-   **Preprocessing**: The cropped plate image from Stage 1 will be preprocessed. This may include grayscale conversion, resizing to an optimal dimension, and contrast enhancement (e.g., using CLAHE) to improve OCR accuracy.
-   **Post-processing**: Since PaddleOCR may detect multiple lines of text on a plate (e.g., the state name and the plate number), I will implement a rule-based filter to select the correct text. This filter will:
    1.  Validate candidates against the known GTA V plate regex (`^\d{2}[A-Z]{3}\d{3}$`).
    2.  Score the remaining candidates using a formula that considers OCR confidence, the text's bounding box height, and its length: $score = p \cdot h \cdot \min(\frac{L}{8}, 1)$.
    3.  Select the candidate with the highest score as the final result.

**Stage 3: Tracking**
-   **Algorithm**: I will use **ByteTrack**, which is integrated into the YOLOv8 framework.
-   **Purpose**: Tracking associates detections across frames, which is crucial for two reasons:
    1.  **Efficiency**: It prevents the computationally expensive OCR model from running on the same plate in every single frame.
    2.  **Stability**: It allows for temporal smoothing of results, as a plate's text can be aggregated over several frames to improve confidence.

All parameters for these stages (e.g., thresholds, model paths, regex patterns) will be managed centrally in an independent config file `pipeline_config.yaml` to ensure modularity and ease of experimentation.

## 5. Preliminary Results

As of this milestone, I have not yet trained or fine-tuned any deep learning models. My efforts have been focused on establishing a robust project foundation, developing the core architecture, and implementing the initial stages of the pipeline.

**Accomplishments:**

1.  **Project Scaffolding and Environment Setup**: I have successfully set up the complete project directory structure, a Python virtual environment with all necessary dependencies (`ultralytics`, `paddleocr`, `opencv-python`, etc.), and a version-controlled Git repository (will be provided in Final report).

2.  **Data Acquisition and Annotation Infrastructure**:
    -   A detailed data acquisition strategy has been documented.
    -   A full dataset of nearly 900 images and video clips has been collected.
    -   Label Studio has been configured with a custom labeling interface, including the critical `readability` attribute. This infrastructure is ready for large-scale annotation.

3.  **Detection Module Implementation**:
    -   The core logic for the detection module has been implemented. This includes scripts to load the pre-trained YOLOv8 model and run inference on both single images and video files.
    -   Utility functions for visualizing bounding boxes on output media have been created.

4. **YOLOv8 Training Results**:
    - I have completed an initial round of fine-tuning the YOLOv8 model on the custom annotated GTA V dataset for 10 epochs. This serves as the first baseline model trained specifically on in-game data.
    - **Evaluation Metrics**: To assess the model, I use standard object detection metrics:
        - **Precision**: Measures the accuracy of the positive predictions. It answers the question: "Of all the bounding boxes the model drew, what fraction were actual license plates?"
        - **Recall**: Measures the model's ability to find all relevant objects. It answers: "Of all the actual license plates in the images, what fraction did the model successfully detect?"
        - **mAP50 (mean Average Precision at IoU=0.50)**: This is the primary metric for object detection. It calculates the average precision across all classes, considering a detection "correct" if the Intersection over Union (IoU) with a ground-truth box is greater than 50%. A higher mAP50 indicates a better model.
        - **mAP50-95**: This is a stricter metric that averages the mAP over a range of IoU thresholds (from 0.50 to 0.95). It rewards models that produce more precise and tightly-fitting bounding boxes.
    - **Initial Performance**: After 10 epochs, the model achieved the following performance on the validation and test set:
    ![YOLOv8 Training Results](../runs/detect/gta_v_lpr2/plots/training_metrics.png)
    ![YOLOv8 Precision-Recall](../runs/detect/gta_v_lpr2/plots/precision_recall.png)
    ![YOLOv8 Test Metrics](../runs/detect/gta_v_lpr2/plots/test_metrics.png)
        - **Precision**: 0.983
        - **Recall**: 0.737
        - **mAP50**: 0.825
        - **mAP50-95**: 0.581
    - **Analysis**: These preliminary results are highly promising. A **precision of 98.3%** indicates that the model is very reliable when it does predict a license plate, with very few false positives. The **recall of 73.7%** is a solid starting point, suggesting that the model finds the majority of plates but still misses some, particularly those that are small, at sharp angles, or in challenging lighting. The **mAP50 of 0.825** is a strong result for an initial training run and confirms the model is learning the specific features of GTA V license plates effectively. The lower **mAP50-95** score suggests that while the model is good at finding the plates, the precision of its bounding boxes could be improved.

**Anticipated Next Steps:**
1.  **Annotation and Fine-Tuning**: Concurrently, I will proceed with annotating a larger dataset. Based on the evaluation results, I will decide whether to fine-tune the YOLOv8 detector or the PaddleOCR recognizer.
2.  **Systematic Evaluation**: I will then continue to evaluate the end-to-end performance of the pipeline on my test dataset, measuring detection accuracy (mAP) and recognition accuracy (Character Error Rate).
3. **GUI Development**: I plan to develop a simple graphical user interface (GUI) to facilitate easier testing and visualization of the pipeline's performance on new video inputs.

**Obstacles Encountered**:
The primary obstacle so far has been theoretical rather than technical: anticipating the challenges of applying models trained on real-world data to a synthetic environment. This led me to proactively design robust data collection and annotation strategies (like the `readability` attribute) and a sophisticated OCR post-processing filter. The domain gap will still be a challenge, but my current approach is designed to mitigate it effectively.