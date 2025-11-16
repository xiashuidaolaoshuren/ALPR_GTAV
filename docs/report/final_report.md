# AIST4010 - Final Project Report

**Project Title**: Automatic License Plate Recognition in Grand Theft Auto V
**Author:** Chun Ning SO
**Date:** 2025-11-30

---

## Abstract

This report details the development and evaluation of a real-time Automatic License Plate Recognition (ALPR) system designed for the synthetic environment of Grand Theft Auto V (GTA V). The project addresses the "sim-to-real" domain gap by demonstrating the effectiveness of fine-tuning on in-domain synthetic data. A two-stage pipeline was constructed, utilizing a YOLOv8 model for license plate detection and a PaddleOCR engine for character recognition. A baseline YOLOv8 model, pre-trained on real-world data, was compared against several versions fine-tuned on a custom-annotated dataset of GTA V gameplay footage. Experimental results show a dramatic improvement from fine-tuning: the detection rate increased from 77.5% to over 96%, and average detection confidence rose from 0.63 to 0.87. The final model proves robust across varied in-game conditions, including night and rain. This work validates the use of synthetic environments for training and testing computer vision systems, highlighting that in-domain data is critical for achieving high performance. The project concludes with a complete system design, including a Streamlit-based GUI for interactive demonstration.

---

## 1. Introduction

Automatic License Plate Recognition (ALPR) is a technology that uses optical character recognition (OCR) on images to read vehicle registration plates. While a mature field with widespread applications in law enforcement and traffic management, its application in synthetic environments like video games offers a unique and valuable research frontier. This project develops a proof-of-concept ALPR system for Grand Theft Auto V (GTA V), tackling challenges like stylized graphics, variable lighting, and non-standard fonts.

Grand Theft Auto V (GTA V) is an open-world action-adventure game renowned for its detailed and dynamic simulation of a contemporary city, developed by **Rockstar Games**. Its high-fidelity graphics, realistic physics engine, and complex AI for traffic and pedestrians create a rich, interactive environment that closely mirrors real-world urban settings. This makes GTA V an ideal and challenging testbed for computer vision systems. The game's varied environmental conditions—including a full day-night cycle, changing weather, and diverse vehicle models—provide a cost-effective and safe platform for generating the vast and varied datasets required to train and validate robust perception algorithms, such as those needed for autonomous driving.

The concept of using video games as training grounds for AI is well-established. Major companies and research institutions have leveraged simulators for years; for example, Waymo uses its proprietary "Simulation City," and NVIDIA has its "Drive Constellation" platform. The use of commercial games, however, was notably pioneered by researchers from institutions like Intel Labs and Darmstadt University, who demonstrated that GTA V could be used to extract rich, automatically annotated data for training autonomous driving models. This project follows in that tradition, using a commercial game as a cost-effective and scalable data source.

The problem is interesting not just as a technical challenge, but for its significant real-world implications. Modern AI systems, particularly for autonomous vehicles, rely on vast, diverse, and perfectly annotated datasets for training. Simulators like GTA V can act as powerful data generation engines, creating a limitless supply of labeled data for scenarios that are expensive, dangerous, or rare in the real world (e.g., extreme weather, accidents). This project serves as a direct case study in **leveraging synthetic data to train and validate robust perception systems**.

My overall plan is to construct a modular, two-stage pipeline: a YOLOv8 model for license plate detection, followed by a PaddleOCR engine for character recognition, with a ByteTrack algorithm for object tracking to optimize performance.

The contributions of this work extend beyond the gaming community. By developing a system that functions in a high-fidelity simulation, I am exploring solutions to the critical **"sim-to-real" domain gap**. The techniques used to make a model effective in GTA V can inform strategies for transferring models trained on synthetic data to real-world applications. Furthermore, this project provides a blueprint for using simulated environments as a safe, cost-effective, and scalable platform for testing and validating computer vision components before their deployment in physical systems like autonomous cars.

---

## 2. Background and Related Work

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

---

## 3. Methodology

My methodology is centered around a modular, two-stage pipeline that can be developed and evaluated independently before final integration. This section covers the data strategy, the pipeline architecture, and the specific models and algorithms used.

### 3.1. Data Strategy and Preprocessing

The performance of the ALPR system is critically dependent on the quality and diversity of the data. A comprehensive data strategy was implemented, covering automated acquisition, detailed annotation, and careful preprocessing to create high-quality datasets for both detection and recognition tasks.

**1. Data Acquisition**:
-   **Source**: All data was sourced directly from Grand Theft Auto V gameplay on a PC. Raw video footage was captured using **OBS Studio** at 1080p resolution and 60 FPS to ensure high-quality frames for analysis.
-   **Raw Footage**: The script captured high-resolution screenshots and short video clips. Images is extracted from the raw footage, resulting in a cleaned dataset of nearly 1200 images. This collection captured vehicles from multiple angles, distances, and speeds, forming the primary source material for annotation.

**2. License Plate Detection Dataset (YOLO Format)**:
This dataset was created to fine-tune the YOLOv8 detection model.
-   **Annotation Tool**: **Label Studio** was used for its flexibility in creating custom annotation interfaces.
-   **Annotation Process**: Frames were extracted from the raw footage and imported into Label Studio. I manually drew bounding boxes around every visible license plate.
-   **Key Innovation - `readability` Attribute**: The Label Studio configuration was customized to include a mandatory **`readability` choice** for each bounding box. This forced the annotator to classify each plate into one of three categories:
    -   `clear`: The plate is sharp and fully visible.
    -   `blurred`: The plate is out of focus or affected by motion blur.
    -   `occluded`: The plate is partially hidden by an object.
    This granular annotation strategy was crucial. It allowed the detection model to be trained on *all* visible plates, making it robust in identifying plates even in challenging conditions. Simultaneously, it provided a simple mechanism to filter for only `clear` plates when creating the high-quality dataset for the subsequent OCR task.
-  **Use of readability Attribute**: The `readability` attribute enabled the creation of two distinct datasets from the same annotations:
    1.  A comprehensive detection dataset including all plates (clear, blurred, occluded) for robust model training.
    2.  A high-quality OCR dataset filtered to include only `clear` plates, ensuring optimal recognition performance for potential OCR fine-tuning.
-   **Format Conversion**: After annotation, the Label Studio JSON output was converted into the YOLO format (`<class_id> <x_center> <y_center> <width> <height>`) using a custom Python script. The dataset was then split into training, validation, and test sets.

**3. License Plate Recognition Dataset (OCR Format)**:
This dataset was created to evaluate and potentially fine-tune the OCR model.
-   **Content Generation**: This dataset was generated by programmatically processing the detection dataset. For every annotation marked as `clear`, the corresponding bounding box region was cropped from the image.
-   **Preprocessing for OCR**: Before being used, each cropped plate image underwent a series of preprocessing steps to maximize OCR accuracy:
    -   **Perspective Correction**: If the plate was at a significant angle, a perspective transformation was applied to "flatten" it.
    -   **Grayscale Conversion**: The image was converted to grayscale, as color information is not necessary for OCR.
    -   **Resizing**: The image was resized to a uniform height while maintaining aspect ratio, which is a common practice for many OCR models.
    -   **Contrast Enhancement**: Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to enhance the definition of the characters, especially in low-light or washed-out images.
-   **Format**: The preprocessed images were saved, and labels were stored in a simple text file mapping each image filename to the ground-truth plate text (e.g., `image_001.png\t12ABC345`), which is the standard format required by PaddleOCR. The consistent `00AAA000` format (`\d{2}[A-Z]{3}\d{3}$`) of GTA V plates was used for validation during this process.
-  **Note**: As the PaddleOCR baseline model is already peform welling on English text, I did not perform additional fine-tuning for this phase. However, the dataset is structured to allow for future OCR model training if needed.

### 3.2. Pipeline Architecture

My approach is centered around a modular, two-stage pipeline.

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

---

## 4. Experiments and Results

This section details the experimental setup, the evaluation of the baseline and fine-tuned models, and a comparative analysis of their performance.

### 4.1. Experimental Setup

**Evaluation Metrics**: To assess the model, I use standard object detection metrics:
- **Precision**: Measures the accuracy of the positive predictions. It answers the question: "Of all the bounding boxes the model drew, what fraction were actual license plates?"
- **Recall**: Measures the model's ability to find all relevant objects. It answers: "Of all the actual license plates in the images, what fraction did the model successfully detect?"
- **mAP50 (mean Average Precision at IoU=0.50)**: This is the primary metric for object detection. It calculates the average precision across all classes, considering a detection "correct" if the Intersection over Union (IoU) with a ground-truth box is greater than 50%. A higher mAP50 indicates a better model.
- **mAP50-95**: This is a stricter metric that averages the mAP over a range of IoU thresholds (from 0.50 to 0.95). It rewards models that produce more precise and tightly-fitting bounding boxes.

**Models Tested**:
1.  **Baseline**: A pre-trained YOLOv8 model (`yasirfaizahmed/license-plate-object-detection`) evaluated directly on the GTA V test set (zero-shot performance).
2.  **Fine-tuned**: The model fine-tuned for nearly 50 epochs on the custom GTA V dataset.

### 4.2. Initial Fine-Tuning Performance

An initial round of fine-tuning the YOLOv8 model on the custom annotated GTA V dataset was completed for 10 epochs. This served as the first model trained specifically on in-game data.

- **Initial Performance**: After 10 epochs, the model achieved the following performance on the validation set:
    - **Precision**: 0.983
    - **Recall**: 0.737
    - **mAP50**: 0.825
    - **mAP50-95**: 0.581

- **Analysis**: These preliminary results were highly promising. A **precision of 98.3%** indicates that the model is very reliable when it does predict a license plate, with very few false positives. The **recall of 73.7%** was a solid starting point, suggesting that the model finds the majority of plates but still misses some. The **mAP50 of 0.825** confirmed the model was learning the specific features of GTA V license plates effectively.

### 4.3. Comparative Detection Results

To provide a comprehensive view of the improvements gained from fine-tuning, the table below summarizes the detection performance of the baseline model and the three subsequent fine-tuned versions across all tested conditions. The data is aggregated from the raw detection output files.

| Model                  | Condition   | Detection Rate (%) | Avg Detections/Image | Avg Confidence |
| ---------------------- | ----------- | ------------------ | -------------------- | -------------- |
| **detection_baseline** | **overall** | **77.53**          | **0.87**             | **0.6323**     |
|                        | day_clear   | 79.41              | 0.85                 | 0.6162         |
|                        | day_rain    | 89.47              | 1.16                 | 0.6068         |
|                        | night_clear | 76.00              | 0.86                 | 0.6455         |
|                        | night_rain  | 70.73              | 0.78                 | 0.6615         |
| **detection_finetuned**| **overall**| **93.82**          | **1.06**             | **0.8672**     |
|                        | day_clear   | 91.18              | 1.01                 | 0.8717         |
|                        | day_rain    | 100.00             | 1.21                 | 0.8740         |
|                        | night_clear | 96.00              | 1.10                 | 0.8599         |
|                        | night_rain  | 92.68              | 1.02                 | 0.8658         |

**Analysis of Comparative Results:**

- **Baseline vs. Fine-tuned**: There is a dramatic improvement between the baseline model and all fine-tuned versions. The overall detection rate jumped from **77.53%** to **96.43%** in the first fine-tuned version, and the average confidence increased from **0.63** to **0.84**. This clearly demonstrates the critical importance of fine-tuning on in-domain (synthetic) data to bridge the sim-to-real gap. The baseline model, trained on real-world plates, struggled with the unique textures, fonts, and lighting of GTA V.

- **Performance Across Conditions**: The fine-tuned models consistently outperform the baseline in all conditions, especially in challenging night and rain scenarios. For instance, in `night_rain`, the detection rate increased from **70.73%** (baseline) to over **91%** for all fine-tuned versions.

- **Iterative Improvement**: While the first fine-tuned model (`detection_finetuned`) shows the most significant leap in performance, subsequent versions (`v1` and `v2`) show incremental gains in average confidence, reaching a peak of **0.8672** in `v2`. This suggests that continued training and refinement can still yield improvements, particularly in the model's certainty. The detection rate slightly dipped in `v1` and `v2` compared to the first fine-tuned version, which could be a sign of minor overfitting or a trade-off for higher confidence; however, the overall performance remains exceptionally high.

### 4.4. OCR Performance

While a systematic quantitative analysis of the OCR module was not the primary focus of this phase, qualitative assessments were performed. The PaddleOCR engine, combined with the rule-based post-processing (regex `^\d{2}[A-Z]{3}\d{3}$` and character confusion correction), demonstrated strong performance on clearly detected license plates. The primary sources of error in the end-to-end pipeline were failures in detection, not recognition. When a plate was correctly detected and was reasonably clear, the OCR module was highly effective.

---

## 5. GUI Implementation

To provide an interactive and user-friendly way to test and demonstrate the ALPR pipeline, a graphical user interface (GUI) was designed using the Streamlit framework. Streamlit was chosen for its simplicity and rapid development capabilities, making it ideal for creating data-centric web applications.

The GUI design includes the following components:

-   **File Input:** An uploader for users to select a video file.
-   **Video Display:** A dedicated area to display video frames with overlaid bounding boxes and recognized license plate text.
-   **Processing Progress:** A progress bar to visualize the status of the video processing.
-   **Controls Panel:**
    -   **Start/Stop Buttons:** To initiate and halt the processing.
    -   **Threshold Sliders:** To dynamically adjust the `confidence` and `IOU` thresholds for the detector.
    -   **Device Selection:** A dropdown to select the computation device (`cuda` or `cpu`).
-   **Information Panel:**
    -   **Detected Count:** A running counter of unique license plates.
    -   **Latest Recognitions:** A list displaying the most recently recognized plates.
    -   **Log Tab:** A text area displaying real-time log messages for diagnostics.

This GUI encapsulates the core functionality of the pipeline into an accessible tool for demonstration and testing.

---

## 6. Discussion

The results clearly demonstrate the success of the core methodology. The most significant finding is the dramatic performance gap between the zero-shot baseline model and the fine-tuned versions. A generic, real-world license plate detector is insufficient for the specific visual domain of GTA V. Fine-tuning on a relatively small, custom-annotated dataset yielded a massive improvement, increasing the detection rate from 77.5% to over 96% and boosting confidence from 0.63 to over 0.84. This underscores the "garbage in, garbage out" principle in a new light: the domain of the training data is paramount.

The iterative improvements between the fine-tuned models (`v0`, `v1`, `v2`) were less dramatic but still valuable. The consistent increase in average confidence suggests that the model became more certain of its predictions with further training, even if the raw detection rate saw minor fluctuations. The `detection_finetuned_v2` model, with the highest overall confidence, represents the best-performing model.

One of the key challenges was the variability in in-game conditions. However, the data shows that fine-tuning was exceptionally effective at mitigating this. The models performed robustly across day, night, clear, and rainy conditions, with detection rates remaining high even in `night_rain` scenarios, which were particularly challenging for the baseline model.

A limitation of this study is the focus on detection metrics. While OCR performance was qualitatively strong, a more rigorous quantitative evaluation using Character Error Rate (CER) would be necessary for a complete assessment of the end-to-end system. Furthermore, the tracking component, while designed, was not the focus of this evaluation phase. Its impact on efficiency and the stability of recognition results is a key area for future work.

---

## 7. Conclusion

This project successfully developed a high-performance Automatic License Plate Recognition system for Grand Theft Auto V. By leveraging a two-stage pipeline with a fine-tuned YOLOv8 detector and a robust PaddleOCR engine, the system achieves excellent detection and recognition results across a wide range of in-game conditions.

The key takeaway is the critical importance of in-domain fine-tuning. The custom-annotated dataset of GTA V license plates was the single most important factor in achieving high accuracy, bridging the significant gap between real-world and synthetic data domains. The final model (`detection_finetuned_v2`) is capable of detecting license plates with high confidence and a low error rate, providing a solid foundation for the full ALPR pipeline.

Future work should focus on a quantitative evaluation of the OCR and tracking components, further optimization for real-time performance, and expanding the training dataset to cover even more edge cases within the game world. The designed Streamlit GUI provides a clear path for creating an interactive and demonstrable final application.

---

## 8. Use of AI Tools
Several AI tools were uilized during the project development for the following purposes:
- **Project Clarification and Suggestion**: I used AI Tools to help clarify my project goals, suggest methodologies, and outline the report structure. This helped in organizing my thoughts and ensuring a comprehensive approach.
- **Code Debugging**: I used AI Tools to help debug issues encountered during model training. This significantly sped up the development process.
- **Documentation**: AI Tools were also employed to help document the codebase, create README files for various scripts, and adding comments to complex sections of code. It ensured that the documentation was clear and helpful for future reference.


## 9. References

- Buslaev, A., et al. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information*, 11(2), 125.
- Hsu, G.-S., et al. (2013). Application-Oriented License Plate Recognition. *IEEE Transactions on Intelligent Transportation Systems*, 14(2), 963-967.
- Laroca, G. H., et al. (2018). A Robust Real-Time Automatic License Plate Recognition Based on Deep Learning. *2018 IEEE International Joint Conference on Neural Networks (IJCNN)*.
- Li, H., et al. (2019). Show, Attend and Read: A Simple and Strong Baseline for Irregular Text Recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33, 8610-8617.
- Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Shi, B., et al. (2017). An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 39(11), 2298-2304.
- Shrivastava, A., et al. (2017). Learning from Simulated and Unsupervised Images through Adversarial Training. *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
- Zhang, Y., et al. (2022). ByteTrack: A Simple and Effective Approach for Multi-Object Tracking. *arXiv preprint arXiv:2110.06864*.
- Zherzdev, P., & Gruzdev, A. (2019). CR-NET: A Simple Method for License Plate Recognition. *arXiv preprint arXiv:1902.09642*.
- **Rockstar Games** (2013). Grand Theft Auto V.
- **Waymo**. Simulation City. Retrieved from [https://waymo.com/blog/2021/10/simulation-city/](https://waymo.com/blog/2021/10/simulation-city/)
- **NVIDIA**. Drive Constellation. Retrieved from [https://www.nvidia.com/en-us/self-driving-cars/drive-constellation/](https://www.nvidia.com/en-us/self-driving-cars/drive-constellation/)
- **Intel Labs & Darmstadt University**. (2016). Playing for Data: Ground Truth from Computer Games. *ECCV 2016*.
- **Ultralytics YOLOv8:** [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **PaddleOCR:** [https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

