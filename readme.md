# Face Mask Detection System

## Project Overview

This project is a real-time computer vision system that detects human faces and classifies whether each detected person is wearing a face mask correctly, incorrectly, or not at all. It combines **YOLO** for fast, accurate face detection with a **MobileNetV2**-based deep learning classifier for mask status prediction, and uses **OpenCV** to process live webcam video for real-time inference.

The goal of this project is to demonstrate an end-to-end applied deep learning pipeline — from dataset preparation and model training to real-time deployment — that could be used in public health, workplace safety, or access control scenarios.

---

## Features

- Real-time face detection using YOLO
- Face mask classification into three categories: **With Mask**, **Without Mask**, and **Mask Worn Incorrectly**
- Transfer learning with MobileNetV2 for lightweight, fast inference
- Live webcam-based detection pipeline using OpenCV
- Confidence score displayed alongside each prediction
- Modular codebase separating detection, classification, and camera handling
- Jupyter notebooks documenting the full training and evaluation workflow

---

## System Architecture

The system is composed of two deep learning components working together in sequence:

1. **Face Detector (YOLO)** — Locates faces within each video frame and returns bounding box coordinates.
2. **Mask Classifier (MobileNetV2)** — Takes each cropped face region and predicts the mask status.

This two-stage design separates the detection task (finding *where* faces are) from the classification task (determining *what* mask status each face has), allowing each model to specialize and perform efficiently.

---

## Working Pipeline

```
Webcam / Image Input
        |
        ↓
YOLO Face Detection
        |
        ↓
Face Region Extraction
        |
        ↓
MobileNetV2 Classification
        |
        ↓
Mask Status + Confidence Score
```

**Step-by-step explanation:**

1. A frame is captured from the webcam (or loaded from an image file).
2. YOLO processes the frame and outputs bounding boxes for all detected faces.
3. Each bounding box region is cropped from the original frame.
4. The cropped face is resized and preprocessed to match MobileNetV2's input format (224x224x3).
5. MobileNetV2 classifies the face into one of three categories and returns a confidence score.
6. The result (label + confidence) is overlaid on the video frame in real time.

---

## Technologies Used

| Category | Tools / Libraries |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| Classification Model | MobileNetV2 |
| Object Detection | YOLO (Ultralytics) |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Model Evaluation | Scikit-learn |

---

## Dataset Description

> *Placeholder — update with your actual dataset details.*

- **Dataset source:** `[Insert dataset name/source, e.g., Kaggle Face Mask Detection Dataset]`
- **Total images:** `[Insert number]`
- **Classes:**
  - With Mask: `[Insert count]`
  - Without Mask: `[Insert count]`
  - Mask Worn Incorrectly: `[Insert count]`
- **Image format:** `[e.g., JPG/PNG]`
- **Data split:** Train / Validation / Test (see `processed_dataset/` folder)

Dataset exploration and class distribution analysis are documented in `01_dataset_analysis.ipynb`.

---

## Data Preprocessing

The following preprocessing steps are applied before training, as documented in `02_data_preprocessing.ipynb`:

- **Resizing:** All images resized to 224x224 pixels to match MobileNetV2's expected input shape.
- **Normalization:** Pixel values scaled using MobileNetV2's `preprocess_input` function.
- **Data Augmentation:** Applied to improve generalization and reduce overfitting, including:
  - Random rotation
  - Horizontal flipping
  - Zoom and shear transformations
  - Brightness adjustment
- **Train/Validation/Test Split:** Dataset organized into separate folders to prevent data leakage during evaluation.
- **Label Encoding:** Class labels mapped to numerical values for model training.

---

## Model Architecture

### MobileNetV2 (Mask Classifier)

- **Approach:** Transfer learning using ImageNet pretrained weights
- **Base model:** MobileNetV2 (frozen or fine-tuned base layers)
- **Input size:** 224 x 224 x 3
- **Custom classification head:**
  - Global Average Pooling layer
  - Dense layer with ReLU activation
  - Dropout layer for regularization
  - Output Dense layer with Softmax activation (3 classes)
- **Output classes:**
  - With Mask
  - Without Mask
  - Mask Worn Incorrectly

### YOLO (Face Detector)

- Used exclusively for real-time face detection, not classification
- Detects bounding boxes for all faces present in a frame
- Lightweight and optimized for real-time performance on video streams
- Detected face regions are cropped and passed to MobileNetV2 for classification

---

## Training Process

Documented in `03_train_mobilenetv2.ipynb`:

1. Load and preprocess the training and validation datasets.
2. Initialize MobileNetV2 with pretrained ImageNet weights, excluding the top classification layer.
3. Add a custom classification head suited to the 3-class mask detection problem.
4. Freeze base layers initially and train only the custom head.
5. (Optional) Fine-tune the top layers of the base model with a low learning rate for improved accuracy.
6. Use callbacks such as `EarlyStopping` and `ModelCheckpoint` to prevent overfitting and save the best-performing model.
7. Save the final trained model as `mask_classifier_mobilenetv2.h5`.

**Training configuration (placeholders):**

| Parameter | Value |
|---|---|
| Optimizer | `[e.g., Adam]` |
| Learning Rate | `[Insert value]` |
| Batch Size | `[Insert value]` |
| Epochs | `[Insert value]` |
| Loss Function | Categorical Crossentropy |

---

## Evaluation Metrics

Model performance is evaluated in `04_evaluate_mobilenetv2.ipynb` using the following metrics:

- **Accuracy** — Overall correctness of predictions
- **Precision** — Correctness of positive predictions per class
- **Recall** — Ability of the model to correctly identify each class
- **F1-Score** — Harmonic mean of precision and recall
- **Confusion Matrix** — Visual breakdown of correct vs. incorrect predictions per class

---

## Results

> *Placeholder — replace with actual evaluation results after training.*

| Metric | Score |
|---|---|
| Accuracy | `[Insert value]%` |
| Precision | `[Insert value]%` |
| Recall | `[Insert value]%` |
| F1-Score | `[Insert value]%` |

**Confusion Matrix:**

```
[Insert confusion matrix image or plot here]
```

**Training/Validation Curves:**

```
[Insert accuracy/loss curve screenshots here]
```

---

## Project Structure

```
Face-Mask-Detection-AI/
├── models/
│   ├── mask_classifier_mobilenetv2.h5     # Trained MobileNetV2 mask classifier
│   └── yolov8n-face.pt                    # YOLO face detection weights
│
├── src/
│   ├── camera.py                          # Webcam capture and video loop
│   ├── yolo_detector.py                   # YOLO face detection logic
│   └── mask_classifier.py                 # MobileNetV2 inference logic
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb          # Dataset exploration and class distribution
│   ├── 02_data_preprocessing.ipynb        # Preprocessing and augmentation
│   ├── 03_train_mobilenetv2.ipynb         # Model training
│   ├── 04_evaluate_mobilenetv2.ipynb      # Model evaluation and metrics
│   └── 05_yolo_mobilenet_integration.ipynb# YOLO + MobileNetV2 pipeline integration
│
├── processed_dataset/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── requirements.txt                       # Project dependencies
└── README.md                              # Project documentation
```

---

## Installation Steps

### 1. Clone the repository

```bash
git clone https://github.com/[your-username]/Face-Mask-Detection.git
cd Face-Mask-Detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` should include (adjust versions as needed):**

```
tensorflow
opencv-python
ultralytics
numpy
pandas
matplotlib
scikit-learn
```

---

## How to Run the Project

### Run real-time detection using webcam

```bash
python src/camera.py
```

This will:
1. Open your webcam feed
2. Detect faces in real time using YOLO
3. Classify each detected face using MobileNetV2
4. Display the mask status and confidence score on screen

Press `q` to exit the webcam window.

### Run detection on a single image (optional)

```bash
python src/camera.py --image path/to/image.jpg
```

> *Note: Update this command to match your actual CLI arguments if implemented differently in `camera.py`.*

---

## Future Improvements

- Extend detection to support multiple camera streams simultaneously
- Deploy the model as a web application using Flask/FastAPI and a browser-based interface
- Optimize models using TensorFlow Lite or ONNX for edge device deployment (e.g., Raspberry Pi)
- Improve dataset diversity to enhance performance across different lighting conditions, angles, and mask types
- Add a logging/alerting system for mask-compliance monitoring in real-world deployments
- Experiment with alternative lightweight backbones (e.g., EfficientNet-Lite) for further speed optimization

---

## Author

**[Prit Gajjar]**
AI/ML Engineer

- GitHub: [https://github.com/your-username](https://github.com/Prit0911)
- LinkedIn: [https://www.linkedin.com/in/your-linkedin-profile](https://www.linkedin.com/in/prit-gajjar-09d1104/)
- Email: `pritgajjar0911@gmail.com`

---

*If you found this project useful, consider giving it a star on GitHub.*
