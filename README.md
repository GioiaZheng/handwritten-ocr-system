# Handwritten Text Recognizer (Course Project)

This repository contains a **handwritten text recognition system** developed as part of the **Artificial Intelligence Laboratory** course at *Sapienza University of Rome*.

It is based on the original group project implemented in **Java**, with additional testing and evaluation scripts contributed in **Python**.  
The project explores techniques in **computer vision**, **deep learning**, and **optical character recognition (OCR)**.

---

## 1. Overview

The goal of this project was to design and implement a system capable of recognizing handwritten text from scanned images using deep learning models.  
The system combines convolutional and recurrent layers with the **Connectionist Temporal Classification (CTC)** loss to align text sequences with image data.

The testing and evaluation modules included in this version were developed by **Gioia Zheng** to measure model performance and visualize results.

---

## 2. Project Objectives

- Implement a **neural network-based OCR pipeline**
- Explore **feature extraction**, **image preprocessing**, and **text decoding**
- Evaluate model accuracy using **Character Error Rate (CER)** and **Word Error Rate (WER)**
- Conduct experiments on real-world datasets (IAM Handwriting Database)
- Understand trade-offs between model complexity and recognition accuracy

---

## 3. Technologies and Tools

| Category | Technologies |
|-----------|--------------|
| **Languages** | Java, Python |
| **Libraries** | PyTorch, NumPy, OpenCV, Matplotlib |
| **Metrics** | CER (Character Error Rate), WER (Word Error Rate) |
| **Frameworks** | TensorBoard for visualization |
| **Dataset** | IAM Handwriting Database |

---

## 4. Project Structure

```

Handwritten-Text-Recognizer/
├── Model/                     # Trained model and checkpoints
├── OCR_WebApp/                # GUI components
├── data_loader.py             # Dataset handling
├── preprocessor.py            # Image preprocessing and augmentation
├── model1.py                  # Model definition (CNN + RNN + CTC)
├── testing.py                 # Evaluation script (Python, by Gioia Zheng)
├── test_sing.py               # Single-image testing example
├── utils.py                   # Utility functions for data and metrics
└── project_planning.ipynb     # Initial design and planning notes

````

---

## 5. How to Run (Python Evaluation)

### Step 1 — Install dependencies

```bash
pip install torch torchvision numpy matplotlib opencv-python torchmetrics
````

### Step 2 — Run the testing module

```bash
python testing.py
```

This will evaluate the model on the IAM dataset and print results including CER and WER metrics.

---

## 6. Sample Output

```bash
our prediction: the quick brown fox
our answer: the quick brown fox
our final character error rate: 0.042
our final word error rate: 0.083
```

---

## 7. Contribution

| Name            | Role                 | Contribution                                                                  |
| --------------- | -------------------- | ----------------------------------------------------------------------------- |
| **Gioia Zheng** | Testing & Evaluation | Implemented Python test module, performance metrics, and output visualization |

---

## 8. Course Information

**Course:** Programming 2 (Java)
**Degree Program:** Applied Computer Science and Artificial Intelligence
**University:** Sapienza University of Rome
**Academic Year:** 2022–2023

---

## 9. License

MIT License © 2025

---

## 10. Notes

> This repository is a **personal fork** of the original group project and includes additional testing and analysis modules for independent evaluation purposes.
> The base model and data pipeline were collaboratively developed as part of the course assignment.
