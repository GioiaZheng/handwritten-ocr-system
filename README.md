# Handwritten OCR System (Deep Learning)

![Python](https://img.shields.io/badge/language-python-blue)
![PyTorch](https://img.shields.io/badge/framework-pytorch-red)
![Task](https://img.shields.io/badge/task-OCR-green)
![Status](https://img.shields.io/badge/status-experimental-yellow)

A deep learning-based **handwritten text recognition system** built using CNN-RNN architectures with **CTC loss**, designed for sequence-to-text transcription from scanned images.

This project explores **end-to-end optical character recognition (OCR)** pipelines, combining computer vision, sequence modeling, and evaluation metrics.

---

# What This Project Demonstrates

- End-to-end **OCR pipeline design** (preprocessing → modeling → decoding)
- Sequence modeling using **CNN + RNN + CTC**
- Evaluation using **CER / WER metrics**
- Practical experimentation on real-world dataset (IAM)
- Integration of model inference and visualization tools

---

# System Overview

The OCR pipeline follows a standard sequence-to-text architecture:

```

Input Image
↓
Preprocessing (grayscale, normalization, augmentation)
↓
CNN (feature extraction)
↓
RNN (sequence modeling)
↓
CTC Loss (alignment)
↓
Decoder (text prediction)
↓
Final Output

```

---

# Key Features

- CNN + RNN architecture for handwritten text recognition  
- CTC-based alignment for variable-length sequences  
- Image preprocessing and augmentation pipeline  
- Evaluation metrics: **Character Error Rate (CER)** and **Word Error Rate (WER)**  
- Visualization of predictions and model outputs  

---

# Tech Stack

- **Python**
- **PyTorch**
- OpenCV
- NumPy / Matplotlib
- IAM Handwriting Dataset

---

# Project Structure

```

handwritten-ocr-system/
├── Model/                     # Trained models and checkpoints
├── OCR_WebApp/                # GUI components
├── data_loader.py             # Dataset loading
├── preprocessor.py            # Image preprocessing
├── model1.py                  # CNN + RNN + CTC model
├── testing.py                 # Evaluation pipeline
├── test_sing.py               # Single image inference
├── utils.py                   # Utility functions (metrics, decoding)
└── project_planning.ipynb     # Initial experiments and notes

````

---

# Running the Project

## Install dependencies

```bash
pip install torch torchvision numpy matplotlib opencv-python torchmetrics
````

## Run evaluation

```bash
python testing.py
```

---

# Example Output

```
prediction: the quick brown fox
ground truth: the quick brown fox
CER: 0.042
WER: 0.083
```

---

# Limitations

* Model architecture is relatively simple (baseline-level)
* No large-scale hyperparameter tuning
* Limited dataset generalization

---

# Future Work

* Transformer-based OCR models
* Attention-based decoding
* Data augmentation strategies
* End-to-end training pipeline optimization

---

# License

MIT License
