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
├── crnn_ctc.py                # CNN + RNN + CTC model
├── evaluate.py                # Evaluation pipeline
├── infer_single.py            # Single image inference
├── utils.py                   # Utility functions (metrics, decoding)
└── project_planning.ipynb     # Initial experiments and notes

````

---

# Quickstart

```bash
git clone https://github.com/GioiaZheng/handwritten-ocr-system.git
cd handwritten-ocr-system

pip install -r requirements.txt

# Put IAM sentence images under data/iam_sentences/dataset
# and the metadata file at data/iam_sentences/metadata/sentences.txt.
python evaluate.py
````

Expected output:

```text
number of batches in the test loader: <count>
blank token: <id>
our prediction: <decoded text>
our answer: <ground truth text>
our final character error rate: <CER>
our final word error rate: <WER>
```

---

# Training and Evaluation Config

The reproducibility settings are recorded in [configs/training_config.yaml](configs/training_config.yaml):

- seed: `2024`
- split random state: `42`
- split: `80% train / 10% validation / 10% test`
- checkpoint expected by evaluation: `Model/model.pt`

The CER/WER results ledger is in [docs/evaluation.md](docs/evaluation.md). The repository does not currently include a reproducible checkpoint/log with verified CER and WER values, so the table marks the baseline result as pending instead of listing example numbers as measured results.

Detailed setup, data layout, field definitions, and smoke-test instructions are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Dataset-free smoke tests:

```bash
python -m unittest discover -s tests
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
