# 🐾 Oxford-IIIT Pets + Caltech 256: Multi-Task & Triplet Learning

## 📄 Project Overview

This repository presents an extensive study of **multi-task learning** (image classification + semantic segmentation) and **metric learning using triplet loss** applied on two renowned datasets:

- **Oxford-IIIT Pet Dataset**: used for simultaneous classification and segmentation
- **Caltech-256 Object Categories**: used for image retrieval and identity verification with triplet networks

Deep learning techniques (CNNs and transfer learning with MobileNet and InceptionV3) and classic machine learning (PCA, LDA, logistic regression) are combined to evaluate and benchmark multiple models.

---

## 📂 Datasets Used

### 1. Oxford-IIIT Pet Dataset
- **Source**: Built-in via `tensorflow_datasets`
- **Tasks**:
  - Multi-class classification (37 classes)
  - Semantic segmentation

### 2. Caltech-256 Object Categories
- **Source**: [https://data.caltech.edu/records/nyy15-4j048](https://data.caltech.edu/records/nyy15-4j048)
- **Tasks**:
  - Triplet learning and feature embedding
  - Retrieval and classification via PCA, LDA, and Logistic Regression

---

## 🚀 Key Features

### Oxford-IIIT Pet Pipeline:
- Data loader supporting joint or individual task output
- Preprocessing with image resizing, normalization, and augmentation
- Custom-built DCNN and fine-tuned MobileNetV3Small
- Evaluation with:
  - Confusion matrix, accuracy, F1 score
  - Side-by-side visualization of predicted and ground truth segmentation masks
  - Training history plots

### Caltech 256 Pipeline:
- Image loading with label parsing from folder names
- Data split: train/validation/test with stratification
- Feature extraction using:
  - Pre-trained InceptionV3 backbone
  - Custom ResNet trained with triplet loss
- Feature dimensionality reduction with Truncated SVD + PCA + LDA
- Logistic regression classifier on reduced features
- Extensive evaluation with:
  - Classification accuracy and F1 score
  - Confusion matrices
  - t-SNE visualization
  - ROC & Precision-Recall curves
  - Recall@K plots

---

## 📊 Performance Summary

| Model | Dataset | Task | Accuracy | F1 Score | Notes |
|-------|---------|------|----------|----------|-------|
| DCNN from scratch | Oxford-IIIT | Classification | ~85% | 0.84 | Joint training with segmentation |
| MobileNetV3Small | Oxford-IIIT | Classification | ~90% | 0.88 | Transfer learning, fine-tuned |
| Logistic Regression | Caltech-256 | Classification (LDA) | ~83% | 0.82 | PCA + LDA features |
| Triplet Network | Caltech-256 | Retrieval | - | Recall@1: ~0.82 | InceptionV3 + ResNet backbone |

---

## 📅 Usage

### Environment Setup
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python train_dcnn.py  # For Oxford-IIIT pets DCNN
python train_mobilenet.py  # For MobileNetV3Small
python train_triplet.py  # For Caltech-256 triplet model
```

### Evaluation
```bash
python evaluate_oxford.py
python evaluate_caltech.py
```

---

## 📆 Project Structure
```
.
├── oxford_pets/
│   ├── data_loader.py
│   ├── model_dcnn.py
│   ├── model_mobilenet.py
│   ├── train.py
│   └── evaluate.py
├── caltech256/
│   ├── triplet_loader.py
│   ├── triplet_model.py
│   ├── feature_extract.py
│   ├── dimensionality.py
│   ├── classification.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

---

## ⚖️ Licensing & Ethics
- Datasets are open-access for academic and research use
- No private, sensitive, or personally identifiable information is used
- This project is intended for educational and research purposes only

---

## 👨‍💼 Author
**Mohan Hao**  
Machine Learning & Full Stack Enthusiast  
imhaom@gmail.com

