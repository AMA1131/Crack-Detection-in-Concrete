# Concrete Crack Classification using Classical Computer Vision and Deep Learning

## Project Overview

This project realises two different approaches for automatically classifying concrete surface images into two categories:

* Crack
* Non-crack

The objective is to compare a traditional Computer Vision pipeline based on handcrafted features with a Deep Learning solution based on Convolutional Neural Networks (CNNs).

---

## Dataset

The dataset used in this project is the **Concrete Crack Images for Classification** dataset from Kaggle.

* 40,000 RGB images
* 20,000 Crack images
* 20,000 Non-crack images
* Image resolution: 227 × 227 pixels

**Dataset download**

https://www.kaggle.com/datasets/arunrk7/surface-crack-detection?resource=download

After downloading the dataset, extract it into:

```text
data/
└── archive/
    ├── Positive/
    └── Negative/
```

---

## Project Structure

```text
Industrial Bottle Defect Detection/

│
├── data/
│
├── models/
│   ├── cnn_model.keras
│   └── svm_rbf_hog.pkl
│
├── notebooks/
│   └── data_exploration.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── feature_extraction.py
│   ├── classical_pipeline.py
│   ├── train_svm.ipynb
│   └── train_cnn.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/AMA1131/Crack-Detection-in-Concrete
```

Install the required Python packages

```bash
pip install -r requirements.txt
```

---

## Methodology

### Classical Computer Vision

* Image preprocessing
* Grayscale conversion
* HOG feature extraction
* Linear SVM
* RBF SVM

### Deep Learning

* Convolutional Neural Network
* Three convolutional blocks
* ReLU activation
* MaxPooling
* Flatten
* Fully Connected Layer
* Softmax classifier

---

## Experimental Results

| Model      | Accuracy   | False Negatives | False Positives |
| ---------- | ---------- | --------------- | --------------- |
| Linear SVM | 96.5%      | 20              | 7               |
| RBF SVM    | 98.62%     | 9               | 2               |
| CNN        | **99.42%** | **3**           | **4**           |

The CNN achieved the highest overall classification accuracy while reducing the number of missed cracks compared to the classical HOG + SVM pipeline.

---

## Technologies

* Python
* OpenCV
* NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Jupyter Notebook

---

