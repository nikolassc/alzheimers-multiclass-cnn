# Alzheimer's Disease Multi-Class Classification with CNNs and Classical Classifiers

This repository contains the implementation of the mandatory assignment for the course  
**"Neural Networks – Deep Learning"**. The goal of the project is to implement a feedforward neural network (in this case, a Convolutional Neural Network – CNN) trained with backpropagation to solve a **multi-class image classification** problem *other than MNIST*, as required by the assignment instructions. :contentReference[oaicite:1]{index=1}

The chosen problem is the multi-class classification of **Alzheimer's disease stages** from brain MRI images.

---

## 1. Problem Description

Alzheimer’s disease progresses through several clinical stages, from normal (non-demented) to moderate dementia. The aim of this project is to automatically classify brain MRI images into different stages of Alzheimer’s disease using supervised learning.

The project addresses a **multi-class classification** task, where each MRI image belongs to one of four classes (disease severity levels). A CNN is trained end-to-end on MRI images, and its performance is compared to simpler classical machine learning methods (k-Nearest Neighbors and Nearest Class Centroid), as requested in the assignment. :contentReference[oaicite:2]{index=2}

---

## 2. Dataset

### 2.1 Source

The dataset used in this project is the **Alzheimer's Disease Multiclass Images Dataset** from Kaggle:

> **Alzheimer's Multiclass Dataset – Equal and Augmented**  
> Author: Aryan Singhal  
> Kaggle URL: `https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented` :contentReference[oaicite:3]{index=3}

The dataset contains grayscale MRI images of the brain, categorized into four classes representing different stages of Alzheimer’s disease.

### 2.2 Classes

The images are divided into **four classes** corresponding to the severity of dementia:

- **NonDemented**
- **VeryMildDemented**
- **MildDemented**
- **ModerateDemented**

Each image is a brain MRI slice. The Kaggle version used here is **balanced and augmented**, meaning that augmentation techniques (e.g., rotations, flips) were applied to equalize the number of images in each class, resulting in a nearly balanced dataset suitable for supervised learning. :contentReference[oaicite:4]{index=4}

### 2.3 Preprocessing

In this project, the following preprocessing steps are applied:

- All images are loaded from the folder structure provided by Kaggle.
- Each image is converted to **grayscale** (if not already).
- Images are resized to a fixed resolution of **64 × 64** pixels for computational efficiency.
- Pixel values are normalized to the range `[0, 1]`.
- Two representations are prepared:
  - A **flattened vector** of size `64 * 64` for classical classifiers (kNN, NCC).
  - A **4D tensor** of shape `(H, W, 1)` suitable for CNN input.

The dataset is then split into:

- **Training set:** 60%  
- **Test set:** 40%  

The split is **stratified** to preserve class proportions across train and test sets, as required by the assignment when no predefined test split is provided. :contentReference[oaicite:5]{index=5}

---

## 3. Methods

The project implements both **classical** and **neural network** classification methods.

### 3.1 Classical Baselines: k-NN and Nearest Class Centroid (NCC)

As required by the intermediate part of the assignment, the following classical classifiers are implemented and evaluated on the same dataset:

1. **k-Nearest Neighbors (k-NN)**
   - k = 1 (1-NN)
   - k = 3 (3-NN)
2. **Nearest Class Centroid (NCC)**

These classifiers operate on the **flattened image vectors** (`64 × 64` pixels → 4096-dimensional feature vectors). The code:

- Fits the classifiers on the training set.
- Evaluates them on the test set.
- Reports classification accuracy and inference time for each model.

This allows a direct comparison between classical methods and the neural network approach.

### 3.2 Convolutional Neural Network (CNN)

A **feedforward Convolutional Neural Network** (CNN) is implemented using TensorFlow/Keras to solve the multi-class classification problem with backpropagation, as requested by the assignment. :contentReference[oaicite:6]{index=6}

#### Architecture (Baseline CNN)

The baseline CNN architecture includes:

- **Input layer** for grayscale images of shape `(64, 64, 1)`.
- Several **convolutional layers** with increasing number of filters (starting from a base number of filters).
- **Activation functions** (e.g., ReLU or a configurable alternative).
- **MaxPooling** layers to downsample feature maps.
- One or more **fully connected (Dense) layers** for high-level feature integration.
- **Dropout** layer to reduce overfitting.
- **Output layer** with `softmax` activation and `num_classes` units (4 classes).

The model is trained with:

- **Loss:** categorical cross-entropy  
- **Optimizer:** Adam  
- **Metrics:** accuracy  

The labels are encoded using **one-hot encoding**.

#### Training / Validation Split

From the training set, a part is further reserved for validation:

- **Train subset** for weight updates.
- **Validation set** to monitor generalization during training.

Typical hyperparameters used for the baseline experiment:

- `dense_units`: 128  
- `base_filters`: 32  
- `dropout_rate`: 0.5  
- `learning_rate`: 1e-3  
- `batch_size`: 32  
- `epochs`: 15  

The training history (loss and accuracy for training and validation) is stored and can be plotted.

---

## 4. Feature Extraction and PCA on CNN Features

In addition to using the CNN directly for classification, the project also explores **feature extraction** from the last hidden dense layer of the CNN:

1. The last hidden Dense layer (before the output layer) is located.
2. A new model is built that outputs the activations of this layer.
3. For a subset of images, the activations are computed and treated as **high-level feature vectors**.
4. **Principal Component Analysis (PCA)** is applied to reduce these feature vectors to 2D.
5. A scatter plot is created where each point corresponds to an image, colored by its class label.

This visualization helps to assess how well the CNN has learned a feature space where different classes are separable.

---

## 5. Evaluation and Results

The project reports:

- **Accuracy** for:
  - k-NN with k = 1
  - k-NN with k = 3
  - Nearest Class Centroid (NCC)
  - Baseline CNN (on the test set)
- **Training and validation curves** (loss and accuracy) for the CNN.
- **Classification report** (precision, recall, F1-score) for the CNN, per class.
- **Confusion matrix** for the CNN on the test set.
- Example **correct** and **incorrect** predictions:
  - Selected test images are plotted alongside their true label and predicted label.

Additional experiments with different CNN hyperparameters (e.g., different numbers of filters, dense units, dropout rates, learning rates, numbers of epochs, etc.) can be logged and compared using CSV files or stored models.

---

## 6. Comparison Between Methods

Following the assignment instructions, the performance of the neural network is compared to:

- **1-NN**
- **3-NN**
- **Nearest Class Centroid**

The comparison is based on:

- Test set accuracy.
- Qualitative analysis of decision boundaries (via PCA visualization).
- Confusion matrices and class-wise metrics.

Typically, the CNN outperforms the classical baselines by learning more informative features directly from the images, especially after augmentation and careful regularization.

---

## 7. How to Run the Code

The project is designed to run easily in **Google Colab**.

### 7.1 Requirements

Main Python libraries used:

- `tensorflow` / `keras`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`
- `kagglehub` (for downloading the dataset directly from Kaggle)

### 7.2 Steps (in Colab)

1. Open the notebook (`.ipynb`) in Google Colab.
2. Make sure you have access to Kaggle through `kagglehub` or manually download and extract the dataset from Kaggle to your Colab or local environment.
3. Run all cells in order:
   - Dataset download and extraction.
   - Image loading and preprocessing.
   - Train/test split.
   - k-NN and NCC training and evaluation.
   - CNN model building, training, and evaluation.
   - PCA feature visualization and additional experiments.
