# 🌟 Image Classification with Custom and Pretrained CNN Models 🌟

## 📋 Table of Contents
1. [Introduction](#-introduction)
2. [Dataset](#-dataset)
3. [Models Implemented](#-models-implemented)
4. [Results](#-results)
5. [References](#-references)

## 🔍 Introduction
This project focuses on building and comparing deep learning models for a multiclass image classification task. The notebook includes a custom Convolutional Neural Network (CNN) model and a popular pretrained architecture, **AlexNet**. The goal is to analyze their performance and identify the most effective model for this dataset.

## 📂 Dataset
The dataset contains images categorized into 5 distinct classes. Each image has been resized to 250x250 pixels with 3 color channels (RGB).

- **Number of Classes:** 5
- **Image Shape:** `(250, 250, 3)`

## 🧠 Models Implemented
This project evaluates multiple deep learning models:

### 1. Custom CNN Models
Three versions of a custom Convolutional Neural Network built from scratch:
- **Architecture:** 
  - Convolutional layers with ReLU activation
  - MaxPooling for dimensionality reduction
  - Fully connected dense layers with dropout for regularization
  - Softmax activation for multiclass classification

### 2. AlexNet
AlexNet is a pioneering deep learning model known for its performance in image recognition tasks:
- **Architecture:** 
  - Multiple convolutional layers with ReLU activations
  - MaxPooling layers to reduce spatial dimensions
  - Dense layers for high-dimensional feature extraction
  - Softmax output layer for classification

``

### Notebook Features
- **Data Preprocessing:** Rescaling, normalization, and splitting into training/validation sets.
- **Model Training:** Training the models using Keras with tracking of various performance metrics.
- **Evaluation:** Generates accuracy, precision, recall, and F1-score metrics.
- **Visualization:** Plots training history and confusion matrices for detailed performance analysis.

## 📊 Results
Here is a summary of the performance metrics for each model:

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Custom Model 1 | 98.69%   | 98.71%    | 98.70% | 98.70%   |
| Custom Model 2 | 98.60%   | 98.61%    | 98.60% | 98.60%   |
| Custom Model 3 | 99.44%   | 99.45%    | 99.45% | 99.45%   |
| AlexNet        | 98.47%   | 98.47%    | 98.48% | 98.47%   |

The **Custom Model 3** achieved the highest accuracy and overall performance among all the tested models.

## 📚 References
- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Keras Documentation](https://keras.io/)
- [Deep Learning Models for Image Classification](https://arxiv.org/abs/2104.00298)

<div align="center">
  <h3>⭐ Don't forget to star the repository if you found it useful! ⭐</h3>
</div>
