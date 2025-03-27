# tuberculosis
# Tuberculosis Detection Neural Network

## Overview
This project implements a deep neural network using TensorFlow's Functional API to predict tuberculosis cases based on various patient symptoms and characteristics. The model uses both categorical and numerical features to make predictions with high accuracy.

## Dataset
The model uses the `tuberculosis_xray_dataset.csv` which contains various patient features including:

Categorical Features:
- Gender
- Chest Pain
- Night Sweats
- Sputum Production
- Blood in Sputum
- Smoking History
- Previous TB History
- Fever

Numerical Features:
- Age
- Cough Severity
- Breathlessness
- Fatigue
- Weight Loss

## Model Architecture
The neural network implements a deep architecture with:
- Input layer matching feature dimensions
- 4 dense layers (512, 256, 128, 64 units)
- Batch Normalization after each dense layer
- Dropout layers for regularization
- L2 regularization on dense layers
- Binary classification output

## Features
- One-hot encoding for categorical variables
- StandardScaler for numerical features
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Visualization of training metrics

## Requirements
- Python 3.11+
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Usage
To run the model:
```python
python main.py
```

The script will:
1. Load and preprocess the data
2. Train the model
3. Display test accuracy
4. Show training history plots

## Performance Visualization
The code includes visualization of:
- Training vs Validation Accuracy
- Training vs Validation Loss

These plots help in monitoring the model's learning progress and identifying potential overfitting or underfitting issues.
