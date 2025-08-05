

```
# This is formatted as code
```

# Fraud Detection Model

This repository contains a PyTorch implementation of a neural network for credit card fraud detection.
**Accuracy: 0.9994**
**Precision: 0.8144**
**Recall: 0.8061**
**F1-score: 0.8103**

## Overview

The notebook walks through the process of building and training a binary classification model using a dataset containing anonymized credit card transactions. The goal is to identify fraudulent transactions.

## Dataset

The model is trained on a dataset that includes transaction details such as time, amount, and 28 anonymized features (V1-V28), along with a 'Class' label indicating whether the transaction is fraudulent (1) or not (0).

## Model Architecture

The model is a simple feedforward neural network with two hidden layers and dropout for regularization.

- Input Layer: Matches the number of features in the dataset.
- Hidden Layer 1: 64 neurons, followed by ReLU activation and dropout.
- Hidden Layer 2: 32 neurons, followed by ReLU activation and dropout.
- Output Layer: 1 neuron with a linear activation (sigmoid is applied during evaluation for probability).

## Dependencies

The following libraries are required to run the notebook:

- `torch`
- `torch.nn`
- `torch.nn.functional`
- `sklearn.model_selection`
- `sklearn.preprocessing`
- `torch.optim`
- `numpy`
- `pandas`
- `sklearn.metrics`

## Usage

1.  **Load the data**: Load the `creditcard.csv` file into a pandas DataFrame.
2.  **Preprocessing**:
    - Separate features (X) and labels (y).
    - Convert data to PyTorch tensors.
    - Handle any potential NaN values.
    - Split the data into training and testing sets.
    - Scale the features using `StandardScaler`.
3.  **Model Definition**: The `FraudDetectionModel` class defines the neural network architecture.
4.  **Training**:
    - Instantiate the model, loss function (`BCEWithLogitsLoss` with `pos_weight` to handle class imbalance), and optimizer (`Adam`).
    - Create `DataLoader` instances for the training and testing datasets.
    - Train the model using the `train_model` method.
5.  **Evaluation**: Evaluate the trained model on the test set using the `evaluate_model` method, which calculates accuracy, precision, recall, and F1-score. A threshold is applied to the model's output to classify predictions.

## Files

- `creditcard.csv`: The dataset containing the transaction data.
- This notebook: Contains the Python code for the model implementation, training, and evaluation.

## Results

The evaluation metrics (Accuracy, Precision, Recall, F1-score) on the test set are printed after the model is evaluated.
