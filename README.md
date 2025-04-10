#Fashion MNIST Classification with ANN and Hyperparameter Tuning

This project implements an Artificial Neural Network (ANN) to classify images from the Fashion MNIST dataset using PyTorch. It leverages Optuna for hyperparameter optimization to maximize classification accuracy. The code includes data preprocessing, model training, evaluation, and visualization of the optimization process.

#Overview

The script performs the following tasks:

Loads and preprocesses the Fashion MNIST dataset.
Visualizes the first 16 images in a 4x4 grid.
Splits the data into training and test sets.
Defines a custom PyTorch Dataset and a configurable ANN model.
Uses Optuna to tune hyperparameters such as the number of hidden layers, neurons per layer, learning rate, and more.
Trains the model and evaluates its accuracy on the test set.
Generates visualizations to analyze the hyperparameter optimization process.

#Requirements
Python 3.8+

Libraries:
pandas
scikit-learn
torch
matplotlib
optuna

Project Structure
script.py: The main script contains the ANN model, data processing, and Optuna optimization logic.
fashion-mnist_train.csv: Input dataset (not included; must be downloaded).
requirements.txt: List of required Python packages (optional; create manually if needed).
Hyperparameter Tuning
The script uses Optuna to optimize the following hyperparameters:

num_hidden_layers: Number of hidden layers (1 to 5).
neurons_per_layer: Neurons per hidden layer (8 to 128, step=8).
epochs: Training epochs (10 to 50, step=10).
learning_rate: Learning rate (1e-5 to 1e-1, logarithmic scale).
dropout_rate: Dropout rate (0.1 to 0.5, step=0.1).
batch_size: Batch size (16, 32, 64, or 128).
optimizer: Optimizer type (Adam, SGD, or RMSprop).
weight_decay: Weight decay for regularization (1e-5 to 1e-3, logarithmic scale).
The objective is to maximize classification accuracy on the test set.

Visualizations:

After optimization, the script generates the following plots using Optuna's visualization tools:

Optimization History: Shows accuracy over trials.
Parallel Coordinates Plot: Visualizes relationships between hyperparameters and accuracy.
Slice Plot: Displays the effect of individual hyperparameters on accuracy.
Contour Plot: Shows interactions between pairs of hyperparameters.
Hyperparameter Importance: Highlights which hyperparameters most influence performance.
These plots are displayed interactively using Matplotlib.
