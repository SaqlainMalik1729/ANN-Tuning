# ANN-Tuning
This project implements a neural network designed to classify fashion images, leveraging deep learning techniques for accurate recognition. The model has been optimized through hyperparameter tuning using Optuna, a powerful framework for automated hyperparameter optimization. The goal is to achieve high accuracy in identifying clothing items from datasets like Fashion MNIST or similar.

The best parameters achieved in this case are as follows:

{'num_hidden_layers': 2, 'neurons_per_layer': 72, 'epochs': 50, 'learning_rate': 4.68265198147004e-05, 'dropout_rate': 0.5, 'batch_size': 32, 'optimizer': 'RMSprop', 'weight_decay': 2.0873900476163065e-05}
