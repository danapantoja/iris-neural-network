# iris-neural-network
This is a feedforward neural network implemented from scratch in Python using only NumPy. The network is trained on the classic Iris dataset (two classes: Setosa and Versicolor) to perform binary classification.

## Features

- Multiple layers with customizable sizes
- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent backpropagation training
- Standardization of input features
- Train/test split evaluation
- Simple accuracy metric for performance
- Training loss tracking and visualization using Matplotlib

## Results

After training for 10,000 epochs, the neural network achieves approximately **97% accuracy** on the test set (binary classification of Setosa vs. Versicolor).


Future work
Improve multiclass classification accuracy

Add confusion matrix and other evaluation metrics

Explore different architectures and activation functions

## Future Work
- Add support for other activation functions (ReLU, tanh, etc.) to improve learning performance
- Implement additional evaluation metrics such as accuracy, precision, recall, and confusion matrix visualization
- Explore applying the network to other datasets and real-world problems
