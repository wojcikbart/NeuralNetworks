# Neural Networks Repository

This repository contains implementations of three major neural network projects:

### 1. **Multilayer Perceptron (MLP)**

This project focuses on implementing a feedforward MLP with various learning algorithms and architectures.

- **NN1**: Basic MLP implementation with custom architecture and activation functions.
- **NN2**: Implementing backpropagation with weight updates (batch and online).
- **NN3**: Adding momentum and RMSProp gradient normalization for faster convergence.
- **NN4**: Implementing the softmax function for classification tasks.
- **NN5**: Testing different activation functions (sigmoid, ReLU, tanh).
- **NN6**: Implementing regularization to combat overfitting with weight decay and early stopping.

Each part includes experiments to optimize hyperparameters and measure performance on various datasets.

---

### 2. **Kohonen Self-Organizing Map (SOM)**

This project implements Kohonen's SOM algorithm to cluster data into meaningful groups without supervision.

- **KOH1**: Basic SOM with a rectangular grid and Gaussian neighborhood function, tested on simple 2D and 3D datasets.
- **KOH2**: Enhanced SOM with a hexagonal grid, applied to MNIST and other real-world datasets (without labels).

The goal is to analyze how well the network clusters data and maps input vectors to neurons.

---

### 3. **Genetic Algorithms**

This project implements optimization tasks using genetic algorithms.

- **AE1**: Basic genetic algorithm with Gaussian mutation and one-point crossover.
- **AE2**: Solving the cutting stock problem using genetic algorithms to pack rectangles into a circle.
- **AE3**: Optimizing MLP weights using genetic algorithms for tasks like classification and regression.

---
