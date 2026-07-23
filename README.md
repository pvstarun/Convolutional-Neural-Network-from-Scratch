# Convolutional Neural Networks from Scratch

A modular implementation of Convolutional Neural Networks (CNNs) built entirely from scratch using **NumPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch.

This project extends the concepts of fully connected neural networks by implementing convolutional architectures, pooling operations, backpropagation, optimization algorithms, and modern weight initialization techniques. The implementation follows an object-oriented design where each layer is implemented independently, making the framework easy to understand and extend.

---

## Project Overview

The objective of this project is to understand how Convolutional Neural Networks work internally by implementing every major component manually.

Instead of using high-level libraries, every mathematical operation, including forward propagation, gradient computation, parameter updates, and convolution operations, is implemented using NumPy.

This project was developed as part of a Deep Learning course.

---

## Features

### Layers

- Convolution Layer (forward & backward propagation)
- Max Pooling Layer
- Flatten Layer
- Fully Connected Layer

### Activation Functions

- ReLU
- SoftMax

### Loss Function

- Cross Entropy Loss

### Optimization Algorithms

- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- Adam Optimizer

### Weight Initialization

- Constant Initialization
- Uniform Random Initialization
- Xavier (Glorot) Initialization
- He Initialization

### Framework Features

- Modular layer-based architecture
- Automatic forward propagation
- Automatic backward propagation
- Gradient-based parameter updates
- Training and inference pipelines
- Batch processing

---

## Project Structure

```
Convolutional-Neural-Network-from-Scratch
│
├── Layers/
│   ├── Conv.py
│   ├── Pooling.py
│   ├── Flatten.py
│   ├── FullyConnected.py
│   ├── ReLU.py
│   ├── SoftMax.py
│   ├── Initializers.py
│   └── ...
│
├── Optimization/
│   ├── Optimizers.py
│   └── Loss.py
│
├── NeuralNetwork.py
├── NeuralNetworkTests.py
├── README.md
└── requirements.txt
```

---

## Network Architecture

A typical CNN implemented using this framework follows the structure:

```
Input Image
      │
      ▼
Convolution
      │
      ▼
ReLU
      │
      ▼
Max Pooling
      │
      ▼
Flatten
      │
      ▼
Fully Connected
      │
      ▼
SoftMax
      │
      ▼
Prediction
```

---

## Mathematical Concepts Implemented

The framework includes manual implementations of:

- Cross-correlation / convolution
- Zero padding
- Strided convolution
- Max pooling
- Gradient computation using backpropagation
- Chain rule for CNNs
- Weight initialization strategies
- Gradient descent optimization
- Momentum optimization
- Adam optimization
- Cross-entropy loss

---

## Technologies Used

- Python 3
- NumPy
- Matplotlib
- SciPy
- scikit-learn (only for loading benchmark datasets)

---

## Results

The framework was evaluated on standard benchmark datasets.

### Iris Dataset

Accuracy obtained during different test runs:

- 100%
- 98%
- 96%
- 94%
- 92%

### UCI Handwritten Digits Dataset

Typical performance:

- 96.16%
- 94.82%
- 94.66%

The variation in accuracy is expected because different network configurations and initialization methods were evaluated during testing.

---

## Learning Outcomes

Through this project, I gained practical experience in:

- Building convolutional neural networks without deep learning libraries
- Understanding convolution and pooling operations mathematically
- Implementing forward and backward propagation manually
- Computing gradients using the chain rule
- Designing reusable neural network layers
- Implementing optimization algorithms from scratch
- Applying modern weight initialization techniques
- Training CNNs using mini-batch gradient descent

---

## Related Project

This project builds upon my previous repository:

**Neural Networks from Scratch**

which implements:

- Fully Connected Neural Networks
- Activation Functions
- Loss Functions
- Optimizers
- Backpropagation

Together, the two repositories demonstrate the implementation of both traditional feedforward neural networks and convolutional neural networks entirely from scratch using NumPy.

---

## License

This project was developed for educational purposes as part of a Deep Learning course.
