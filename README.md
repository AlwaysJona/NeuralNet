# NeuralNet

A lightweight neural network framework implemented from scratch in modern C++.

NeuralNet is a deep learning library designed to explore the fundamental principles behind neural networks, automatic differentiation, optimization, and numerical computing. Instead of relying on existing machine learning frameworks, this project implements the core components required to train neural networks from the ground up, including tensors, computational graphs, gradient propagation, layers, activation functions, loss functions, and optimization algorithms.

The main goal of this project is not to compete with large-scale frameworks such as PyTorch or TensorFlow, but to better understand how modern deep learning frameworks work, by implementing the main building blocks manually. 

The project is written in modern C++. It makes use of object-oriented programming, templates, memory management techniques, and numerical operations to create a flexible foundation for experimenting with machine learning algorithms.

# Features

## Core Tensor Engine

The foundation of NeuralNet is a custom Tensor implementation capable of storing and manipulating numerical data.

Current tensor capabilities include:

- Multi-dimensional array storage.
- Shape tracking.
- Element-wise operations.
- Matrix multiplication.
- Reduction operations.
- Gradient storage.
- Operator overloading for intuitive mathematical expressions.

Example:

```cpp
Tensor a({1, 2, 3});
Tensor b({4, 5, 6});

Tensor c = a + b;
```
