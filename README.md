# NeuralNet

A lightweight neural network framework implemented from scratch in modern C++.

NeuralNet is a deep learning library designed to explore the fundamental principles behind neural networks, automatic differentiation, optimization, and high-performance numerical computing. Instead of relying on existing machine learning frameworks, this project implements the core components required to train neural networks from the ground up, including tensors, computational graphs, gradient propagation, layers, activation functions, loss functions, and optimization algorithms.

The main goal of this project is not to compete with large-scale frameworks such as PyTorch or TensorFlow, but to provide a transparent and extensible implementation where every component of the learning process can be inspected and understood.

The project is written in modern C++ with performance and software design in mind. It makes use of object-oriented programming, templates, memory management techniques, and efficient numerical operations to create a flexible foundation for experimenting with machine learning algorithms.

---

# Motivation

Modern deep learning frameworks provide powerful abstractions that allow researchers and engineers to train complex models with only a few lines of code. However, these abstractions often hide the mathematical and computational processes that make neural networks work.

This project was created to better understand the internal mechanisms behind deep learning systems by implementing the main building blocks manually.

The main motivations are:

- Understanding how tensors store and manipulate multidimensional data.
- Learning how computational graphs represent mathematical operations.
- Implementing automatic differentiation and backpropagation from first principles.
- Exploring how gradient-based optimization updates model parameters.
- Designing a modular neural network framework in modern C++.
- Investigating performance considerations involved in numerical computing.

By building each component manually, NeuralNet provides a deeper understanding of what happens inside frameworks such as PyTorch when a model performs forward propagation, computes a loss, calculates gradients, and updates its parameters.

The project also serves as an opportunity to apply software engineering principles to scientific computing, including:

- Clean architecture design.
- Unit testing.
- Memory management.
- Efficient data structures.
- Separation between mathematical operations and high-level model definitions.

---

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
