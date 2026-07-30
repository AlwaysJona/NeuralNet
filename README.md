# NeuralNet

A lightweight neural network framework implemented from scratch in modern C++.

NeuralNet is a deep learning library designed to explore the fundamental principles behind neural networks, automatic differentiation, optimization, and numerical computing. Instead of relying on existing machine learning frameworks, this project implements the core components required to train neural networks from the ground up, including tensors, computational graphs, gradient propagation, layers, activation functions, loss functions, and optimization algorithms.

The main goal of this project is not to compete with large-scale frameworks such as PyTorch or TensorFlow, but to better understand how modern deep learning frameworks work, by implementing the main building blocks manually. 

The project is written in modern C++. It makes use of object-oriented programming, templates, memory management techniques, and numerical operations to create a flexible foundation for experimenting with machine learning algorithms.

---

## Table of Contents

- [Build and Run](#build-and-run)

## Build and Run
### Requirements
- C++17 compatible compiler (GCC / Clang / MSVC)
- CMake &ge; 3.10
- Git (to download doctest) 

### Clone the repository
```bash
git clone https://github.com/AlwaysJona/NeuralNet.git
cd NeuralNet
```

### Build
Create a build directory and configure the project with CMake:

```bash 
mkdir build 
cd build 
cmake ..
```

Compile the project

```bash
cmake --build .
```

### Run 
The repository also includes a fully functional example that can be fiddled with, in ```bash src/main.cpp ```. After a successful build, run the executable:

```bash
./build/src/neural
```
### Run Tests
After a successful build, run the tests:

```bash
./build/tests/tests
```

---

## Features

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
