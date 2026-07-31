# NeuralNet

A lightweight neural network framework implemented from scratch in modern C++.

NeuralNet is a deep learning library designed to explore the fundamental principles behind neural networks, automatic differentiation, optimization, and numerical computing. Instead of relying on existing machine learning frameworks, this project implements the core components required to train neural networks from the ground up, including tensors, computational graphs, gradient propagation, layers, activation functions, loss functions, and optimization algorithms.

The main goal of this project is not to compete with large-scale frameworks such as PyTorch or TensorFlow, but to better understand how modern deep learning frameworks work, by implementing the main building blocks manually. 

The project is written in modern C++. It makes use of object-oriented programming, templates, memory management techniques, and numerical operations to create a flexible foundation for experimenting with machine learning algorithms.

---

## Table of Contents

- [Build and Run](#build-and-run)
- [Features](#features)
  - [Core Tensor Engine](#core-tensor-engine)
  - [Gradient Backpropagation](#gradient-backpropagation)
  - [Modularity](#modularity)
- [Project Architecture](#project-architecture)
  - [Header Files](#header-files)
  - [Source Files](#source-files)
  - [Tests](#tests)
  - [Data](#data)
  - [Models](#models)
- [Execution Flow](#execution-flow)

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
The repository also includes a fully functional example that can be fiddled with, in ```src/main.cpp ```. After a successful build, run the executable:

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

### Core Tensor Engine

The foundation of NeuralNet is a custom Tensor implementation capable of storing and manipulating numerical data.

Current tensor capabilities include:

- Multi-dimensional array storage.
- Element-wise operations.
- Matrix multiplication.
- Gradient storage.

Example:

```cpp
Tensor a({1, 2, 3});
Tensor b({4, 5, 6});

Tensor c = a + b;
```
The Tensor class is however just an interface that stores a pointer to a 
```Node``` object, which is responsible to hold information about its data, 
its shape, its parent 
Nodes, its gradient and gradient function.

### Gradient Backpropagation

A core feature of any deep learning framework is the ability to compute 
the gradient of the loss function with respect to every trainable parameter. 
Since the loss function is typically a **composite function** formed by a 
sequence of tensor operations, layers, and activation functions, gradients 
are computed by repeatedly applying the **chain rule** during the 
backpropagation process.

To support automatic differentiation, each `Node` object stores two 
fundamental components:

- A gradient function, which defines the local derivative of the operation 
that produced the node.
- A `std::vector<std::shared_ptr<Node>>` containing pointers to its 
parent nodes in the computational graph.

Starting from the loss node, the framework traverses the computational 
graph in reverse order. At each node, the stored gradient function is 
evaluated and the resulting gradient is propagated to its parent nodes 
through the stored pointers. By recursively applying the chain rule, the 
framework accumulates the gradients required to update all trainable parameters.

### Modularity

The neural network architecture is constructed using **modules**, which
are objects belonging to classes derived from the base `Module` class. 
Each module represents a specific operation or transformation applied to 
the data as it flows through the network.

These modules can be divided into three main categories:

- **Layer Modules:** Transform the input received from the previous 
layer by applying learnable parameters, such as weights and biases, 
modifying the shape and values of the underlying `Node`.
- **Activation Function Modules:** Apply non-linear activation functions, 
such as ReLU or Softmax, introducing non-linearity into the network and 
allowing it to learn more complex relationships.
- **Loss Computing Modules:** Evaluate the difference between the network 
output and the target labels, providing the loss value used during the 
optimization process.

This modular design allows the network architecture to be easily extended and modified. Layers, activation functions, and loss functions can be added, removed, or rearranged without requiring the implementation of an entirely new network from scratch.
## Project Architecture

### Header Files

A typical C++ codebase separates class declarations from their implementations. In this project, the `include` directory contains the header files, where classes are defined together with their member variables and method declarations.

Since header files primarily describe the structure and interface of the classes, they are generally shorter and include detailed comments explaining the purpose and functionality of each component.

### Source Files

The implementation of the methods declared in the header files can be found in the `src` directory. Source files may also contain additional helper functions that are not required outside of a specific implementation and are therefore kept local to the corresponding source file.

This separation between declarations and implementations improves code organization, readability, and maintainability.

### Tests

The project includes unit tests written using `doctest`, a fast and lightweight testing framework for C++. The tests are located in `tests/tests.cpp`.

These tests are designed to verify the correctness of the implemented classes and methods, helping identify potential bugs, unexpected behaviour, or missing edge cases during development.

### Data

The MNIST dataset is freely available online; however, to simplify the setup process, the required dataset files are included directly in the `data` directory.

Users are advised not to modify or move these files. The dataset is stored in a binary format, and editing it with a standard text editor may corrupt the data and lead to unexpected behaviour when it is loaded by the program.

Additionally, the `data` directory is automatically copied into the build directory during compilation. Moving the folder without updating the corresponding `CMakeLists.txt` configuration will therefore result in build errors.

### Models

Since this project is primarily designed for educational purposes, performance and training speed have not been the main objectives. Consequently, training a model on the complete MNIST dataset may require a significant amount of time.

The example program located in `src/main.cpp` trains and evaluates a model using, by default, only a subset of the dataset to reduce execution time. However, users can modify the configuration to train on any number of images, up to the full dataset size of 60,000 training samples.

Increasing the number of training samples generally improves model accuracy. For convenience, a pre-trained model obtained by training on the complete MNIST dataset is provided in the `models` directory.

## Execution Flow

### Tensor Creation
### Forward Pass
### Computational Graph Construction
### Loss Computation
### Topological Sorting
### Backward Pass
### Gradient Accumulation
### Optimizer Step
### Gradient Reset
