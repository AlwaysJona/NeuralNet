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
  - [Dataset Creation](#dataset-creation)
  - [Forward Pass](#forward-pass)
  - [Memory Management](#memory-management)
  - [Computational Graph Construction](#computational-graph-construction)
  - [Loss Computation](#loss-computation)
  - [Backward Pass](#backward-pass)
  - [Gradient Accumulation](#gradient-accumulation)
  - [Optimizer Step](#optimizer-step)
  - [Gradient Reset](#gradient-reset)

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
<p align="center">
  <img src="/images/test.jpeg" width="250">
</p>
[Back to top](#neuralnet)


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

[Back to top](#neuralnet)
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

[Back to top](#neuralnet)
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

[Back to top](#neuralnet)
## Project Architecture

### Header Files

A typical C++ codebase separates class declarations from their implementations. In this project, the `include` directory contains the header files, where classes are defined together with their member variables and method declarations.

Since header files primarily describe the structure and interface of the classes, they are generally shorter and include detailed comments explaining the purpose and functionality of each component.

[Back to top](#neuralnet)
### Source Files

The implementation of the methods declared in the header files can be found in the `src` directory. Source files may also contain additional helper functions that are not required outside of a specific implementation and are therefore kept local to the corresponding source file.

This separation between declarations and implementations improves code organization, readability, and maintainability.

[Back to top](#neuralnet)
### Tests

The project includes unit tests written using `doctest`, a fast and lightweight testing framework for C++. The tests are located in `tests/tests.cpp`.

These tests are designed to verify the correctness of the implemented classes and methods, helping identify potential bugs, unexpected behaviour, or missing edge cases during development.

[Back to top](#neuralnet)
### Data

The MNIST dataset is freely available online; however, to simplify the setup process, the required dataset files are included directly in the `data` directory.

Users are advised not to modify or move these files. The dataset is stored in a binary format, and editing it with a standard text editor may corrupt the data and lead to unexpected behaviour when it is loaded by the program.

Additionally, the `data` directory is automatically copied into the build directory during compilation. Moving the folder without updating the corresponding `CMakeLists.txt` configuration will therefore result in build errors.

[Back to top](#neuralnet)
### Models

Since this project is primarily designed for educational purposes, performance and training speed have not been the main objectives. Consequently, training a model on the complete MNIST dataset may require a significant amount of time.

The example program located in `src/main.cpp` trains and evaluates a model using, by default, only a subset of the dataset to reduce execution time. However, users can modify the configuration to train on any number of images, up to the full dataset size of 60,000 training samples.

Increasing the number of training samples generally improves model accuracy. For convenience, a pre-trained model obtained by training on the complete MNIST dataset is provided in the `models` directory.

[Back to top](#neuralnet)
## Execution Flow

The example program in `src/main.cpp` provides a good starting point for understanding how the framework operates. By following the sequence of function calls, it is possible to trace the complete execution flow, from loading the dataset to training the neural network. This section describes each stage of the pipeline in the order in which it is executed.

[Back to top](#neuralnet)
### Dataset Creation

The execution begins by calling the `train_new_mnist()` function. This function creates an `MNIST` object, defined in `Dataset.h`, which reads the MNIST dataset from disk and stores the images and labels in memory.

The image data is stored as a three-dimensional `std::vector<float>`, representing a collection of grayscale images, while the corresponding labels are stored in a `std::vector<int>`. Individual samples can be accessed through the `get_item(int index)` member function, which returns a pair containing the image label and a pointer to the corresponding `Node` representing the input tensor.

Once the dataset has been loaded, the `MNIST` object is passed to a `Dataloader`, together with the desired batch size. The `Dataloader` is responsible for grouping samples into mini-batches.


[Back to top](#neuralnet)
### Forward Pass

During each iteration of the training loop, every image in the current mini-batch is passed through the neural network by invoking the model's `forward()` method. The input tensor is first flattened into a one-dimensional vector before being processed by three fully connected (`Linear`) layers. A `ReLU` activation function is applied after the first two linear transformations, while the final layer produces the raw output scores, commonly referred to as *logits*, for the ten digit classes. 

[Back to top](#neuralnet)
### Memory Management

Throughout the framework, tensors and modules return `std::shared_ptr<Node>` objects rather than `Node` instances. A `std::shared_ptr` is a smart pointer that enables multiple objects to share ownership of the same `Node`, avoiding unnecessary copies as tensors are passed between operations.

This ownership model is particularly well suited to the computational graph. During the forward pass, each `Node` stores pointers to its parent nodes while simultaneously acting as a parent for the nodes created by subsequent operations. As a result, the same `Node` may be referenced from multiple locations within the graph, making shared ownership a natural choice.

The lifetime of each `Node` is managed automatically through reference counting. A `Node` remains alive as long as at least one `std::shared_ptr` points to it, and is automatically destroyed once all references have gone out of scope. This eliminates the need for manual memory management while ensuring that the computational graph remains valid throughout the forward and backward passes.

[Back to top](#neuralnet)
### Computational Graph Construction

During the forward pass, every tensor operation creates a new `Node` and records the nodes that contributed to its computation. As a result, the entire sequence of operations is represented as a **Directed Acyclic Graph (DAG)**, where each node corresponds to a tensor and each edge represents a dependency between two operations.

Before gradients can be propagated, the graph must be traversed in an order that guarantees every node is processed only after all of its descendants have been considered. This is achieved by the `build_topo()` function, defined in `Tensor.cpp`:

```cpp
void Tensor::build_topo(const std::shared_ptr<Tensor::Node>& node,
                        std::unordered_set<Tensor::Node*>& visited,
                        std::vector<std::shared_ptr<Tensor::Node>>& topo) {
  // if node was already visited, exit loop
  if (visited.find(node.get()) != visited.end()) {
    return;
  }

  // if not, record it as visited
  visited.insert(node.get());

  for (const auto& parent : node->m_parents) {
    build_topo(parent, visited, topo);
  }

  topo.push_back(node);
}
```

The function performs a recursive **Depth-First Search (DFS)** of the computational graph. An `std::unordered_set` is used to keep track of the nodes that have already been visited, ensuring that each node is explored exactly once, even when it contributes to multiple downstream operations. After all of a node's parents have been visited, the node itself is appended to the `topo` vector, producing a valid topological ordering of the graph.

As an example, consider the following code:

```cpp
Tensor A(data);
Tensor B = 2 * A;
Tensor C = 4 * A;
Tensor D = B + C;
```

The corresponding computational graph is:

```text
    A
   / \
  B   C
   \ /
    D
```

Performing a depth-first traversal produces the topological ordering:

```text
A -> B -> C -> D
```

During backpropagation, this ordering is simply traversed in reverse (`D -> C -> B -> A`), ensuring that the gradient of every child node has already been computed before propagating it to its parents. This reverse traversal naturally implements the chain rule across the entire computational graph.

[Back to top](#neuralnet)
### Loss Computation

Once the network has produced its output, the resulting logits are passed to the `Cross_Entropy_Loss` module together with the ground-truth class label. The loss implementation follows the standard definition of the cross-entropy loss by composing two simpler modules: a `Softmax` module and a `NLL_Loss` (Negative Log-Likelihood) module.

The `Softmax` module first converts the raw logits into a probability distribution over the output classes. The resulting probabilities are then passed to the `NLL_Loss` module, which extracts the probability corresponding to the target class and computes the scalar loss as

$$
L = -\log(p_{\text{target}}),
$$

where $p_{\text{target}}$ is the predicted probability assigned to the correct class. To improve numerical stability, the probability is clamped to a minimum value of `1e-12` before applying the logarithm, preventing undefined values caused by computing $\log(0)$.

As with every differentiable operation in the framework, the loss tensor stores its own gradient function together with a pointer to its parent node. During backpropagation, the gradient function computes

$$
\frac{\partial L}{\partial p_{\text{target}}}
=
-\frac{1}{p_{\text{target}}},
$$

while assigning a zero gradient to all remaining classes. Since the loss is built by composing the `Softmax` and `NLL_Loss` modules, the automatic differentiation engine subsequently propagates this gradient through the Softmax operation and back into the network, allowing the chain rule to compute the gradients of every trainable parameter.

During training, the loss computed for each sample in the mini-batch is accumulated and averaged, producing a single scalar value that represents the prediction error of the entire batch.

[Back to top](#neuralnet)
### Backward Pass

Once the batch loss has been computed, the automatic differentiation engine is triggered by calling the `backward()` member function on the loss tensor. Since the loss is a scalar, the parameterless overload of `backward()` is used, which initializes the backward pass by seeding the gradient of the output with the value `1.0`. Mathematically, this corresponds to

$$
\frac{\partial L}{\partial L} = 1,
$$

which serves as the starting point for all subsequent gradient computations.

The `backward(const std::vector<float>&)` overload first constructs a topological ordering of the computational graph using the `build_topo()` function. Before propagating any gradients, the framework iterates through every node in the graph and resets its stored gradient to zero, ensuring that gradients from previous backward passes do not interfere with the current computation.

After assigning the seed gradient to the loss node, the computational graph is traversed in **reverse topological order**. For each node, the framework checks whether a local gradient function has been stored. If so, the function is invoked with the node's accumulated gradient, computing the local derivatives of the operation that created the node and propagating the resulting contributions to its parent nodes.

Because every differentiable operation stores its own gradient function, the backward pass is completely generic: tensor addition, matrix multiplication, activation functions, and loss functions all contribute their local derivatives through the same interface. By recursively applying these local gradient functions while traversing the graph in reverse, the framework automatically implements the chain rule and computes the gradient of the loss with respect to every trainable parameter in the network.


[Back to top](#neuralnet)
### Gradient Accumulation

As gradients are propagated through the graph, each parameter accumulates the contributions received from every path leading to it. This behaviour is particularly important because a single parameter may influence the loss through multiple operations. Once the backward pass is complete, every trainable weight and bias stores the gradient of the batch loss with respect to that parameter.


[Back to top](#neuralnet)
### Optimizer Step

The optimizer is declared in `SGD.h` and implemented in `SGD.cpp`. The `SGD` class is responsible for updating the trainable parameters of the model after the gradients have been computed during the backward pass.

Internally, the `SGD` object stores a list of parameters as a private member variable. This list is defined through a type alias as a `std::vector<std::pair<std::string, std::shared_ptr<Node>>>`, where each element represents a trainable parameter of the network, such as the weights and biases associated with the different layers. Since the optimization step only requires access to these parameters, storing references to them allows the optimizer to directly update their values without requiring access to the entire model structure.

During training, the optimizer applies the Stochastic Gradient Descent (SGD) update rule to each parameter. The goal of the optimization process is to minimise the loss function by adjusting the model parameters in the direction opposite to their gradients. The magnitude of each update is controlled by the learning rate, which acts as a scaling factor:

$$
\theta_{new} = \theta_{old} - \eta \frac{\partial L}{\partial \theta}
$$

where $\theta$ represents a trainable parameter, $L$ is the loss function, and $\eta$ is the learning rate. In the current implementation, the default learning rate is set to `0.001f`.

By repeatedly applying this update after each backward pass, the optimizer gradually adjusts the model parameters towards values that minimise the loss function.


[Back to top](#neuralnet)
### Gradient Reset

Once the optimization step has been completed, the stored gradients are reset to zero before processing the next mini-batch. Clearing the gradients is necessary because they are accumulated during the backward pass; failing to reset them would cause gradients from previous iterations to be added to those of subsequent batches, resulting in incorrect parameter updates.

[Back to top](#neuralnet)
