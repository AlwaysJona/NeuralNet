#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

// The Tensor class is at the core of the library, operations are excecuted on
// Tensor objects, that store pixel values for each image at the start, and 
// the values for the subsequent transformations.
// 
// The Tensor class itself is an interface for the Node struct, which stores the
// actual data, the Tensor class only holds a pointer to a specific Node, so it 
// can read and write the data.
//
// Implementation in Tensor.cpp

class Tensor {
 public:
  struct Node {
    // container for the matrix elements 
    std::vector<float> m_data;
    
    // empty if scalar, 1 element if 1D, 2 elements if 2D (rows and cols)
    std::vector<std::size_t> m_shape; 

    // stride = 1 for 1D tensor, for 2D Tensor stride = num of cols
    std::size_t m_stride;  

    // gradient of node, same size as m_data
    std::vector<float> m_grad;  

    // tracks if the Node requires gradient computation
    bool m_requires_grad;

    // updates gradient of parent based on gradient of child
    std::function<void(const std::vector<float> &)> m_gradfn;

    // vector storing parent Nodes, to which gradients will
    // need to be propagated if m_requires_grad = true
    std::vector<std::shared_ptr<Node>> m_parents;

    // Node constructor
    Node(std::vector<float> data, std::vector<std::size_t> shape,
         bool requires_grad);
  };

  // Tensor constructors, multiple overloads for different argument
  // types
  explicit Tensor(
      const float data, bool requires_grad = false,
      std::function<void(const std::vector<float> &)> gradfn = nullptr,
      std::vector<std::shared_ptr<Node>> parents = {});  // scalar

  explicit Tensor(
      const std::vector<float> &data, bool requires_grad = false,
      std::function<void(const std::vector<float> &)> gradfn = nullptr,
      std::vector<std::shared_ptr<Node>> parents = {});  // 1D tensor

  explicit Tensor(
      const std::vector<std::vector<float>> &data, bool requires_grad = false,
      std::function<void(const std::vector<float> &)> gradfn = nullptr,
      std::vector<std::shared_ptr<Node>> parents = {});  // 2D tensor

  explicit Tensor(std::shared_ptr<Node> node) : m_node(std::move(node)) {}

  // flat Tensor with specified shape
  Tensor(const std::vector<float> &data, const std::vector<std::size_t> &shape,
         bool requires_grad = false,
         std::function<void(const std::vector<float> &)> gradfn = nullptr,
         std::vector<std::shared_ptr<Node>> parents = {});

  // default constructor
  Tensor() { Tensor(0.0f); }

  // returns the only item for scalars
  const float &item() const;  // read only
  float &item();              // write

  // operator() to access elements: data(0) returns first element of data
  const float &operator()(const std::size_t i) const;  // 1D read only
  float &operator()(const std::size_t i);              // 1D write
  const float &operator()(const std::size_t i,
                          const std::size_t j) const;           // 2D read only
  float &operator()(const std::size_t i, const std::size_t j);  // 2D write

  // getters
  // total number of elements in the tensor
  const std::size_t size() const {
    return m_node->m_data.size();
  }
  const std::vector<std::size_t> &shape() const { return m_node->m_shape; }
  const std::size_t &stride() const { return m_node->m_stride; }
  const std::vector<float> &data() const { return m_node->m_data; }
  std::vector<float> &data() { return m_node->m_data; }
  const bool requires_grad() const { return m_node->m_requires_grad; }
  const std::vector<float> &grad() const { return m_node->m_grad; }
  std::shared_ptr<Node> node() const { return m_node; }

  // setters
  void set_data(const std::vector<float>& data);
  void set_node(std::shared_ptr<Node> node) { m_node = node; }

  // operations
  Tensor operator+(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;  // element wise multiplication
  Tensor operator-(const Tensor &other) const;
  Tensor matmul(const Tensor &other) const;  // matrix multiplication
  Tensor sum() const;  // accumulates all the values of a higher dimensional
                       // tensor, returns a Scalar tensor

  // utility methods
  // element wise addtion to gradient
  void add_to_grad(const std::vector<float> &grad_update);
  // set the gradient elements to 0
  void zero_grad();
  // return the index of the largest element
  std::size_t argmax() const;

  // backward propagation of gradients up the Node tree.
  // Starts from child node, with a seed tensor if given,
  // otherwise it is set to a vector of 1s
  void backward();
  void backward(const std::vector<float> &seed_grad);

  // "friend" keyword let's an std::ostream object call and access Tensor
  // private elements so we can output data conventionally using
  // std::cout << obj
  friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);

 private:
  std::shared_ptr<Node> m_node;

  // topological sort helper for faster navigation
  // Starts from leaf Node, adds it to visited, then recursively adds
  // its parents to visted, stops when there are no parents, or the
  // Node is already in visited.
  // The resulting graph will then be traversed in reverse in function
  // backwards()
  static void build_topo(const std::shared_ptr<Tensor::Node> &Node,
                         std::unordered_set<Tensor::Node *> &visited,
                         std::vector<std::shared_ptr<Tensor::Node>> &topo);
};

#endif
