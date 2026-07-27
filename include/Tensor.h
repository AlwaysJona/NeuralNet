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

class Tensor {
 public:
  struct Node {
    std::vector<float> m_data;
    std::vector<std::size_t> m_shape;  // empty if scalar, 1 element if 1D,
                                       // 2 elements if 2D (rows and cols)
    std::size_t m_stride;  // stride = 1 for 1D tensor, for 2D Tensor stride
                           // = num of cols
    std::vector<float> m_grad;  // same size as m_data

    bool m_requires_grad;

    // updates gradient of parent based on gradient of child
    std::function<void(const std::vector<float> &)> m_gradfn;

    std::vector<std::shared_ptr<Node>> m_parents;

    // constructor
    Node(std::vector<float> data, std::vector<std::size_t> shape,
         bool requires_grad);
  };

  // constructors
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
  const std::size_t size() const {
    return m_node->m_data.size();
  }  // total number of elements in the tensor
  const std::vector<std::size_t> &shape() const { return m_node->m_shape; }
  const std::size_t &stride() const { return m_node->m_stride; }
  const std::vector<float> &data() const { return m_node->m_data; }
  const bool requires_grad() const { return m_node->m_requires_grad; }
  const std::vector<float> &grad() const { return m_node->m_grad; }
  std::shared_ptr<Node> node() const { return m_node; }

  // setters
  // TODO: need to add some control on shape and size for set_data
  void set_data(std::vector<float> data) { m_node->m_data = data; }

  // operations
  Tensor operator+(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;  // element wise multiplication
  Tensor operator-(const Tensor &other) const;
  Tensor matmul(const Tensor &other) const;  // matrix multiplication
  Tensor sum() const;  // accumulates all the values of a higher dimensional
                       // tensor, returns a Scalar tensor

  void add_to_grad(const std::vector<float> &grad_update);
  void zero_grad();
  // bacward propagation
  void backward();
  void backward(const std::vector<float> &seed_grad);
  friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);

 private:
  std::shared_ptr<Node> m_node;

  // topological sort helper for faster navigation
  static void build_topo(const std::shared_ptr<Tensor::Node> &Node,
                         std::unordered_set<Tensor::Node *> &visited,
                         std::vector<std::shared_ptr<Tensor::Node>> &topo);
};

#endif
