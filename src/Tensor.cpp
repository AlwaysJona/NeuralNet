#include "Tensor.h"

std::size_t compute_stride(const std::vector<std::size_t>& shape) {
  std::size_t result;

  if (shape.empty()) {
    result = 0;
  }

  if (shape.size() == 1) {
    result = 1;
  }

  if (shape.size() == 2) {
    result = shape[1];
  }

  return result;
}

Tensor::Node::Node(std::vector<float> data, std::vector<std::size_t> shape,
                   bool requires_grad)
    : m_data(std::move(data)),
      m_shape(std::move(shape)),
      m_stride(compute_stride(m_shape)),
      m_grad(m_data.size(), 0.0f),
      m_requires_grad(requires_grad),
      m_gradfn([](const std::vector<float>& grad_output) {}),
      m_parents{} {}

Tensor::Tensor(const float data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Node>> parents)
    : m_node(std::make_shared<Node>(std::vector<float>{data},
                                    std::vector<std::size_t>{},
                                    requires_grad)) {
  if (requires_grad) {
    zero_grad();
  }

  m_node->m_gradfn = gradfn;
  m_node->m_parents = parents;
}

Tensor::Tensor(const std::vector<float>& data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Node>> parents) {
  if (data.empty()) {
    throw std::invalid_argument("Cannot create Tensor from empty vector");
  }

  m_node = std::make_shared<Node>(data, std::vector<size_t>{data.size()},
                                  requires_grad);

  if (requires_grad) {
    zero_grad();
  }

  m_node->m_gradfn = gradfn;
  m_node->m_parents = parents;
}

Tensor::Tensor(const std::vector<std::vector<float>>& data, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Node>> parents) {
  if (data.empty()) {
    throw std::invalid_argument("Can't create 2D Tensor from empty vector");
  }
  if (data[0].empty()) {
    throw std::invalid_argument("Can't create 2D Tensor with no columns");
  }

  std::vector<std::size_t> shape = {data.size(), data[0].size()};
  auto stride = data[0].size();
  std::vector<float> node_data;

  // Check if all the rows of the tensor have the same size
  for (std::size_t i = 0; i < data.size(); ++i) {
    if (data[i].size() != stride) {
      throw std::invalid_argument("Dimensions are inconsistent.");
    }
    // Fill up the tensor data with the input data in a row major order
    for (std::size_t j = 0; j < data[i].size(); ++j) {
      node_data.push_back(data[i][j]);
    }
  }

  m_node = std::make_shared<Node>(node_data, shape, requires_grad);

  if (requires_grad) {
    zero_grad();
  }

  m_node->m_gradfn = gradfn;
  m_node->m_parents = parents;
}

Tensor::Tensor(const std::vector<float>& data,
               const std::vector<std::size_t>& shape, bool requires_grad,
               std::function<void(const std::vector<float>&)> gradfn,
               std::vector<std::shared_ptr<Node>> parents) {
  if (shape.size() > 2) {
    throw std::invalid_argument(
        "Only scalar, 1D and 2D tensors are currently supported");
  }

  std::size_t expected_size = 1;
  for (auto dim : shape) {
    expected_size *= dim;
  }

  if (expected_size != data.size()) {
    throw std::invalid_argument("Data size does not match tensor shape");
  }

  m_node = std::make_shared<Node>(data, shape, requires_grad);

  if (requires_grad) {
    zero_grad();
  }

  m_node->m_gradfn = gradfn;
  m_node->m_parents = parents;
}

const float& Tensor::item() const {
  // works only with scalars and 1D tensors
  if (m_node->m_data.size() == 1) {
    return m_node->m_data[0];
  } else {
    throw std::runtime_error(
        "item() can only be called on tensors with a single element");
  }
}

float& Tensor::item() {
  // works only with scalars and 1D tensors
  if (m_node->m_data.size() == 1) {
    return m_node->m_data[0];
  } else {
    throw std::runtime_error(
        "item() can only be called on tensors with a single element");
  }
}

const float& Tensor::operator()(const std::size_t i) const {
  if (m_node->m_shape.size() == 0) {
    throw std::invalid_argument(
        "Can't index into a scalar, use item() instead");
  }
  if (m_node->m_shape.size() == 1) {
    if (i >= m_node->m_shape[0]) {
      throw std::invalid_argument("Index " + std::to_string(i) +
                                  " is out of bounds for array of size " +
                                  std::to_string(m_node->m_shape[0]));
    }
    return m_node->m_data[i];
  }
  throw std::invalid_argument("Expected 1D tensor");
}

float& Tensor::operator()(const std::size_t i) {
  if (m_node->m_shape.size() == 0) {
    throw std::invalid_argument(
        "Can't index into a scalar, use item() instead");
  }
  if (m_node->m_shape.size() == 1) {
    if (i >= m_node->m_shape[0]) {
      throw std::invalid_argument("Index " + std::to_string(i) +
                                  " is out of bounds for array of size " +
                                  std::to_string(m_node->m_shape[0]));
    }
    return m_node->m_data[i];
  }
  throw std::invalid_argument(
      "Only one index given, can only index into a 1D Tensor");
}

const float& Tensor::operator()(const std::size_t i,
                                const std::size_t j) const {
  // the operator should only be called on 2D tensors
  if (m_node->m_shape.size() < 2) {
    throw std::invalid_argument(
        "2 indexes given, but Tensor has fewer dimensions");
  } else {
    if (i >= m_node->m_shape[0] || j >= m_node->m_shape[1]) {
      throw std::invalid_argument(
          "Out of bounds, shape is " + std::to_string(m_node->m_shape[0]) +
          " " + std::to_string(m_node->m_shape[1]) + " but given indeces are " +
          std::to_string(i) + " " + std::to_string(j));
    } else {
      return m_node->m_data[i * m_node->m_stride + j];
    }
  }
}

float& Tensor::operator()(const std::size_t i, const std::size_t j) {
  if (m_node->m_shape.size() < 2) {
    throw std::invalid_argument(
        "2 indexes given, but Tensor has fewer dimensions");
  } else {
    if (i >= m_node->m_shape[0] || j >= m_node->m_shape[1]) {
      throw std::invalid_argument(
          "Out of bounds, shape is " + std::to_string(m_node->m_shape[0]) +
          " " + std::to_string(m_node->m_shape[1]) + " but given indeces are " +
          std::to_string(i) + " " + std::to_string(j));
    } else {
      return m_node->m_data[i * m_node->m_stride + j];
    }
  }
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  std::string output = "[";
  auto size = obj.size();
  auto shape = obj.shape();
  auto dim = shape.size();

  if (dim == 0) {
    output += std::to_string(obj.item());
  } else if (dim == 1) {
    for (auto i = 0; i < size; ++i) {
      output += std::to_string(obj(i));
      if (i != size - 1) {
        output += ", ";
      }
    }
  } else {
    for (auto i = 0; i < size; ++i) {
      if ((i + 1) % shape[1] == 1) {
        output += "[";
      }
      int row = i / shape[1];
      int col = i % shape[1];
      output += std::to_string(obj(row, col));
      if ((i + 1) % shape[1] == 0) {
        output += "]";
      }
      if (i != size - 1) {
        output += ", ";
      }
    }
  }
  output += "]";
  os << output;
  return os;
}

Tensor Tensor::operator+(const Tensor& other) const {
  auto other_size = other.size();
  const bool this_scalar = m_node->m_shape.empty();
  const bool other_scalar = other.shape().empty();

  std::vector<float> result;
  std::vector<std::size_t> new_shape;

  // addition logic for sums of Tensors

  if (this_scalar) {
    if (other_scalar) {  // scalar + scalar
      result.push_back(item() + other.item());
    } else {  // scalar + 1D or scalar + 2D
      for (std::size_t i = 0; i < other_size; ++i) {
        result.push_back(item() + other.data()[i]);
      }
      new_shape = other.shape();
    }
  } else if (other_scalar) {  // 1D + scalar or 2D + scalar
    for (std::size_t i = 0; i < size(); ++i) {
      result.push_back(m_node->m_data[i] + other.item());
    }
    new_shape = m_node->m_shape;
  } else if (m_node->m_shape == other.shape()) {  // 1D + 1D or 2D + 2D
    for (std::size_t i = 0; i < size(); ++i) {
      result.push_back(m_node->m_data[i] + other.data()[i]);
    }
    new_shape = m_node->m_shape;
  } else {  // 1D + 2D or 2D + 1D, or higher dimensions
    throw std::invalid_argument("These Tensor shapes can't be added");
  }

  // gradient logic

  auto this_node = m_node;
  auto other_node = other.m_node;
  const bool req_grad =
      this_node->m_requires_grad || other_node->m_requires_grad;

  Tensor out(result, new_shape, req_grad);

  if (req_grad) {
    auto out_node = out.m_node;

    out_node->m_parents = {this_node, other_node};
    out_node->m_gradfn = [this_node, other_node, out_node, this_scalar,
                          other_scalar](const std::vector<float>& grad_output) {
      for (std::size_t i = 0; i < out_node->m_data.size(); ++i) {
        const float g = grad_output[i];

        // c = a + b
        // dc/da = 1    and    dc/db = 1
        // if Loss = c
        // dLoss/da = dLoss/dc * dc/da
        // dLoss/db = dLoss/dc * dc/db
        // in this context dLoss/dc = g

        // propagate gradient to parents
        if (this_node->m_requires_grad) {
          if (this_scalar) {
            this_node->m_grad[0] += g;
          } else {
            this_node->m_grad[i] += g;
          }
        }

        if (other_node->m_requires_grad) {
          if (other_scalar) {
            other_node->m_grad[0] += g;
          } else {
            other_node->m_grad[i] += g;
          }
        }
      }
    };
  }
  return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
  auto other_size = other.size();
  const bool this_scalar = m_node->m_shape.empty();
  const bool other_scalar = other.shape().empty();

  std::vector<float> result;
  std::vector<std::size_t> new_shape;

  // multiplication logic
  if (this_scalar) {
    if (other_scalar) {
      result.push_back(item() * other.item());
    } else {
      for (std::size_t i = 0; i < other_size; ++i) {
        result.push_back(item() * other.data()[i]);
      }
      new_shape = other.shape();
    }
  } else if (other_scalar) {
    for (std::size_t i = 0; i < size(); ++i) {
      result.push_back(m_node->m_data[i] * other.item());
    }
    new_shape = m_node->m_shape;
  } else if (m_node->m_shape == other.shape()) {
    for (std::size_t i = 0; i < size(); ++i) {
      result.push_back(m_node->m_data[i] * other.data()[i]);
    }
    new_shape = m_node->m_shape;
  } else {
    throw std::invalid_argument(
        "These Tensor shapes are incompatible for element wise "
        "multiplication");
  }

  auto this_node = m_node;
  auto other_node = other.m_node;

  const bool req_grad =
      this_node->m_requires_grad || other_node->m_requires_grad;
  Tensor out(result, new_shape, req_grad);

  // gradient logic
  if (req_grad) {
    auto out_node = out.m_node;

    out_node->m_parents = {this_node, other_node};

    out_node->m_gradfn = [this_node, other_node, out_node, this_scalar,
                          other_scalar](const std::vector<float>& grad_output) {
      for (std::size_t i = 0; i < out_node->m_data.size(); ++i) {
        const float g = grad_output[i];

        // c = a * b
        // dc/da = b    and    dc/db = a
        // and if Loss = c
        // dLoss/da = dLoss/dc * dc/da
        // and
        // dLoss/db = dLoss/dc * dc/db
        // in this contex dLoss/dc = g

        // propagate gradient to parents
        const float this_value =
            this_scalar ? this_node->m_data[0] : this_node->m_data[i];
        const float other_value =
            other_scalar ? other_node->m_data[0] : other_node->m_data[i];

        if (this_node->m_requires_grad) {
          if (this_scalar) {
            this_node->m_grad[0] += g * other_value;
          } else {
            this_node->m_grad[i] += g * other_value;
          }
        }

        if (other_node->m_requires_grad) {
          if (other_scalar) {
            other_node->m_grad[0] += g * this_value;
          } else {
            other_node->m_grad[i] += g * this_value;
          }
        }
      }
    };
  }
  return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
  return (*this) + Tensor(-1) * other;
}

Tensor Tensor::matmul(const Tensor& other) const {
  auto this_dims = m_node->m_shape.size();
  auto other_dims = other.shape().size();

  if (this_dims == 0 || other_dims == 0) {
    throw std::invalid_argument("Both arguments need to be at least 1D");
  }
  if (m_node->m_shape[this_dims - 1] != other.shape()[0]) {
    throw std::invalid_argument(
        "The last dimension of the 1st tensor isn't the same as the first "
        "dimension of the 2nd tensor");
  }

  std::vector<float> result;
  std::vector<std::size_t> new_shape;
  // 1D x 1D = Scalar
  if (this_dims == 1 && other_dims == 1) {
    float acc = 0;
    for (std::size_t i = 0; i < size(); ++i) {
      acc += m_node->m_data[i] * other(i);
    }
    result.push_back(acc);
  }
  // 1D x 2D = 1D -> (m) x (m, k) = (k)
  if (this_dims == 1 && other_dims == 2) {
    auto m = size();
    auto k = other.stride();
    new_shape = {k};
    for (std::size_t i = 0; i < k; ++i) {
      float result_i = 0;
      for (std::size_t j = 0; j < m; ++j) {
        result_i += m_node->m_data[j] * other(j, i);
      }
      result.push_back(result_i);
    }
  }
  // 2D x 1D = 1D -> (k,m) x (m) = (k)
  if (this_dims == 2 && other_dims == 1) {
    auto m = m_node->m_stride;
    auto k = shape()[0];
    new_shape = {k};
    for (std::size_t i = 0; i < k; ++i) {
      float result_i = 0;
      for (std::size_t j = 0; j < m; ++j) {
        result_i += (*this)(i, j) * other(j);
      }
      result.push_back(result_i);
    }
  }
  // 2D x 2D = 2D -> (k,m) x (m,l) = (k,l)
  if (this_dims == 2 && other_dims == 2) {
    auto k = shape()[0];
    auto m = m_node->m_stride;
    auto l = other.stride();
    new_shape = {k, l};
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < l; j++) {
        float sum = 0;
        for (std::size_t t = 0; t < m; ++t) {
          sum += (*this)(i, t) * other(t, j);
        }
        result.push_back(sum);
      }
    }
  }

  else if (this_dims > 2 || other_dims > 2) {
    throw std::invalid_argument("matmul only supported for 1D or 2D tensors");
  }

  auto this_node = m_node;
  auto other_node = other.m_node;
  const bool req_grad =
      this_node->m_requires_grad || other_node->m_requires_grad;

  Tensor out(result, new_shape, req_grad);

  // gradient logic
  if (req_grad) {
    auto out_node = out.m_node;

    out_node->m_parents = {this_node, other_node};

    std::function<void(const std::vector<float>&)> gradfn =
        [this_node, other_node, this_dims,
         other_dims](const std::vector<float>& grad_output) {
          // 1D x 1D = Scalar
          if (this_dims == 1 && other_dims == 1) {
            const float g = grad_output[0];
            const std::size_t n = this_node->m_shape[0];

            for (auto i = 0; i < n; ++i) {
              if (this_node->m_requires_grad) {
                this_node->m_grad[i] += g * other_node->m_data[i];
              }

              if (other_node->m_requires_grad) {
                other_node->m_grad[i] += g * this_node->m_data[i];
              }
            }
          }

          // 1D x 2D = 1D -> m x (m,k) = k
          if (this_dims == 1 && other_dims == 2) {
            auto m = this_node->m_shape[0];
            auto k = other_node->m_shape[1];

            for (auto j = 0; j < k; ++j) {
              const float g = grad_output[j];

              for (auto i = 0; i < m; ++i) {
                if (this_node->m_requires_grad) {
                  this_node->m_grad[i] += g * other_node->m_data[i * k + j];
                }

                if (other_node->m_requires_grad) {
                  other_node->m_grad[i * k + j] += g * this_node->m_data[i];
                }
              }
            }
          }

          // 2D x 1D = 1D -> (k,m) x m = k
          if (this_dims == 2 && other_dims == 1) {
            auto m = this_node->m_shape[1];
            auto k = this_node->m_shape[0];

            for (auto j = 0; j < k; ++j) {
              const float g = grad_output[j];

              for (auto i = 0; i < m; ++i) {
                if (this_node->m_requires_grad) {
                  this_node->m_grad[j * m + i] += g * other_node->m_data[i];
                }

                if (other_node->m_requires_grad) {
                  other_node->m_grad[i] += g * this_node->m_data[j * m + i];
                }
              }
            }
          }

          // 2D x 2D = 2D -> (k,m) x (m,l) = (k,l)
          if (this_dims == 2 && other_dims == 2) {
            auto k = this_node->m_shape[0];
            auto m = this_node->m_shape[1];
            auto l = other_node->m_shape[1];

            for (auto j = 0; j < k; ++j) {
              for (auto i = 0; i < l; ++i) {
                const float g = grad_output[j * l + i];

                for (auto t = 0; t < m; ++t) {
                  if (this_node->m_requires_grad) {
                    this_node->m_grad[j * m + t] +=
                        g * other_node->m_data[t * l + i];
                  }

                  if (other_node->m_requires_grad) {
                    other_node->m_grad[t * l + i] +=
                        g * this_node->m_data[j * m + t];
                  }
                }
              }
            }
          }
        };
    out_node->m_gradfn = gradfn;
  }

  return out;
}

Tensor Tensor::sum() const {
  auto parent = m_node;

  float acc =
      std::accumulate(parent->m_data.begin(), parent->m_data.end(), 0.0f);

  Tensor out(acc, parent->m_requires_grad);
  if (parent->m_requires_grad) {
    auto out_node = out.m_node;
    out_node->m_parents = {parent};

    out_node->m_gradfn = [parent,
                          out_node](const std::vector<float>& grad_output) {
      const float g = grad_output[0];

      for (auto& comp : parent->m_grad) {
        comp += g;
      }
    };
  }

  return out;
}

void Tensor::add_to_grad(const std::vector<float>& grad_update) {
  if (m_node->m_requires_grad) {
    if (grad_update.size() == m_node->m_grad.size()) {
      for (auto i = 0; i < grad_update.size(); ++i) {
        m_node->m_grad[i] += grad_update[i];
      }
    } else {
      throw std::runtime_error("Gradient shape mismatch during addition");
    }
  }
}

void Tensor::zero_grad() { m_node->m_grad = std::vector<float>(size(), 0.0f); }

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

// backward propagation for scalar outputs
void Tensor::backward() {
  if (size() != 1) {
    throw std::runtime_error(
        "backward() with no arguments only works for scalar outputs."
        "For vector/matrix outputs, pass a seed gradient");
  }

  backward(std::vector<float>{1.0f});
}

// overload for higher dimensional outputs
void Tensor::backward(const std::vector<float>& seed_grad) {
  if (seed_grad.size() != size()) {
    throw std::runtime_error("Seed size must match tensor size");
  }

  std::vector<std::shared_ptr<Node>> topo;
  std::unordered_set<Node*> visited;

  build_topo(m_node, visited, topo);

  // clear all gradients
  for (auto& node : topo) {
    std::fill(node->m_grad.begin(), node->m_grad.end(), 0.0f);
  }

  // Seed dOutput/dOutput
  m_node->m_grad = seed_grad;

  // Traverse graph backwards
  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->m_gradfn) {
      (*it)->m_gradfn((*it)->m_grad);
    }
  }
}

// return index of the element with the highest value in the vector
std::size_t Tensor::argmax() const {
  return std::distance(data().begin(),
                       std::max_element(data().begin(), data().end()));
}
