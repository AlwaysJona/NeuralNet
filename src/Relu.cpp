#include "Relu.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>

#include "Tensor.h"

std::shared_ptr<Node> Relu::forward(std::shared_ptr<Node> input) {
  Tensor t_input(std::move(input));
  auto shape = t_input.shape();
  // Scalar
  if (shape.size() == 0) {
    float result = std::max(std::abs(t_input.item()), 0.0f);
    // gradient logic
    if (t_input.requires_grad()) {
      std::vector<std::shared_ptr<Node>> parents = {t_input.node()};
      std::function<void(const std::vector<float>&)> gradfn =
          [t_input](const std::vector<float>& grad_output) mutable {
            std::vector<float> grad_input;
            if (t_input.item() > 0) {
              grad_input = grad_output;
            } else {
              grad_input.push_back(0.0f);
            }
            t_input.add_to_grad(grad_input);
          };
      Tensor t_output(result, true, gradfn, parents);
      return t_output.node();
    }
    Tensor output(result);
    return output.node();
  }
  // 1D
  if (shape.size() == 1) {
    std::vector<float> result;
    for (auto d : t_input.data()) {
      result.push_back(std::max(std::abs(d), 0.0f));
    }
    // gradient logic
    if (t_input.requires_grad()) {
      std::vector<std::shared_ptr<Node>> parents{t_input.node()};
      std::function<void(const std::vector<float>&)> gradfn =
          [t_input, shape](const std::vector<float>& grad_output) mutable {
            std::vector<float> grad_input;
            for (std::size_t i = 0; i < shape[0]; ++i) {
              if (t_input(i) > 0) {
                grad_input.push_back(grad_output[i]);
              } else {
                grad_input.push_back(0);
              }
            }
            t_input.add_to_grad(grad_input);
          };
      Tensor t_output(result, true, gradfn, parents);
      return t_output.node();
    }
    Tensor output(result);
    return output.node();
  }
  // 2D
  else {
    std::vector<float> result;
    for (auto d : t_input.data()) {
      result.push_back(std::max(std::abs(d), 0.0f));
    }
    // gradient logic
    if (t_input.requires_grad()) {
      std::vector<std::shared_ptr<Node>> parents{t_input.node()};
      std::function<void(const std::vector<float>&)> gradfn =
          [t_input](const std::vector<float>& grad_output) mutable {
            std::vector<float> grad_input;
            for (std::size_t i = 0; i < t_input.size(); ++i) {
              if (t_input.data()[i] > 0) {
                grad_input.push_back(grad_output[i]);
              } else {
                grad_input.push_back(0.0f);
              }
            }
            t_input.add_to_grad(grad_input);
          };
      Tensor t_output(result, shape, true, gradfn, parents);
      return t_output.node();
    }
    Tensor output(result, shape);
    return output.node();
  }
}
