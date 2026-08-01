#include "Softmax.h"

#include <algorithm>
#include <memory>
#include <random>

#include "Tensor.h"

std::shared_ptr<Node> Softmax::forward(std::shared_ptr<Node> input) {
  Tensor input_t(input);
  // scalar
  if (input_t.shape().size() == 0) {
    float result = 1.0f;
    // gradient logic
    if (input_t.requires_grad()) {
      std::vector<std::shared_ptr<Node>> parents{input};
      std::function<void(const std::vector<float>&)> gradfn =
          [input_t](const std::vector<float>& grad_output) mutable {
            // for scalar the grad of softmax is 0
            std::vector<float> grad_input{0.0f};
            input_t.add_to_grad(grad_input);
          };
      Tensor output_t(result, true, gradfn, parents);
      return output_t.node();
    }
    Tensor output(result);
    return output.node();
  }
  // 1D (no need for 2D as the output will already have been flattened)
  if (input_t.shape().size() == 1) {
    auto data = input_t.data();
    // max_val needed to prevent divergence
    float max_val = *(std::max_element(data.begin(), data.end()));
    std::vector<float> s;
    float sum_exp = 0.0f;
    for (auto& d : data) {
      sum_exp += std::exp(d - max_val);
    }
    for (auto& d : data) {
      s.push_back((std::exp(d - max_val)) / sum_exp);
    }
    // gradient logic
    if (input_t.requires_grad()) {
      std::vector<std::shared_ptr<Node>> parents{input};
      std::function<void(const std::vector<float>&)> gradfn =
          [input_t, s](const std::vector<float>& grad_output) mutable {
            std::vector<float> grad_input;
            for (std::size_t j = 0; j < input_t.size(); ++j) {
              float grad_j = 0;
              for (std::size_t i = 0; i < grad_output.size(); ++i) {
                if (i == j) {
                  grad_j += (grad_output[i] * (s[i] * (1 - s[i])));
                } else {
                  grad_j += (grad_output[i] * (-s[i] * s[j]));
                }
              }
              grad_input.push_back(grad_j);
            }
            input_t.add_to_grad(grad_input);
          };
      Tensor output_t(s, true, gradfn, parents);
      return output_t.node();
    }
    Tensor output(s);
    return output.node();
  }
  throw std::invalid_argument(
    "Softmax only supports scalar and 1D tensors");
}
