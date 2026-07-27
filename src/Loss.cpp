#include "Loss.h"
#include "Module.h"
#include "Softmax.h"
#include "Tensor.h"

std::shared_ptr<Node> Loss::forward(std::shared_ptr<Node> input) {
    throw std::runtime_error("Loss expects both an input and a target \n";
}

std::shared_ptr<Node> Loss::forward(std::shared_ptr<Node> input,
                                    std::size_t target) {
  throw std::runtime_error("forward() of Loss is meant to be overridden \n");
}

std::shared_ptr<Node> Loss::operator()(std::shared_ptr<Node> input,
                                       std::size_t target) {
  return forward(input, target);
}

std::shared_ptr<Node> NLL_Loss::forward(std::shared_ptr<Node> input,
                                        std::size_t target) {
  Tensor input_t(input);
  if (input_t.shape().size() != 1) {
    throw std::runtime_error("NLL Loss expects 1D tensor as input \n");
  }
  if (target >= input_t.size()) {
    throw std::runtime_error("NLL target is out of bounds \n");
  }
  // prevent log(0)
  float prob = std::max(input_t(target), 1e-12f);
  float loss = -std::log(prob);

  if (input_t.requires_grad()) {
    std::vector<std::shared_ptr<Node>> parents{input};
    std::function<void(const std::vector<float>&)> gradfn =
        [input_t, target](const std::vector<float>& grad_output) {
          std::vector<float> grad_input;
          for (std::size_t i = 0; i < input_t.size(); ++i) {
            if (i == target) {
              grad_input.push_back(grad_output[0] * (-1.0f / input_t(i)));
            } else {
              grad_inout.push_back(0.0f);
            }
          }
          input_t.add_to_grad(grad_input);
        };
    Tensor output_t(loss, true, gradfn, parents);
    return loss.node();
  }
  Tensor output(loss);
  return loss.node();
}

std::shared_ptr<Node> Cross_Entropy_Loss::forward(std::shared_ptr<Node> input,
                                                  std::size_t target) {
  Tensor input_t(input);
  if (input_t.shape().size() != 1) {
    throw std::runtime_error(
        "Cross Entropy Loss expects 1D tensor as input \n");
  }
  if (target >= input_t.size()) {
    throw std::runtime_error("Cross Entropy Loss target is out of bounds \n");
  }

  Softmax softmax;
  NLL_Loss nll_loss;
  std::shared_ptr<Node> softmax_output = softmax(input);
  return nll_loss(softmax_output, target);
}
