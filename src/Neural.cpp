#include "Neural.h"

NeuralNetwork::NeuralNetwork() {
  register_module("linear_1", m_lin1);
  register_module("linear_2", m_lin2);
  register_module("linear_3", m_lin3);
}

std::shared_ptr<Node> NeuralNetwork::forward(std::shared_ptr<Node> input) {
  Tensor t_input(std::move(input));
  std::shared_ptr<Node> flat = (*m_flatten)(t_input.node());
  std::shared_ptr<Node> linear_1 = (*m_lin1)(flat);
  std::shared_ptr<Node> relu_1 = (*m_relu)(linear_1);
  std::shared_ptr<Node> linear_2 = (*m_lin2)(relu_1);
  std::shared_ptr<Node> relu_2 = (*m_relu)(linear_2);
  std::shared_ptr<Node> linear_3 = (*m_lin3)(relu_2);
  return linear_3;
}
