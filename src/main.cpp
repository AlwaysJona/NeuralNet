#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Flatten.h"
#include "Linear.h"
#include "Module.h"
#include "Relu.h"
#include "Serialization.h"
#include "Tensor.h"

class NeuralNetwork : public Module {
 private:
  // layers
  std::shared_ptr<Flatten> m_flatten = std::make_shared<Flatten>();
  std::shared_ptr<Linear> m_lin1 = std::make_shared<Linear>(28 * 28, 512);
  std::shared_ptr<Linear> m_lin2 = std::make_shared<Linear>(512, 512);
  std::shared_ptr<Linear> m_lin3 = std::make_shared<Linear>(512, 10);
  // activation
  std::shared_ptr<Relu> m_relu = std::make_shared<Relu>();

 public:
  NeuralNetwork() {
    register_module("linear_1", m_lin1);
    register_module("linear_2", m_lin2);
    register_module("linear_3", m_lin3);
    std::cout << "Constructor of NeuralNetwork called \n";
  }
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) {
    std::cout << "Entered forward() function \n";
    Tensor t_input(std::move(input));
    std::cout << "Before flat \n";
    std::shared_ptr<Node> flat = (*m_flatten)(t_input.node());
    std::cout << "Before linear 1 \n";
    std::shared_ptr<Node> linear_1 = (*m_lin1)(flat);
    std::cout << "Before relu 1 \n";
    std::shared_ptr<Node> relu_1 = (*m_relu)(linear_1);
    std::cout << "Before linear 2 \n";
    std::shared_ptr<Node> linear_2 = (*m_lin2)(relu_1);
    std::cout << "Before relu 2 \n";
    std::shared_ptr<Node> relu_2 = (*m_relu)(linear_2);
    std::cout << "Before linear 3 \n";
    std::shared_ptr<Node> linear_3 = (*m_lin3)(relu_2);
    std::cout << "After linear 3 \n";
    return linear_3;
  }
};

int main() {
  NeuralNetwork model;

  // random input
  std::vector<std::vector<float>> input_data(28, std::vector<float>(28));
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& row : input_data) {
    for (auto& val : row) {
      val = dist(rng);
    }
  }
  // create input tensor
  Tensor input_t(input_data);

  std::cout << "Tensor created \n";

  // forward pass
  std::shared_ptr<Node> output = model(input_t.node());

  std::cout << "Forward pass called, output created \n";

  Tensor output_t(output);

  std::cout << "Output tensor: " << output_t << std::endl;

  return 0;
}
