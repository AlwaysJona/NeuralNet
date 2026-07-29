#ifndef NEURAL_H
#define NEURAL_H

#include "Flatten.h"
#include "Linear.h"
#include "Module.h"
#include "Relu.h"
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
  NeuralNetwork();

  std::shared_ptr<Node> forward(std::shared_ptr<Node> input);
};

#endif
