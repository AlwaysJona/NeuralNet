#ifndef NEURAL_H
#define NEURAL_H

#include "Dataloader.h"
#include "Dataset.h"
#include "Flatten.h"
#include "Linear.h"
#include "Loss.h"
#include "Module.h"
#include "Neural.h"
#include "Relu.h"
#include "SGD.h"
#include "Serialization.h"
#include "Softmax.h"
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

void train(Dataloader& dataloader, NeuralNetwork& model,
           Cross_Entropy_Loss& loss_fn, SGD& optimizer);

void test(Dataloader& dataloader, NeuralNetwork& model,
          Cross_Entropy_Loss& loss_fn);

void train_new_mnist_model(const bool impose_limit,
                           const int size_limit = 60000);

void inference_on_saved_model(const bool impose_limit,
                              const int size_limit = 60000,
                              const bool use_pre_trained = false);
#endif
