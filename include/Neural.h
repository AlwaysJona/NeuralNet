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

// The NeuralNetwork class implements a simple neural network,
// with a few linear layers and ReLU activation functions.
// It is worth to mention that having linear layers one after another
// is equivalent to only having one. Thus the proposed
// Model is built as such:
// Linear1 -> Relu1 -> Linear2 -> Relu2 -> Linear3
//
// Implementation in Neural.cpp

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

// takes output from last layer, computes loss, propagates it up the tree
// takes step away from gradient, and resets to recompute losses 
void train(Dataloader& dataloader, NeuralNetwork& model,
           Cross_Entropy_Loss& loss_fn, SGD& optimizer);

// takes a pre-trained model, runs test images through the neural network,
// guesses output, measures accuracy and average loss
void test(Dataloader& dataloader, NeuralNetwork& model,
          Cross_Entropy_Loss& loss_fn);

// read dataset from file, trains and tests a model, and finally saves it to file
void train_new_mnist_model(const bool impose_limit,
                           const int size_limit = 60000);
// read pre-trained model from file, runs test images through it, and prints guess
void inference_on_saved_model(const bool impose_limit,
                              const int size_limit = 60000,
                              const bool use_pre_trained = false);
#endif
