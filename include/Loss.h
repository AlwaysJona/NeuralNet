#ifndef LOSS_H
#define LOSS_H

#include <memory>

#include "Module.h"
#include "Tensor.h"

// Losses are needed to evaluate the performance of a model,
// in this classification task I choose to measure the loss
// with the log likelyhood:
//
// loss = -log(guess)
//
// where guess is the output Tensor resulting from the forward pass through
// all the layers of the neural network.
// Cross Entropy Loss class calls NLL loss with the probabilities from
// the Softmax layer.
//
// Implementation in Loss.cpp

class Loss : public Module {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
  virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> input,
                                        std::size_t target);
  std::shared_ptr<Node> operator()(std::shared_ptr<Node> input,
                                   std::size_t target);
};

// Negative log likelyhood Loss
class NLL_Loss : public Loss {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input,
                                std::size_t target);
};

// Cross Entropy loss
class Cross_Entropy_Loss : public Loss {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input,
                                std::size_t target);
};
#endif
