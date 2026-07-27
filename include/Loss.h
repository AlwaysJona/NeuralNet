#ifndef LOSS_H
#define LOSS_H

#include <memory>

#include "Module.h"
#include "Tensor.h"

class Loss : public Module {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
  virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> input,
                                        std::size_t target);
  std::shared_ptr<Node> operator()(std::shared_ptr<Node> input,
                                   std::size_t target);
};

// Negative log likelyhood
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
