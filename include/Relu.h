#ifndef RELU_H
#define RELU_H

#include "Module.h"
#include "Tensor.h"

class Relu : public Module {
   public:
    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
};

#endif
