#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <memory>

#include "Module.h"
#include "Tensor.h"

class Softmax : public Module {
   public:
    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
};

#endif
