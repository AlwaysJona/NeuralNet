#ifndef FLATTEN_H
#define FLATTEN_H

#include <memory>

#include "Module.h"
#include "Tensor.h"

// The Flatten module is used to linearize the input, in whatever format it is
// given
class Flatten : public Module {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
};

#endif
