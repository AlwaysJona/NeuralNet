#ifndef RELU_H
#define RELU_H

#include "Module.h"
#include "Tensor.h"

// The Relu class implements the ReLU(rectified linear unit) activation
// function, which returns 0 for negative values, and keeps non negative
// values unchanged. 
//
// Implementation in Relu.cpp

class Relu : public Module {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
};

#endif
