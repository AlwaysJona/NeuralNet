#ifndef SGD_H
#define SGD_H

#include <memory>
#include <string>
#include <vector>

#include "Module.h"
#include "Tensor.h"

// Stochastic Gradient Descent: w_i+1 = w_i - LR * dL/dw_i
// where LR = Learning Rate, and L = Loss
// After backpropagating the gradient up the tree, the 
// Neural Network takes a step in the opposite direction as the
// gradient, with the aim of moving towards a minimum of the 
// Loss function
//
// Implementation in SGD.cpp

class SGD {
 private:
  list<Node> m_params;
  float m_learning_rate;

 public:
  SGD(list<Node> par, const float& lr = 0.001f);
  void step();
  void zero_grad();
};

#endif
