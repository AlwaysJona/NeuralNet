#ifndef LINEAR_H
#define LINEAR_H

#include "Module.h"
#include "Tensor.h"

// The Linear class implements a linear activation functions, such that
// 
// output = input*W + B
//
// where W is the Tensor of weigths, and B is a Bias Tensor
// W and B are updated at each gradient descent step
//
// Implementation in Linear.cpp

class Linear : public Module {
 private:
  Tensor m_weight;
  Tensor m_bias;
  std::size_t m_in_feature;   // rows
  std::size_t m_out_feature;  // cols
  std::size_t m_seed;

 public:
  Linear(std::size_t in_feat, std::size_t out_feat, std::size_t seed = 7);
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

  void reset_parameters();
};

#endif
