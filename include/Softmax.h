#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <memory>

#include "Module.h"
#include "Tensor.h"

// The Softmax class implements the Softmax activation function,
// used in the final layer of the neural network to convert the 
// output of the previous layer into descrete probabilites. 
// It is used in classification problems such as the MNIST one,
// where the aim is to choose one of many descrete classes
//
// If i is the class index, and z_i is the corresponding output from
// the previous layer, then the probability for class i is
// 
// s_i = exp(z_i)/sum_l(exp(z_l))
// 
// where sum_l is the sum over all the indices.
// It can be proven that
//
// ds_i/dz_j = s_i*(&_ij - s_j)
// 
// where i used &_ij for the Kroneker delta
//
// Implementation in Softmax.cpp

class Softmax : public Module {
 public:
  std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;
};

#endif
