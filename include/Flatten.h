#ifndef FLATTEN_H
#define FLATTEN_H

#include "Tensor.h"
#include "Module.h"
#include <memory>

// The Flatten module is used to linearize the input, in whatever format it is given
class Flatten : public Module {
public:
    std::shared_ptr<Node> forward(std::shared_ptr<Node> input) override;

};

#endif
