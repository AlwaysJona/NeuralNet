#ifndef SERIAL_H
#define SERIAL_H

#include <string>
#include <unordered_map>

#include "Tensor.h"

using Node = Tensor::Node;

void save(
    const std::unordered_map<std::string, std::shared_ptr<Node>>& state_dict,
    const std::string& filename);

std::unordered_map<std::string, std::shared_ptr<Node>> load(
    const std::string& filename);

#endif
