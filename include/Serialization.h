#ifndef SERIAL_H
#define SERIAL_H

#include "Tensor.h"
#include <string>
#include <unordered_map>

void save(const std::unordered_map<std::string, std::shared_ptr<Node>>& state_dict,
          const std::string& filename);

std::unordered_map<std::string, std::shared_ptr<Node>> load(const std::string& filename);


#endif
