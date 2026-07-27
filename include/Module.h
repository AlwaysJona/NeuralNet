#ifndef MODULE_H
#define MODULE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Tensor.h"

template <typename T>
// basically a python list
using list = std::vector<std::pair<std::string, std::shared_ptr<T>>>;

template <typename T>
// basically a python dictionary
using dict = std::unordered_map<std::string, std::shared_ptr<T>>;

using Node = Tensor::Node;

class Module {
 private:
  list<Node> m_parameters;
  list<Module> m_modules;

 public:
  virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> input) = 0;
  std::shared_ptr<Node> operator()(std::shared_ptr<Node> input) {
    return forward(input);
  }

  void register_parameter(std::string name, std::shared_ptr<Node> param);
  void register_module(std::string name, std::shared_ptr<Module> mod);

  list<Node> parameters() const;
  dict<Node> state_dict() const;
  void load_state_dict(dict<Node>& state_dict);
};

#endif
