#ifndef MODULE_H
#define MODULE_H

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Tensor.h"

// Instead of defining activation functions, optimizers and
// other operations each time, it is convenient to have Module
// objects that already implement them. 
// Module is just the base class, defining the operations that
// each derived class should have and implement
//
// Implementation in Module.cpp

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
  // A virtual method is one that must be defined by a derived class,
  // otherwise instantiation is not allowed
  virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> input) = 0;
  std::shared_ptr<Node> operator()(std::shared_ptr<Node> input) {
    return forward(input);
  }

  // setters
  void register_parameter(std::string name, std::shared_ptr<Node> param);
  void register_module(std::string name, std::shared_ptr<Module> mod);
  
  // getters
  list<Node> parameters() const;
  dict<Node> state_dict() const;
  
  // updated member variable m_parameters with elements in state_dict
  void load_state_dict(dict<Node>& state_dict);
};

#endif
