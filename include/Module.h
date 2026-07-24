#ifndef MODULE_H
#define MODULE_H

#include "NeuralNet/include/Tensor.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename T>
// basically a python dictionary
using dict = std::vector<std::pair<std::string, std::shared_ptr<T>>>;

class Module
{

private:
    dict<Node> m_parameters;
    dict<Module> m_modules;

public: 
    virtual ~Module() = default;

    virtual std::shared_ptr<Node> forward(std::shared_ptr<Node> input) = 0;
    std::shared_ptr<Node> operator()(std::shared_ptr<Node> input) { return forward(input); }
    
    void register_parameter(std::string name, std::shared_ptr<Node> param);
    void register_module(std::string name, std::shared_ptr<Module> mod);
    
    dict<Node> parameters() const;

    
};


#endif
