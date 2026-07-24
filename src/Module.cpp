#include "Module.h"

void Module::register_parameter(std::string name, std::shared_ptr<Node> param) {
    for(const auto& p : m_parameters){
        if(p.first == name){
            throw std::runtime_error("Parameter " + name + " already registered");
        }
    }
    m_parameters.push_back({name, param});
}


void Module::register_module(std::string name, std::shared_ptr<Module> mod) {
    for(const auto& m : m_modules){
        if(m.first == name){
            throw std::runtime_error("Module " + name + " already registered");
        }
    }
    m_parameters.push_back({name, mod});
}

dict<Node> Module::parameters() const {
    dict<Node> params;
    for(const auto& p : m_parameters){
        params.push_back(p);
    }
    for(const auto& m : m_modules){
        for(const auto& p: m.second->parameters()){
            std:.string full_name = m.first.empty() ? p.first : m.first + "." p.first;
            params.push_back({full_name, p.second});
        }
    }
    return params;

}
