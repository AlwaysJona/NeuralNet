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

list<Node> Module::parameters() const {
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

dict<Node> Module::state_dict() const {
    map<Node> state_dict;
    for(const auto& p : parameters()){
        state_dict[p.first] = p.second;
    }
    return state_dict;
}

void Module::load_state_dict(dict<Node>& state_dict){
    for(const auto& p : parameters()){
        auto it = state_dict.find(p.first);
        if (it == state_dict.end())
        {
            std::cerr << "W: Param " << p.first << " not found in state dict \n";
            continue;
        }
        Tensor stored_param(it->second);
        Tensor p_second(p.second);
        if(p_second.shape() != stored_param.shape()){
            throw std::runtime_error("Param " + p.first + " has different shape in state_dict");
        }
        p_second.set_data(stored_param.data());
    }

}
