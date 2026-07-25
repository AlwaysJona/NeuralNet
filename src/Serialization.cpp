#include "Serialization.h"
#include "Tensor.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

const int MAGIC_NUMBER = 777;

void save(const std::unordered_map<std::string, std::shared_ptr<Node>>& state_dict,
          const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char *> (&MAGIC_NUMBER), sizeof(int));
    for (const auto &[weight_name, weight] : state_dict){

        std::size_t name_len = weight_name.size();
        file.write(reinterpret_cast<const char *> (&name_len), sizeof(std::size_t));
        file.write(weight_name.data(), name_len);
    
        Tensor t_weight(weight);
        std::size_t shape_len = t_weight->shape().size();
        file.write(reinterpret_cast<const char *>(&shape_len), sizeof(std::size_t));
        file.write(reinterpret_cast<const char *>(t_weight.shape().data()),
                                                  shape_len * sizeof(size_t));
    
        std::size_t data_len = t_weight.size();
        file.write(reinterpret_cast<const char *>(&data_len), sizeof(std::size_t));
        file.write(reinterpret_cast<const char *>(t_weight.data().data()),
                                                  data_len * sizeof(float));
    }
}

std::unordered_map<std::string, std::shared_ptr<Node>> load(const std::string& filename) {
    std::unordered_map<std::string, std::shared_ptr<Node>> state_dict;
    std::ifstream file(filename, std::ios::binary);
    if(!file){
        throw std::runtime_error("Could not open file " + filename);
    }

    int magic = 0;
    file.read(reinterpret_cast<char *>(&magic), sizeof(int));
    if (magic != MAGIC_NUMBER)
    {
        throw std::runtime_error("Bad fle format: wrong magic number");
    }

    while(file.peek() != EOF){
        size_t name_len = 0;
        if (!file.read(reinterpret_cast<char *>(&name_len), sizeof(size_t))){
            break;
        }

        std::string weight_name(name_len, '\0');
        file.read(weight_name.data(), name_len);

        std::size_t shape_length = 0;
        file.read(reinterpret_cast<char *>(&shape_length), sizeof(std::size_t));

        std::vector<std::size_t> shape(shape_length);
        file.read(reinterpret_cast<char *>(shape.data()), shape_length * sizeof(std::size_t));

        std::size_t data_length = 0;
        file.read(reinterpret_cast<char *>(&data_length), sizeof(std::size_t));

        std::vector<float> raw(data_length);
        file.read(reinterpret_cast<char *>(raw.data()), data_length * sizeof(float));

        Tensor tensor;
        if(shape_length == 0){
            tensor = Tensor(raw[0]);
        }
        else if(shape_length == 1 || shape_length == 2){
            tensor = Tensor(raw, shape);
        }
        else
        {
            throw std::runtime_error("Unsupported tensor dimensionality: " +
                                     std::to_string(shape_length));
        }

        state_dict[weight_name] = tensor.node();
    }
    return state_dict;
}
