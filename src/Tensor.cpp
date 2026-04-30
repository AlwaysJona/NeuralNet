#include "Tensor.h"

Tensor::Tensor(const float data) : m_data{data}, m_shape{}, m_stride{} {};

Tensor::Tensor(const std::vector<float> &data) : m_data{data}, m_shape{data.size()}, m_stride{1} {};

Tensor::Tensor(const std::vector<std::vector<float>> &data) : m_shape{data.size(), data[0].size()}, m_stride{data[0].size()} {
    
// Check if all the rows of the tensor have the same size        
    for(std::size_t i = 0; i < data.size(); ++i){
        if(data[i].size() != m_stride) {
            throw std::invalid_argument("Dimensions are inconsistent.");
        }
// Fill up the tensor data with the input data in a row major order
        for(std::size_t j = 0; j < data[i].size(); ++j) {
            m_data.push_back(data[i][j]);
        }

    }

}

const float &Tensor::item() const {
    
    // works only with scalars and 1D tensors
    if(m_data.size() == 1){
        return m_data[0];
    }
    else {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }

}

float &Tensor::item() {
    
    // works only with scalars and 1D tensors
    if(m_data.size() == 1){
        return m_data[0];
    }
    else {
        throw std::runtime_error("item() can only be called on tensors with a single element");
    }
}

const float &Tensor::operator()(const std::size_t i) const {
    if(m_shape.size() == 0){
        throw std::invalid_argument("Can't index into a scalar, use item() instead");
    }
    if(m_shape.size() == 1){
        if( i >= m_shape[0]){
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(m_shape[0]));
        }
        return m_data[i];
    }
    throw std::invalid_argument("This is a 1D tensor");
}

float &Tensor::operator()(const std::size_t i){
    if(m_shape.size() == 0){
        throw std::invalid_argument("Can't index into a scalar, use item() instead");
    }
    if(m_shape.size() == 1){
        if( i >= m_shape[0]){
            throw std::invalid_argument("Index " + std::to_string(i) + " is out of bounds for array of size " + std::to_string(m_shape[0]));
        }
        return m_data[i];
    }
    throw std::invalid_argument("This is a 1D tensor");
}


const float &Tensor::operator()(const std::size_t i, const std::size_t j) const {
    // the operator should only be called on 2D tensors
    if(m_shape.size() < 2){
        throw std::invalid_argument("Too many arguments given");
    }
    else{
        if(i >= m_shape[0] || j >= m_shape[1]){
            throw std::invalid_argument("Out of bounds, shape is " + std::to_string(m_shape[0]) + " " + std::to_string(m_shape[1]) + " but given indeces are " + std::to_string(i) + " " + std::to_string(j));
        }
        else{
            return m_data[i*m_stride + j];
        }
    }
}

float &Tensor::operator()(const std::size_t i,const std::size_t j) {
    if(m_shape.size() < 2){
        throw std::invalid_argument("Too many arguments given");
    }
    else{
        if(i >= m_shape[0] || j >= m_shape[1]){
            throw std::invalid_argument("Out of bounds, shape is " + std::to_string(m_shape[0]) + " " + std::to_string(m_shape[1]) + " but given indeces are " + std::to_string(i) + " " + std::to_string(j));
        }
        else{
            return m_data[i*m_stride + j];
        }
    }
}

std::ostream &operator<<(std::ostream &os, const Tensor &obj){
    std::string output = "[";
    auto size = obj.size();
    auto shape = obj.shape();
    auto dim = shape.size();

    if(dim == 0){
        output += std::to_string(obj.item());
    }
    else if(dim == 1){
        for(auto i = 0; i < size; ++i){
            output += std::to_string(obj(i));
            if(i != size - 1){
                output += ", ";
            }
        }
    }
    else{
        for(auto i = 0; i < size; ++i){
            if((i+1) % shape[1] == 1){
                output += "[";
            }
            int row = i / shape[1];
            int col = i % shape[1];
            output += std::to_string(obj(row, col));
            if((i+1) % shape[1] == 0){
                output += "]";
            }
            if( i != size - 1){
             output += ", ";
            }
        }
    }
    output += "]";
    os << output;
    return os;

}

// TODO: sum of Tensors of different dimensions implemented, a bit sus
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other){
    // figure out the dimensions of the operands
    auto dim1 = m_shape.size();
    auto other_shape = other->shape();
    auto dim2 = other_shape.size();
    // scalar + scalar : summing the values 
    if(dim1 == 0 && dim2 == 0){
        float result = item() + other->item();
    }
    // scalar + 1D: adding the scalar to each element of the 1D tensor
    if(dim1 == 0 && dim2 == 1){
        std::vector<float> result;
        for(auto i = 0; i < other_shape[0]; ++i){
            result.push_back(item() + (*other)(i)):
        }
    }
    // scalard + 2D: adding the scalar to each element of the 2D tensor
    if(dim1 == 0 && dim2 == 2){
        auto rows = other_shape[0];
        auto cols = other_shape[1];
        std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
        for(auto i = 0; i < rows; ++i){
            for(auto j = 0; j < cols; ++j){
                result[i][j] = item() + (*other)(i,j);
            }
        }
    }
    // 1D + scalar: adding the scalar to each element of the 1D tensor
    if(dim1 == 1 && dim2 == 0){
        std::vector<float> result;
        for(auto i = 0; i < shape()[0]; ++i){
            result.push_back(m_data[i] + other->item()):
        }
    }
    // 2D + scalar: adding the scalar to each element of the 2D tensor
    if(dim1 == 2 && dim2 == 0){
        auto rows = m_shape[0];
        auto cols = m_shape[1];
        std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
        for(auto i = 0; i < rows; ++i){
            for(auto j = 0; j < cols; ++j){
                result[i][j] = other->item() + m_data[i*m_stride + j];
            }
        }
    }
    // 1D + 1D: classic vector addition
    if(dim1 == 1 && dim2 == 1){
        if(size() != other->size()){
            throw std::runtime_error("The two 1D tensors have different sizes");
        }
        std::vector<float> result;
        for(auto i = 0; i < size(); ++i){
            result.push_back(m_data[i] + (*other)(i));
        }
    }
    // 2D + 2D: classic vector addition
    if(dim1 == 2 && dim2 == 2){
        if(m_stride != other->shape()[0] || size() != other->size()){
            throw std::runtime_error("The two 2D tensors have different sizes");
        }
        auto rows = m_shape[0];
        auto cols = m_shape[1];
        std::vector<std::vector<float>> result(rows, std::vector<float>(cols));
        for(auto i = 0; i < rows; ++i){
            for(auto j = 0; j < cols; ++j){
                result[i][j] = (*other)(i,j) + m_data[i*m_stride + j];
            }
        }
    }


}
