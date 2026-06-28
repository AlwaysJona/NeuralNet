#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>

class Tensor {
private:
        std::vector<float> m_data;
        std::vector<std::size_t> m_shape; // empty if scalar, 1 element if 1D, 2 elements if 2D (rows and cols)
        std::size_t m_stride; // stride = 1 for 1D tensor, for 2D Tensor stride = num of cols
public:
        // constructors
        Tensor(const float data); // scalar
        Tensor(const std::vector<float> &data); // 1D tensor
        Tensor(const std::vector<std::vector<float>> &data); // 2D tensor
        Tensor(const std::vector<float> &data, const std::vector<std::size_t> &shape); 
        // flat Tensor with specified shape                                                     
        // returns the only item for scalars
        const float &item() const; // read only
        float &item(); // write

        // operator() to access elements: data(0) returns first element of data
        const float &operator()(const std::size_t i) const; // 1D read only
        float &operator()(const std::size_t i); // 1D write
        const float &operator()(const std::size_t i, const std::size_t j) const; // 2D read only
        float &operator()(const std::size_t i, const std::size_t j); // 2D write
                                                                     
        const std::size_t size() const{ return m_data.size(); } // total number of elements in the tensor
        const std::vector<std::size_t> &shape() const{ return m_shape; }
        const std::size_t &stride() const{ return m_stride; }
        const std::vector<float> &data() const{ return m_data; }
 
        Tensor operator+(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const; // element wise multiplication
        
        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};

#endif
