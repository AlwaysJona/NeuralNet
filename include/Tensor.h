#ifndef TENSOR
#define TENSOR

#include <vector>
#include <iostream>

class Tensor {
private:
        std::vector<float> m_data;
        std::vector<int> m_shape;
        int m_stride;
public:
        Tensor(const float data); // scalar
        Tensor(const std::vector<float> &data); // 1D tensor
        Tensor(const std::vector<std::vector<float>> &data); // 2D tensor
        const float &item() const; // read only
        float &item(); // write
        const float &operator()(const std::size_t i) const; // 1D read only
        float &operator()(const std::size_t i); // 1D write
        const float &operator()(const std::size_t i, const std::size_t j) const; // 2D read only
        float &operator()(const std::size_t i, const std::size_t j); // 2D write
        const std::size_t size() const{ return m_data.size(); }
        const std::vector<int> &shape() const{ return m_shape;}
        const int &stride() const{ return m_stride;}
        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};

#endif
