#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <iostream>
#include <string>
#include <memory>
#include <functional>

class Tensor {
private:
        struct Node {
            std::vector<float> m_data;
            std::vector<std::size_t> m_shape; // empty if scalar, 1 element if 1D, 2 elements if 2D (rows and cols)
            std::size_t m_stride; // stride = 1 for 1D tensor, for 2D Tensor stride = num of cols
            std::vector<float> m_grad; // same size as m_data
                                       
            bool m_requires_grad;

            // updates gradient of parent based on gradient of child
            std::function<void(const std::vector<float>&)> m_gradfn; 

            std::vector<std::shared_ptr<Tensor>> m_parents;
            
            // constructor
            Node(std::vector<float> data, std::vector<std::size_t> shape, bool requires_grad);
        };
        
        std::shared_ptr<Node> m_node;
public:
        // constructors
        explicit Tensor(const float data, bool requires_grad = false,
                std::function<void(const std::vector<float>&)> gradfn = nullptr,
                std::vector<std::shared_ptr<Tensor>> parents = {}); // scalar
        
        explicit Tensor(const std::vector<float> &data, 
                bool requires_grad = false,
                std::function<void(const std::vector<float>&)> gradfn = nullptr,
                std::vector<std::shared_ptr<Tensor>> parents = {}); // 1D tensor
                                              
        explicit Tensor(const std::vector<std::vector<float>> &data, 
                bool requires_grad = false,
                std::function<void(const std::vector<float>&)> gradfn = nullptr,
                std::vector<std::shared_ptr<Tensor>> parents = {}); // 2D tensor
        
        // flat Tensor with specified shape                                                     
        Tensor(const std::vector<float> &data, 
                const std::vector<std::size_t> &shape,
                bool requires_grad = false,
                std::function<void(const std::vector<float>&)> gradfn = nullptr,
                std::vector<std::shared_ptr<Tensor>> parents = {}); 
        
        // returns the only item for scalars
        const float &item() const; // read only
        float &item(); // write

        // operator() to access elements: data(0) returns first element of data
        const float &operator()(const std::size_t i) const; // 1D read only
        float &operator()(const std::size_t i); // 1D write
        const float &operator()(const std::size_t i, const std::size_t j) const; // 2D read only
        float &operator()(const std::size_t i, const std::size_t j); // 2D write
                                                                     
        const std::size_t size() const{ return m_node->m_data.size(); } // total number of elements in the tensor
        const std::vector<std::size_t> &shape() const{ return m_node->m_shape; }
        const std::size_t &stride() const{ return m_node->m_stride; }
        const std::vector<float> &data() const{ return m_node->m_data; }
 
        Tensor operator+(const Tensor& other) const;
        Tensor operator*(const Tensor& other) const; // element wise multiplication
        Tensor matmul(const Tensor& other) const; // matrix multiplication

        friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};

#endif
