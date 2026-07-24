#include <iostream>
#include "Tensor.h"

float compute_loss(std::vector<float> a,
                   std::vector<float> b,
                   std::vector<std::size_t> shape) {
    
    Tensor A(a, shape, false);
    Tensor B(b, shape, false);
    
    return (A.matmul(B) * B.matmul(A)).sum().item();
}

int main(){
        std::vector<float> v1{1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> v2{5.0f, 6.0f, 7.0f, 8.0f};

        std::vector<std::size_t> shape{2,2};

        Tensor A(v1, shape, true);
        Tensor B(v2, shape, true);

        const float delta = 1e-5f;
        const std::size_t i = 0;

        std::vector<float> a_plus = v1;
        std::vector<float> a_minus = v1;

        a_plus[i] += delta;
        a_minus[i] -= delta;

        const float loss_plus = compute_loss(a_plus, v2, shape);
        const float loss_minus = compute_loss(a_minus, v2, shape);

        Tensor loss = (A.matmul(B) * B.matmul(A)).sum();
        loss.backward();

        // dLoss/dA[i]
        const float analytical_grad = A.grad()[i];
        const float numerical_grad = (loss_plus - loss_minus) / (2*delta);

        const float abs_diff = std::abs(analytical_grad - numerical_grad);
        const float rel_ana = abs_diff / analytical_grad;
        const float rel_num = abs_diff / numerical_grad;

        std::cout << "Delta: " << delta << "\n";
        std::cout << "Analytical grad: " << analytical_grad << "\n";
        std::cout << "Numerical grad: " << numerical_grad << "\n";
        std::cout << "Absolute diff: " << abs_diff << "\n";
        std::cout << "Diff relative to Analytical: " << rel_ana << "\n"; 
        std::cout << "Diff relative to Numerical: " << rel_num << "\n"; 

}

