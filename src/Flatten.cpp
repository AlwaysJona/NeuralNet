#include "Flatten.h"
#include <functional>
#include <memory>

std::shared_ptr<Node> Flatten::forward(std::shared_ptr<Node> input){
    Tensor t_input(std::move(input));
    std::vector<float> result;
    result = t_input.data();
    if(t_input.required_grad()){
        std::vector<std::shared_ptr<Node>> parents{t_input.node()};
        std::function<void()> gradfn = 
            [t_input](const std::vector<float>& grad_output){
                std::vector<float> grad_input;
                t_input.add_to_grad(grad_output);
                };
            }
        Tensor t_output(result, true, gradfn, parents);
        return t_output.node();
    }
    Tensor output(result);
    return output.node();

}
