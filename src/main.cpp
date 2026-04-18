#include <iostream>
#include "Tensor.h"


int main(){
        std::vector<float> v({1.89, 1.56, 3, 24});
        std::vector<float> v1({1,2,3,5});
        std::vector<std::vector<float>> v2({v,v1});
        Tensor t(3.4);
        Tensor t1(v);
        Tensor t2(v2);
        std::cout << t << std::endl; 
        std::cout << t1 << std::endl;
        std::cout << t2 << std::endl;
}

