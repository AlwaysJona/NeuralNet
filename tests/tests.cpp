#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "Tensor.h"

TEST_CASE("Testing proper creation of Tensors"){
    // scalar
    Tensor t1 = Tensor(3.14);
    CHECK_EQ(t1.shape(), std::vector<int>({}));
    CHECK_THROWS_AS(t1(0), std::invalid_argument); // for scalars use item()
    CHECK_EQ(t1.item(), doctest::Approx(3.14));
    CHECK_THROWS_AS(t1(1,2), std::invalid_argument);
    // TODO: check all the exceptions


    // 1D
    std::vector<float> v = {1, 2, 3, 4};
    Tensor t2 = Tensor(v);
    CHECK_EQ(t2.shape(), std::vector<int>({4}));
    CHECK_THROWS_AS(t2.item(), std::runtime_error); // only works for single element tensors
    CHECK_THROWS_AS(t2(4), std::invalid_argument); // out of bounds
    CHECK_EQ(t2(1), doctest::Approx(2));
    CHECK_THROWS_AS(t2(1,2), std::invalid_argument);
    // TODO: check all the exceptions


    // 2D
    std::vector<std::vector<float>> vv = {v, v, v};
    Tensor t3 = Tensor(vv);
    CHECK_EQ(t3.shape(), std::vector<int>({3,4}));
    CHECK_THROWS_AS(t3.item(), std::runtime_error);
    CHECK_EQ(t3(1,3), doctest::Approx(4));
    CHECK_THROWS_AS(t3(3,1), std::invalid_argument); // out of bounds
    CHECK_THROWS_AS(t3(1,4), std::invalid_argument);
    CHECK_THROWS_AS(t3(3,4), std::invalid_argument);
    // TODO: check all the exceptions
}
