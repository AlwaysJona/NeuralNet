#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "Tensor.h"

TEST_CASE("Testing proper creation of Tensors"){
    // scalar
    Tensor t1 = Tensor(3.14);
    CHECK_EQ(t1.shape(), std::vector<std::size_t>({}));
    CHECK_THROWS_AS(t1(0), std::invalid_argument); // for scalars use item()
    CHECK_EQ(t1.item(), doctest::Approx(3.14));
    CHECK_THROWS_AS(t1(1,2), std::invalid_argument);
    // TODO: check all the exceptions


    // 1D
    std::vector<float> v = {1, 2, 3, 4};
    Tensor t2 = Tensor(v);
    CHECK_EQ(t2.shape(), std::vector<std::size_t>({4}));
    CHECK_THROWS_AS(t2.item(), std::runtime_error); // only works for single element tensors
    CHECK_THROWS_AS(t2(4), std::invalid_argument); // out of bounds
    CHECK_EQ(t2(1), doctest::Approx(2));
    CHECK_THROWS_AS(t2(1,2), std::invalid_argument);
    // TODO: check all the exceptions


    // 2D
    std::vector<std::vector<float>> vv = {v, v, v};
    Tensor t3 = Tensor(vv);
    CHECK_EQ(t3.shape(), std::vector<std::size_t>({3,4}));
    CHECK_THROWS_AS(t3.item(), std::runtime_error);
    CHECK_EQ(t3(1,3), doctest::Approx(4));
    CHECK_THROWS_AS(t3(3,1), std::invalid_argument); // out of bounds
    CHECK_THROWS_AS(t3(1,4), std::invalid_argument);
    CHECK_THROWS_AS(t3(3,4), std::invalid_argument);
    // TODO: check all the exceptions
    

    // flat Tensor
    std::vector<std::size_t> shape({2,4});
    std::vector<float> v1 = v;
    v1.insert(v1.end(), v.begin(), v.end()); // 1D vector with 8 elements {1,2,3,4,1,2,3,4}
    Tensor t4 = Tensor(v1, shape);
    CHECK_EQ(t4(0,0), doctest::Approx(1));
    CHECK_EQ(t4(0,1), doctest::Approx(2));
    CHECK_EQ(t4(0,2), doctest::Approx(3));
    CHECK_EQ(t4(0,3), doctest::Approx(4));
    CHECK_EQ(t4(1,0), doctest::Approx(1));
    CHECK_EQ(t4(1,1), doctest::Approx(2));
    CHECK_EQ(t4(1,2), doctest::Approx(3));
    CHECK_EQ(t4(1,3), doctest::Approx(4));
    
}

TEST_CASE("Tensor Addition"){
    Tensor t1 = Tensor(2);
    Tensor t2 = Tensor(5);
    Tensor t3 = t1 + t2;
    CHECK_EQ(t3.item(), 7);
    
    std::vector<float> v = {3, 4, 5, 6};

    Tensor t4 = Tensor(v);
    Tensor t5 = t1 + t4;
    CHECK_EQ(t5(0), doctest::Approx(5));
    CHECK_EQ(t5(1), doctest::Approx(6));
    CHECK_EQ(t5(2), doctest::Approx(7));
    CHECK_EQ(t5(3), doctest::Approx(8));

    Tensor t6 = t4 + t1;
    CHECK_EQ(t6(0), doctest::Approx(5));
    CHECK_EQ(t6(1), doctest::Approx(6));
    CHECK_EQ(t6(2), doctest::Approx(7));
    CHECK_EQ(t6(3), doctest::Approx(8));
    // TODO: Check all the cases
    
}

TEST_CASE("Element wise multiplication"){
    Tensor t1 = Tensor(2);
    Tensor t2 = Tensor(5);
    Tensor t3 = t1 * t2;
    CHECK_EQ(t3.item(), 10);
    
    std::vector<float> v = {3, 4, 5, 6};

    Tensor t4 = Tensor(v);
    Tensor t5 = t1 * t4;
    CHECK_EQ(t5(0), doctest::Approx(6));
    CHECK_EQ(t5(1), doctest::Approx(8));
    CHECK_EQ(t5(2), doctest::Approx(10));
    CHECK_EQ(t5(3), doctest::Approx(12));

    Tensor t6 = t4 * t1;
    CHECK_EQ(t6(0), doctest::Approx(6));
    CHECK_EQ(t6(1), doctest::Approx(8));
    CHECK_EQ(t6(2), doctest::Approx(10));
    CHECK_EQ(t6(3), doctest::Approx(12));
    // TODO: Check all the cases
}

TEST_CASE("Matrix Multiplication"){
    std::vector<float> v1 = {1, 2, 3, 4, 5};
    std::vector<float> v2 = {2, 4, 6, 8, 10};

    Tensor t1 = Tensor(v1);
    Tensor t2 = Tensor(v2);

    Tensor t12 = t1.matmul(t2);

    CHECK_EQ(t12.item(), doctest::Approx(110));

    std::vector<float> v3 = {1,2,3,4,5,6}; // 1 2
                                           // 3 4
                                           // 5 6 
    std::vector<std::size_t> shape1 = {3, 2};

    std::vector<float> v4 = {1,2,3,4,5,6}; // 1 2 3
                                           // 4 5 6
    std::vector<std::size_t> shape2 = {2,3};

    Tensor t3 = Tensor(v3, shape1);
    Tensor t4 = Tensor(v4, shape2);
    Tensor t34 = t3.matmul(t4);

    CHECK_EQ(t34(0,0), doctest::Approx(9));
    CHECK_EQ(t34(0,1), doctest::Approx(12));
    CHECK_EQ(t34(0,2), doctest::Approx(15));
    CHECK_EQ(t34(1,0), doctest::Approx(19));
    CHECK_EQ(t34(1,1), doctest::Approx(26));
    CHECK_EQ(t34(1,2), doctest::Approx(33));
    CHECK_EQ(t34(2,0), doctest::Approx(29));
    CHECK_EQ(t34(2,1), doctest::Approx(40));
    CHECK_EQ(t34(2,2), doctest::Approx(51));
    // TODO: Write at least the bare minimum cases
}
