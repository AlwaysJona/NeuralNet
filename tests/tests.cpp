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

TEST_CASE("Matrix Multiplication 1D x 1D and 2D x 2D"){
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
    // TODO: errors due to dimension mismatch
}

TEST_CASE("Matmul 2D x 1D and 1D x 2D") {
    
    std::vector<float> v1{5.0f, 10.0f};
    Tensor t1(v1);
    
    std::vector<float> v2{1.0f, 5.0f, 10.0f};
    Tensor t2(v2);

    std::vector<float> v3 = {1,2,3,4,5,6}; // 1 2
                                           // 3 4
                                           // 5 6 
    std::vector<std::size_t> shape1 = {3, 2};

    std::vector<float> v4 = {1,2,3,4,5,6}; // 1 2 3
                                           // 4 5 6
    std::vector<std::size_t> shape2 = {2,3};

    Tensor t3 = Tensor(v3, shape1);
    Tensor t4 = Tensor(v4, shape2);

    // 2D x 1D
    Tensor t31 = t3.matmul(t1);
    Tensor t42 = t4.matmul(t2);

    // 1D x 2D
    Tensor t14 = t1.matmul(t4);
    Tensor t23 = t2.matmul(t3);
    
    // t3 x t1
    CHECK_EQ(t31(0), doctest::Approx(25.0f));
    CHECK_EQ(t31(1), doctest::Approx(55.0f));
    CHECK_EQ(t31(2), doctest::Approx(85.0f));

    // t4 x t2
    CHECK_EQ(t42(0), doctest::Approx(41.0f));
    CHECK_EQ(t42(1), doctest::Approx(89.0f));

    // t1 x t4
    CHECK_EQ(t14(0), doctest::Approx(45.0f));
    CHECK_EQ(t14(1), doctest::Approx(60.0f));
    CHECK_EQ(t14(2), doctest::Approx(75.0f));

    // t2 x t3
    CHECK_EQ(t23(0), doctest::Approx(66.0f));
    CHECK_EQ(t23(1), doctest::Approx(82.0f));
    // TODO:: errors due to dimension mismatch

}

TEST_CASE("Scalar autograd multiplication and addition") {
    Tensor a(2.0f, true);
    Tensor b(3.0f, true);

    Tensor c = a * b + a;

    c.backward();

    CHECK_EQ(c.item(), doctest::Approx(8.0f));

    CHECK_EQ(a.grad()[0], doctest::Approx(4.0f));
    CHECK_EQ(b.grad()[0], doctest::Approx(2.0f));
}

TEST_CASE("Scalar broadcasting autograd") {
    Tensor a(2.0f, true);
    Tensor x(std::vector<float>{1.0f, 2.0f, 3.0f}, true);

    Tensor y = a * x;
    Tensor loss = y.sum();

    loss.backward();

    CHECK_EQ(loss.item(), doctest::Approx(12.0f));

    CHECK_EQ(a.grad()[0], doctest::Approx(6.0f));

    CHECK_EQ(x.grad()[0], doctest::Approx(2.0f));
    CHECK_EQ(x.grad()[1], doctest::Approx(2.0f));
    CHECK_EQ(x.grad()[2], doctest::Approx(2.0f));
}

TEST_CASE("1D Tensor autograd square sum") {
    Tensor x(std::vector<float>{1.0f, 2.0f, 3.0f}, true);

    Tensor y = x * x;

    Tensor loss = y.sum();

    loss.backward();

    CHECK_EQ(loss.item(), doctest::Approx(14.0f));

    CHECK_EQ(x.grad()[0], doctest::Approx(2.0f));
    CHECK_EQ(x.grad()[1], doctest::Approx(4.0f));
    CHECK_EQ(x.grad()[2], doctest::Approx(6.0f));
}

TEST_CASE("2D Tensor autograd square sum") {
    Tensor A(std::vector<std::vector<float>>{
            {1.0f, 2.0f},
            {3.0f, 4.0f}}, 
            true);    
    
    Tensor B = A * A;

    Tensor loss = B.sum();
    
    loss.backward();
    
    CHECK_EQ(loss.item(), doctest::Approx(30.0f));

    CHECK_EQ(A.grad()[0], doctest::Approx(2.0f));
    CHECK_EQ(A.grad()[1], doctest::Approx(4.0f));
    CHECK_EQ(A.grad()[2], doctest::Approx(6.0f));
    CHECK_EQ(A.grad()[3], doctest::Approx(8.0f));
}

TEST_CASE("1D backward with explicit seed gradient") {
    Tensor x(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    Tensor y = x * x;

    y.backward(std::vector<float>{1.0f, 1.0f, 1.0f});

    CHECK_EQ(x.grad()[0], doctest::Approx(2.0f));
    CHECK_EQ(x.grad()[1], doctest::Approx(4.0f));
    CHECK_EQ(x.grad()[2], doctest::Approx(6.0f));
}

TEST_CASE("1D backward with weighted seed gradient") {
    Tensor x(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    Tensor y = x * x;

    y.backward(std::vector<float>{10.0f, 20.0f, 30.0f});

    CHECK_EQ(x.grad()[0], doctest::Approx(20.0f));
    CHECK_EQ(x.grad()[1], doctest::Approx(80.0f));
    CHECK_EQ(x.grad()[2], doctest::Approx(180.0f));
}

TEST_CASE("matmul autograd 1D x 1D") {
    Tensor a(std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    Tensor b(std::vector<float>{4.0f, 5.0f, 6.0f}, true);

    Tensor loss = a.matmul(b);

    loss.backward();

    CHECK_EQ(loss.item(), doctest::Approx(32.0f));

    CHECK_EQ(a.grad()[0], doctest::Approx(4.0f));
    CHECK_EQ(a.grad()[1], doctest::Approx(5.0f));
    CHECK_EQ(a.grad()[2], doctest::Approx(6.0f));

    CHECK_EQ(b.grad()[0], doctest::Approx(1.0f));
    CHECK_EQ(b.grad()[1], doctest::Approx(2.0f));
    CHECK_EQ(b.grad()[2], doctest::Approx(3.0f));
}

TEST_CASE("matmul autograd 1D x 2D") {
    Tensor x(std::vector<float>{1.0f, 2.0f}, true);

    Tensor W(
        std::vector<float>{
            3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f
        },
        std::vector<std::size_t>{2, 3},
        true
    );

    Tensor y = x.matmul(W);
    Tensor loss = y.sum();

    loss.backward();

    CHECK_EQ(y(0), doctest::Approx(15.0f));
    CHECK_EQ(y(1), doctest::Approx(18.0f));
    CHECK_EQ(y(2), doctest::Approx(21.0f));

    CHECK_EQ(loss.item(), doctest::Approx(54.0f));

    CHECK_EQ(x.grad()[0], doctest::Approx(12.0f));
    CHECK_EQ(x.grad()[1], doctest::Approx(21.0f));

    CHECK_EQ(W.grad()[0], doctest::Approx(1.0f));
    CHECK_EQ(W.grad()[1], doctest::Approx(1.0f));
    CHECK_EQ(W.grad()[2], doctest::Approx(1.0f));
    CHECK_EQ(W.grad()[3], doctest::Approx(2.0f));
    CHECK_EQ(W.grad()[4], doctest::Approx(2.0f));
    CHECK_EQ(W.grad()[5], doctest::Approx(2.0f));
}

TEST_CASE("matmul autograd 2D x 1D") {
    Tensor A(
        std::vector<float>{
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f
        },
        std::vector<std::size_t>{3, 2},
        true
    );

    Tensor x(std::vector<float>{10.0f, 20.0f}, true);

    Tensor y = A.matmul(x);
    Tensor loss = y.sum();

    loss.backward();

    CHECK_EQ(y(0), doctest::Approx(50.0f));
    CHECK_EQ(y(1), doctest::Approx(110.0f));
    CHECK_EQ(y(2), doctest::Approx(170.0f));

    CHECK_EQ(loss.item(), doctest::Approx(330.0f));

    CHECK_EQ(A.grad()[0], doctest::Approx(10.0f));
    CHECK_EQ(A.grad()[1], doctest::Approx(20.0f));
    CHECK_EQ(A.grad()[2], doctest::Approx(10.0f));
    CHECK_EQ(A.grad()[3], doctest::Approx(20.0f));
    CHECK_EQ(A.grad()[4], doctest::Approx(10.0f));
    CHECK_EQ(A.grad()[5], doctest::Approx(20.0f));

    CHECK_EQ(x.grad()[0], doctest::Approx(9.0f));
    CHECK_EQ(x.grad()[1], doctest::Approx(12.0f));
}

TEST_CASE("matmul autograd 2D x 2D") {
    Tensor A(
        std::vector<float>{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        },
        std::vector<std::size_t>{2, 3},
        true
    );

    Tensor B(
        std::vector<float>{
            7.0f, 8.0f,
            9.0f, 10.0f,
            11.0f, 12.0f
        },
        std::vector<std::size_t>{3, 2},
        true
    );

    Tensor C = A.matmul(B);
    Tensor loss = C.sum();

    loss.backward();

    CHECK_EQ(C(0, 0), doctest::Approx(58.0f));
    CHECK_EQ(C(0, 1), doctest::Approx(64.0f));
    CHECK_EQ(C(1, 0), doctest::Approx(139.0f));
    CHECK_EQ(C(1, 1), doctest::Approx(154.0f));

    CHECK_EQ(loss.item(), doctest::Approx(415.0f));

    CHECK_EQ(A.grad()[0], doctest::Approx(15.0f));
    CHECK_EQ(A.grad()[1], doctest::Approx(19.0f));
    CHECK_EQ(A.grad()[2], doctest::Approx(23.0f));
    CHECK_EQ(A.grad()[3], doctest::Approx(15.0f));
    CHECK_EQ(A.grad()[4], doctest::Approx(19.0f));
    CHECK_EQ(A.grad()[5], doctest::Approx(23.0f));

    CHECK_EQ(B.grad()[0], doctest::Approx(5.0f));
    CHECK_EQ(B.grad()[1], doctest::Approx(5.0f));
    CHECK_EQ(B.grad()[2], doctest::Approx(7.0f));
    CHECK_EQ(B.grad()[3], doctest::Approx(7.0f));
    CHECK_EQ(B.grad()[4], doctest::Approx(9.0f));
    CHECK_EQ(B.grad()[5], doctest::Approx(9.0f));
}
