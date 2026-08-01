#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "Linear.h"
#include "Module.h"
#include "Tensor.h"
#include "Relu.h"
#include "Softmax.h"
#include "Loss.h"

TEST_CASE("Testing proper creation of Tensors") {
  // scalar
  Tensor t1 = Tensor(3.14);
  CHECK_EQ(t1.shape(), std::vector<std::size_t>({}));
  CHECK_EQ(t1.item(), doctest::Approx(3.14));
  CHECK_EQ(t1.size(), std::size_t(1));

  CHECK_THROWS_AS(t1(0), std::invalid_argument);  // for scalars use item()
  CHECK_THROWS_AS(t1(1, 2), std::invalid_argument);

  // 1D
  std::vector<float> v = {1, 2, 3, 4};
  Tensor t2 = Tensor(v);
  CHECK_EQ(t2.shape(), std::vector<std::size_t>({4}));
  CHECK_EQ(t2(1), doctest::Approx(2));
  CHECK_EQ(t2.size(), doctest::Approx(4));

  CHECK_THROWS_AS(t2.item(),
                  std::runtime_error);  // only works for scalar tensors
  CHECK_THROWS_AS(t2(1, 2), std::invalid_argument);
  CHECK_THROWS_AS(t2(4), std::invalid_argument);  // out of bounds

  // 2D
  std::vector<std::vector<float>> vv = {v, v, v};
  Tensor t3 = Tensor(vv);
  CHECK_EQ(t3.shape(), std::vector<std::size_t>({3, 4}));
  CHECK_EQ(t3(1, 3), doctest::Approx(4));
  CHECK_EQ(t3.size(), doctest::Approx(12));

  CHECK_THROWS_AS(t3.item(), std::runtime_error);
  CHECK_THROWS_AS(t3(3, 1), std::invalid_argument);  // out of bounds
  CHECK_THROWS_AS(t3(1, 4), std::invalid_argument);
  CHECK_THROWS_AS(t3(3, 4), std::invalid_argument);

  // flat Tensor
  std::vector<std::size_t> shape({2, 4});
  std::vector<float> v1 = v;
  v1.insert(v1.end(), v.begin(), v.end());  // 1D vector with {1,2,3,4,1,2,3,4}
  Tensor t4 = Tensor(v1, shape);

  CHECK_EQ(t4.size(), 8);
  CHECK_EQ(t4(0, 0), doctest::Approx(1));
  CHECK_EQ(t4(0, 1), doctest::Approx(2));
  CHECK_EQ(t4(0, 2), doctest::Approx(3));
  CHECK_EQ(t4(0, 3), doctest::Approx(4));
  CHECK_EQ(t4(1, 0), doctest::Approx(1));
  CHECK_EQ(t4(1, 1), doctest::Approx(2));
  CHECK_EQ(t4(1, 2), doctest::Approx(3));
  CHECK_EQ(t4(1, 3), doctest::Approx(4));

  CHECK_THROWS_AS(t4(0), std::invalid_argument);  // only for 1D tensors
  CHECK_THROWS_AS(t4.item(), std::runtime_error);
  CHECK_THROWS_AS(t4(2, 0), std::invalid_argument);  // out of bounds
  CHECK_THROWS_AS(t4(2, 1), std::invalid_argument);
  CHECK_THROWS_AS(t4(2, 2), std::invalid_argument);
  CHECK_THROWS_AS(t4(2, 3), std::invalid_argument);
  CHECK_THROWS_AS(t4(2, 4), std::invalid_argument);
}

TEST_CASE("Tensor Addition") {
  // scalar + scalar
  Tensor t1 = Tensor(2);
  Tensor t2 = Tensor(5);
  Tensor t3 = t1 + t2;
  CHECK_EQ(t3.item(), 7);

  std::vector<float> v = {3, 4, 5, 6};

  // scalar + 1D
  Tensor t4 = Tensor(v);
  Tensor t5 = t1 + t4;
  CHECK_EQ(t5(0), doctest::Approx(5));
  CHECK_EQ(t5(1), doctest::Approx(6));
  CHECK_EQ(t5(2), doctest::Approx(7));
  CHECK_EQ(t5(3), doctest::Approx(8));

  // 1D + scalar
  Tensor t6 = t4 + t1;
  CHECK_EQ(t6(0), doctest::Approx(5));
  CHECK_EQ(t6(1), doctest::Approx(6));
  CHECK_EQ(t6(2), doctest::Approx(7));
  CHECK_EQ(t6(3), doctest::Approx(8));

  // 2D + scalar
  Tensor t7(std::vector<std::vector<float>>{v, v});
  Tensor t8 = t7 + t1;
  CHECK_EQ(t8(0, 0), doctest::Approx(5));
  CHECK_EQ(t8(0, 1), doctest::Approx(6));
  CHECK_EQ(t8(0, 2), doctest::Approx(7));
  CHECK_EQ(t8(0, 3), doctest::Approx(8));
  CHECK_EQ(t8(1, 0), doctest::Approx(5));
  CHECK_EQ(t8(1, 1), doctest::Approx(6));
  CHECK_EQ(t8(1, 2), doctest::Approx(7));
  CHECK_EQ(t8(1, 3), doctest::Approx(8));

  CHECK_THROWS_AS(t8(0, 4), std::invalid_argument);
  CHECK_THROWS_AS(t8(4, 0), std::invalid_argument);

  // scalar + 2D
  Tensor t9 = t2 + t7;
  CHECK_EQ(t9(0, 0), doctest::Approx(8));
  CHECK_EQ(t9(0, 1), doctest::Approx(9));
  CHECK_EQ(t9(0, 2), doctest::Approx(10));
  CHECK_EQ(t9(0, 3), doctest::Approx(11));
  CHECK_EQ(t9(1, 0), doctest::Approx(8));
  CHECK_EQ(t9(1, 1), doctest::Approx(9));
  CHECK_EQ(t9(1, 2), doctest::Approx(10));
  CHECK_EQ(t9(1, 3), doctest::Approx(11));

  CHECK_THROWS_AS(t9(0, 4), std::invalid_argument);
  CHECK_THROWS_AS(t9(4, 0), std::invalid_argument);

  // 2D + 2D
  Tensor t10 = t8 + t9;
  CHECK_EQ(t10(0, 0), doctest::Approx(13));
  CHECK_EQ(t10(0, 1), doctest::Approx(15));
  CHECK_EQ(t10(0, 2), doctest::Approx(17));
  CHECK_EQ(t10(0, 3), doctest::Approx(19));
  CHECK_EQ(t10(1, 0), doctest::Approx(13));
  CHECK_EQ(t10(1, 1), doctest::Approx(15));
  CHECK_EQ(t10(1, 2), doctest::Approx(17));
  CHECK_EQ(t10(1, 3), doctest::Approx(19));

  CHECK_THROWS_AS(t10(0, 4), std::invalid_argument);
  CHECK_THROWS_AS(t10(4, 0), std::invalid_argument);

  // 1D + 2D & 2D + 1D
  CHECK_THROWS_AS(t4 + t9, std::invalid_argument);
  CHECK_THROWS_AS(t9 + t4, std::invalid_argument);
}

TEST_CASE("Element wise multiplication") {
  Tensor t1 = Tensor(2);
  Tensor t2 = Tensor(5);
  // scalar * scalar
  Tensor t3 = t1 * t2;
  CHECK_EQ(t3.item(), 10);

  std::vector<float> v = {3, 4, 5, 6};

  // scalar * 1D
  Tensor t4 = Tensor(v);
  Tensor t5 = t1 * t4;
  CHECK_EQ(t5(0), doctest::Approx(6));
  CHECK_EQ(t5(1), doctest::Approx(8));
  CHECK_EQ(t5(2), doctest::Approx(10));
  CHECK_EQ(t5(3), doctest::Approx(12));

  // 1D * scalar
  Tensor t6 = t4 * t1;
  CHECK_EQ(t6(0), doctest::Approx(6));
  CHECK_EQ(t6(1), doctest::Approx(8));
  CHECK_EQ(t6(2), doctest::Approx(10));
  CHECK_EQ(t6(3), doctest::Approx(12));

  // scalar * 2D
  Tensor t7(std::vector<std::vector<float>>{v, v});
  Tensor t8 = t1 * t7;
  CHECK_EQ(t8(0, 0), doctest::Approx(6));
  CHECK_EQ(t8(0, 1), doctest::Approx(8));
  CHECK_EQ(t8(0, 2), doctest::Approx(10));
  CHECK_EQ(t8(0, 3), doctest::Approx(12));
  CHECK_EQ(t8(1, 0), doctest::Approx(6));
  CHECK_EQ(t8(1, 1), doctest::Approx(8));
  CHECK_EQ(t8(1, 2), doctest::Approx(10));
  CHECK_EQ(t8(1, 3), doctest::Approx(12));

  CHECK_THROWS_AS(t8(0, 4), std::invalid_argument);
  CHECK_THROWS_AS(t8(4, 0), std::invalid_argument);

  // 2D * scalar
  Tensor t9 = t7 * t1;
  CHECK_EQ(t9(0, 0), doctest::Approx(6));
  CHECK_EQ(t9(0, 1), doctest::Approx(8));
  CHECK_EQ(t9(0, 2), doctest::Approx(10));
  CHECK_EQ(t9(0, 3), doctest::Approx(12));
  CHECK_EQ(t9(1, 0), doctest::Approx(6));
  CHECK_EQ(t9(1, 1), doctest::Approx(8));
  CHECK_EQ(t9(1, 2), doctest::Approx(10));
  CHECK_EQ(t9(1, 3), doctest::Approx(12));

  CHECK_THROWS_AS(t9(0, 4), std::invalid_argument);
  CHECK_THROWS_AS(t9(4, 0), std::invalid_argument);

  // 1D * 1D
  Tensor t10 = t4 * t4;

  CHECK_EQ(t10(0), doctest::Approx(9));
  CHECK_EQ(t10(1), doctest::Approx(16));
  CHECK_EQ(t10(2), doctest::Approx(25));
  CHECK_EQ(t10(3), doctest::Approx(36));

  // 2D * 2D
  Tensor t11 = t7 * t7;

  CHECK_EQ(t11(0, 0), doctest::Approx(9));
  CHECK_EQ(t11(0, 1), doctest::Approx(16));
  CHECK_EQ(t11(0, 2), doctest::Approx(25));
  CHECK_EQ(t11(0, 3), doctest::Approx(36));
  CHECK_EQ(t11(1, 0), doctest::Approx(9));
  CHECK_EQ(t11(1, 1), doctest::Approx(16));
  CHECK_EQ(t11(1, 2), doctest::Approx(25));
  CHECK_EQ(t11(1, 3), doctest::Approx(36));

  // 1D * 2D & 2D * 1D
  CHECK_THROWS_AS(t10 + t11, std::invalid_argument);
  CHECK_THROWS_AS(t11 + t10, std::invalid_argument);
}

TEST_CASE("Matrix Multiplication 1D x 1D and 2D x 2D") {
  // scalar x scalar
  Tensor sc1(5);
  Tensor sc2(6);
  CHECK_THROWS_AS(sc1.matmul(sc2), std::invalid_argument);

  std::vector<float> v1 = {1, 2, 3, 4, 5};
  std::vector<float> v2 = {2, 4, 6, 8, 10};
  // 1D x 1D
  Tensor t1 = Tensor(v1);
  Tensor t2 = Tensor(v2);
  Tensor t12 = t1.matmul(t2);
  std::vector<float> v3 = {1, 2, 3, 4, 5, 6};  // 1 2
                                               // 3 4
                                               // 5 6
  std::vector<std::size_t> shape1 = {3, 2};
  std::vector<float> v4 = {1, 2, 3, 4, 5, 6};  // 1 2 3
                                               // 4 5 6
  std::vector<std::size_t> shape2 = {2, 3};

  CHECK_EQ(t12.item(), doctest::Approx(110));
  // 2D x 2D
  Tensor t3 = Tensor(v3, shape1);
  Tensor t4 = Tensor(v4, shape2);
  Tensor t34 = t3.matmul(t4);

  CHECK_EQ(t34(0, 0), doctest::Approx(9));
  CHECK_EQ(t34(0, 1), doctest::Approx(12));
  CHECK_EQ(t34(0, 2), doctest::Approx(15));
  CHECK_EQ(t34(1, 0), doctest::Approx(19));
  CHECK_EQ(t34(1, 1), doctest::Approx(26));
  CHECK_EQ(t34(1, 2), doctest::Approx(33));
  CHECK_EQ(t34(2, 0), doctest::Approx(29));
  CHECK_EQ(t34(2, 1), doctest::Approx(40));
  CHECK_EQ(t34(2, 2), doctest::Approx(51));

  CHECK_THROWS_AS(t3.matmul(t3), std::invalid_argument);  // size mismatch
  CHECK_THROWS_AS(t4.matmul(t4), std::invalid_argument);
}

TEST_CASE("Matmul 2D x 1D and 1D x 2D") {
  std::vector<float> v1{5.0f, 10.0f};
  Tensor t1(v1);
  std::vector<float> v2{1.0f, 5.0f, 10.0f};
  Tensor t2(v2);
  std::vector<float> v3 = {1, 2, 3, 4, 5, 6};  // 1 2
                                               // 3 4
                                               // 5 6
  std::vector<std::size_t> shape1 = {3, 2};
  std::vector<float> v4 = {1, 2, 3, 4, 5, 6};  // 1 2 3
                                               // 4 5 6
  std::vector<std::size_t> shape2 = {2, 3};

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

  CHECK_THROWS_AS(t3.matmul(t2), std::invalid_argument);
  CHECK_THROWS_AS(t4.matmul(t1), std::invalid_argument);
  CHECK_THROWS_AS(t2.matmul(t4), std::invalid_argument);
  CHECK_THROWS_AS(t1.matmul(t3), std::invalid_argument);
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
  Tensor A(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}}, true);
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

  Tensor W(std::vector<float>{3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
           std::vector<std::size_t>{2, 3}, true);

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
  Tensor A(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
           std::vector<std::size_t>{3, 2}, true);

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
  Tensor A(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
           std::vector<std::size_t>{2, 3}, true);

  Tensor B(std::vector<float>{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
           std::vector<std::size_t>{3, 2}, true);

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

TEST_CASE("Linear layer creation") {
  Linear layer(3, 2);

  auto params = layer.parameters();

  CHECK_EQ(params.size(), std::size_t(2));

  CHECK_EQ(params[0].first, "weight");
  CHECK_EQ(params[1].first, "bias");

  Tensor weight(params[0].second);
  Tensor bias(params[1].second);

  CHECK_EQ(weight.shape(), std::vector<std::size_t>({3, 2}));

  CHECK_EQ(bias.shape(), std::vector<std::size_t>({2}));

  CHECK_EQ(weight.size(), std::size_t(6));
  CHECK_EQ(bias.size(), std::size_t(2));
}

TEST_CASE("Linear Kaiming initialization") {
  Linear layer(4, 3, 42);

  auto params = layer.parameters();

  Tensor weight(params[0].second);

  // Check weights are not all zero

  bool non_zero = false;

  for (std::size_t i = 0; i < 4; i++) {
    for (std::size_t j = 0; j < 3; j++) {
      if (weight(i, j) != 0.0f) {
        non_zero = true;
      }
    }
  }

  CHECK(non_zero);

  // Check values are inside Kaiming uniform range

  float bound = std::sqrt(6.0f / 4.0f);

  for (std::size_t i = 0; i < 4; i++) {
    for (std::size_t j = 0; j < 3; j++) {
      CHECK(weight(i, j) <= bound);
      CHECK(weight(i, j) >= -bound);
    }
  }
}

TEST_CASE("Linear forward pass") {
  Linear layer(3, 2);

  auto params = layer.parameters();

  Tensor weight(params[0].second);
  Tensor bias(params[1].second);

  /*
      Weight:

      |1 2|
      |3 4|
      |5 6|

      Bias:

      |10 20|
  */

  weight(0, 0) = 1;
  weight(0, 1) = 2;
  weight(1, 0) = 3;
  weight(1, 1) = 4;
  weight(2, 0) = 5;
  weight(2, 1) = 6;

  bias(0) = 10;
  bias(1) = 20;

  Tensor input(std::vector<float>{1, 2, 3});
  Tensor output(layer.forward(input.node()));

  CHECK_EQ(output.shape(), std::vector<std::size_t>({2}));

  /*
      output:

      y0 = 1*1 + 2*3 + 3*5 + 10 = 32

      y1 = 1*2 + 2*4 + 3*6 + 20 = 48
  */

  CHECK_EQ(output(0), doctest::Approx(32));
  CHECK_EQ(output(1), doctest::Approx(48));
}

TEST_CASE("ReLU scalar") {
  Relu relu;

  // Negative scalar
  Tensor t1(-3.5f);
  Tensor out1(relu.forward(t1.node()));
  CHECK_EQ(out1.item(), doctest::Approx(0.0));

  // Zero
  Tensor t2(0.0f);
  Tensor out2(relu.forward(t2.node()));
  CHECK_EQ(out2.item(), doctest::Approx(0.0));

  // Positive scalar
  Tensor t3(4.2f);
  Tensor out3(relu.forward(t3.node()));
  CHECK_EQ(out3.item(), doctest::Approx(4.2));
}

TEST_CASE("ReLU 1D Tensor") {
  Relu relu;

  Tensor input(std::vector<float>{-4, -1, 0, 2, 5});

  Tensor output(relu.forward(input.node()));

  CHECK_EQ(output.shape(), std::vector<std::size_t>({5}));

  CHECK_EQ(output(0), doctest::Approx(0));
  CHECK_EQ(output(1), doctest::Approx(0));
  CHECK_EQ(output(2), doctest::Approx(0));
  CHECK_EQ(output(3), doctest::Approx(2));
  CHECK_EQ(output(4), doctest::Approx(5));
}

TEST_CASE("ReLU 2D Tensor") {
  Relu relu;

  Tensor input(std::vector<std::vector<float>>{
      {-1, 2, -3},
      {4, -5, 6}
  });

  Tensor output(relu.forward(input.node()));

  CHECK_EQ(output.shape(), std::vector<std::size_t>({2, 3}));

  CHECK_EQ(output(0, 0), doctest::Approx(0));
  CHECK_EQ(output(0, 1), doctest::Approx(2));
  CHECK_EQ(output(0, 2), doctest::Approx(0));

  CHECK_EQ(output(1, 0), doctest::Approx(4));
  CHECK_EQ(output(1, 1), doctest::Approx(0));
  CHECK_EQ(output(1, 2), doctest::Approx(6));
}

TEST_CASE("ReLU preserves tensor shape") {
  Relu relu;

  Tensor input(std::vector<std::vector<float>>{
      {-1, -2},
      {3, 4},
      {5, -6}
  });

  Tensor output(relu.forward(input.node()));

  CHECK_EQ(output.shape(), input.shape());
  CHECK_EQ(output.size(), input.size());
}

TEST_CASE("ReLU does not modify input Tensor") {
  Relu relu;

  Tensor input(std::vector<float>{-2, 3, -4});

  Tensor output(relu.forward(input.node()));

  // Original tensor remains unchanged
  CHECK_EQ(input(0), doctest::Approx(-2));
  CHECK_EQ(input(1), doctest::Approx(3));
  CHECK_EQ(input(2), doctest::Approx(-4));

  // Output is activated
  CHECK_EQ(output(0), doctest::Approx(0));
  CHECK_EQ(output(1), doctest::Approx(3));
  CHECK_EQ(output(2), doctest::Approx(0));
}

TEST_CASE("ReLU boundary values") {
  Relu relu;

  Tensor input(std::vector<float>{-0.001f, 0.0f, 0.001f});

  Tensor output(relu.forward(input.node()));

  CHECK_EQ(output(0), doctest::Approx(0.0));
  CHECK_EQ(output(1), doctest::Approx(0.0));
  CHECK_EQ(output(2), doctest::Approx(0.001));
}

TEST_CASE("Softmax scalar") {
  Softmax softmax;

  Tensor input(3.14f);

  Tensor output(softmax.forward(input.node()));

  CHECK_EQ(output.shape(), std::vector<std::size_t>({}));
  CHECK_EQ(output.item(), doctest::Approx(1.0));
}

TEST_CASE("Softmax 1D Tensor") {
  Softmax softmax;

  Tensor input(std::vector<float>{1, 2, 3});

  Tensor output(softmax.forward(input.node()));

  CHECK_EQ(output.shape(), std::vector<std::size_t>({3}));

  CHECK_EQ(output(0), doctest::Approx(0.0900306));
  CHECK_EQ(output(1), doctest::Approx(0.244728));
  CHECK_EQ(output(2), doctest::Approx(0.665241));

  float sum = output(0) + output(1) + output(2);
  CHECK_EQ(sum, doctest::Approx(1.0));
}

TEST_CASE("Softmax identical inputs") {
  Softmax softmax;

  Tensor input(std::vector<float>{5, 5, 5, 5});

  Tensor output(softmax.forward(input.node()));

  CHECK_EQ(output(0), doctest::Approx(0.25));
  CHECK_EQ(output(1), doctest::Approx(0.25));
  CHECK_EQ(output(2), doctest::Approx(0.25));
  CHECK_EQ(output(3), doctest::Approx(0.25));
}

TEST_CASE("Softmax probabilities are between zero and one") {
  Softmax softmax;

  Tensor input(std::vector<float>{-5, -1, 0, 3, 10});

  Tensor output(softmax.forward(input.node()));

  for (std::size_t i = 0; i < output.size(); ++i) {
    CHECK(output(i) >= 0.0f);
    CHECK(output(i) <= 1.0f);
  }
}

TEST_CASE("Softmax probabilities sum to one") {
  Softmax softmax;

  Tensor input(std::vector<float>{-10, 2, 7, 1, 5});

  Tensor output(softmax.forward(input.node()));

  float sum = 0.0f;

  for (std::size_t i = 0; i < output.size(); ++i)
    sum += output(i);

  CHECK_EQ(sum, doctest::Approx(1.0));
}

TEST_CASE("Softmax preserves tensor shape") {
  Softmax softmax;

  Tensor input(std::vector<float>{1, 2, 3, 4, 5});

  Tensor output(softmax.forward(input.node()));

  CHECK_EQ(output.shape(), input.shape());
  CHECK_EQ(output.size(), input.size());
}

TEST_CASE("NLL Loss basic") {
  NLL_Loss loss;

  Tensor input(std::vector<float>{0.1f, 0.2f, 0.7f});

  Tensor output(loss.forward(input.node(), 2));

  CHECK_EQ(output.item(), doctest::Approx(-std::log(0.7f)));
}

TEST_CASE("NLL Loss different targets") {
  NLL_Loss loss;

  Tensor input(std::vector<float>{0.2f, 0.3f, 0.5f});

  Tensor out0(loss.forward(input.node(), 0));
  Tensor out1(loss.forward(input.node(), 1));
  Tensor out2(loss.forward(input.node(), 2));

  CHECK_EQ(out0.item(), doctest::Approx(-std::log(0.2f)));
  CHECK_EQ(out1.item(), doctest::Approx(-std::log(0.3f)));
  CHECK_EQ(out2.item(), doctest::Approx(-std::log(0.5f)));
}

TEST_CASE("NLL Loss probability one") {
  NLL_Loss loss;

  Tensor input(std::vector<float>{0.0f, 1.0f, 0.0f});

  Tensor output(loss.forward(input.node(), 1));

  CHECK_EQ(output.item(), doctest::Approx(0.0));
}

TEST_CASE("NLL Loss input unchanged") {
  NLL_Loss loss;

  Tensor input(std::vector<float>{0.2f, 0.8f});

  Tensor output(loss.forward(input.node(), 1));

  CHECK_EQ(input(0), doctest::Approx(0.2));
  CHECK_EQ(input(1), doctest::Approx(0.8));
}

TEST_CASE("NLL Loss rejects scalar") {
  NLL_Loss loss;

  Tensor input(1.0f);

  CHECK_THROWS_AS(
      loss.forward(input.node(), 0),
      std::runtime_error);
}

TEST_CASE("NLL Loss rejects 2D Tensor") {
  NLL_Loss loss;

  Tensor input(std::vector<std::vector<float>>{
      {0.1f,0.9f},
      {0.3f,0.7f}
  });

  CHECK_THROWS_AS(
      loss.forward(input.node(),0),
      std::runtime_error);
}

TEST_CASE("NLL Loss rejects invalid target") {
  NLL_Loss loss;

  Tensor input(std::vector<float>{0.2f,0.8f});

  CHECK_THROWS_AS(
      loss.forward(input.node(),2),
      std::runtime_error);
}

TEST_CASE("Cross Entropy Loss basic") {
  Cross_Entropy_Loss loss;

  Tensor logits(std::vector<float>{1.0f,2.0f,3.0f});

  Tensor output(loss.forward(logits.node(),2));

  float expected =
      -std::log(std::exp(3.0f) /
                (std::exp(1.0f)+std::exp(2.0f)+std::exp(3.0f)));

  CHECK_EQ(output.item(), doctest::Approx(expected));
}

TEST_CASE("Cross Entropy equals Softmax plus NLL") {
  Cross_Entropy_Loss ce;
  Softmax softmax;
  NLL_Loss nll;

  Tensor logits(std::vector<float>{2,4,1});

  Tensor ce_loss(ce.forward(logits.node(),1));

  Tensor probs(softmax.forward(logits.node()));

  Tensor nll_loss(nll.forward(probs.node(),1));

  CHECK_EQ(ce_loss.item(),
           doctest::Approx(nll_loss.item()));
}

TEST_CASE("Cross Entropy rejects scalar") {
  Cross_Entropy_Loss loss;

  Tensor input(5);

  CHECK_THROWS_AS(
      loss.forward(input.node(),0),
      std::runtime_error);
}

TEST_CASE("Cross Entropy rejects 2D Tensor") {
  Cross_Entropy_Loss loss;

  Tensor input(std::vector<std::vector<float>>{
      {1,2},
      {3,4}
  });

  CHECK_THROWS_AS(
      loss.forward(input.node(),0),
      std::runtime_error);
}

TEST_CASE("Cross Entropy rejects invalid target") {
  Cross_Entropy_Loss loss;

  Tensor input(std::vector<float>{1,2,3});

  CHECK_THROWS_AS(
      loss.forward(input.node(),5),
      std::runtime_error);
}

TEST_CASE("Loss base class throws") {
  Loss loss;

  Tensor input(std::vector<float>{0.2f,0.8f});

  CHECK_THROWS_AS(
      loss.forward(input.node()),
      std::runtime_error);

  CHECK_THROWS_AS(
      loss.forward(input.node(),0),
      std::runtime_error);
}

TEST_CASE("NLL Loss handles zero probability") {
  NLL_Loss loss;

  Tensor input(std::vector<float>{0.0f, 1.0f});

  Tensor output(loss.forward(input.node(), 0));

  CHECK(std::isfinite(output.item()));
  CHECK(output.item() > 20.0f);
}
