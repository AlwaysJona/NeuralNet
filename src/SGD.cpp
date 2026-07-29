#include "SGD.h"

#include <memory>
#include <string>
#include <vector>

#include "Tensor.h"

SGD::SGD(list<Node> par, const float& lr)
    : m_params(par), m_learning_rate(lr) {}

void SGD::step() {
  for (auto& par : m_params) {
    Tensor t(par.second);
    for (std::size_t i = 0; i < t.size(); ++i) {
      t.data()[i] -= m_learning_rate * t.grad()[i];
    }
  }
}

void SGD::zero_grad() {
  for (auto& par : m_params) {
    Tensor t(par.second);
    t.zero_grad();
  }
}
