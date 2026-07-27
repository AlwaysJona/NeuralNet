#include "Linear.h"

#include <memory>
#include <random>

Linear::Linear(std::size_t in_feat, std::size_t out_feat, std::size_t seed)
    : m_in_feature(in_feat),
      m_out_feature(out_feat),
      m_weight(Tensor(std::vector<std::vector<float>>(
                          in_feat, std::vector<float>(out_feat, 0.0f)),
                      true)),
      m_bias(Tensor(std::vector<float>(out_feat, 0.0f), true)),
      m_seed(seed) {
  // register parameters
  register_parameter("weight", m_weight.node());
  register_parameter("bias", m_bias.node());

  // Kaiming initialization
  reset_parameters();
}

void Linear::reset_parameters() {
  float gain = std::sqrt(2.0f);
  std::size_t fan_in = m_in_feature;
  float bound = gain * std::sqrt(3.0f / fan_in);
  std::mt19937 generator(m_seed);

  for (std::size_t i = 0; i < m_weight.shape()[0]; ++i) {
    for (std::size_t j = 0; j < m_weight.shape()[1]; ++j) {
      m_weight(i, j) =
          std::uniform_real_distribution<float>(-bound, bound)(generator);
    }
  }
}

std::shared_ptr<Node> Linear::forward(std::shared_ptr<Node> input) {
  Tensor t_input = Tensor(input);
  Tensor result = t_input.matmul(m_weight) + m_bias;
  return result.node();
}
