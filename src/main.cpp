#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Dataloader.h"
#include "Dataset.h"
#include "Flatten.h"
#include "Linear.h"
#include "Loss.h"
#include "Module.h"
#include "Relu.h"
#include "SGD.h"
#include "Serialization.h"
#include "Softmax.h"
#include "Tensor.h"
#include "Neural.h"

void neural_net_test() {
  NeuralNetwork model;

  // random input
  std::vector<std::vector<float>> input_data(28, std::vector<float>(28));
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto& row : input_data) {
    for (auto& val : row) {
      val = dist(rng);
    }
  }
  // create input tensor
  Tensor input_t(input_data);

  std::cout << "Tensor created \n";

  // forward pass
  std::shared_ptr<Node> output = model(input_t.node());

  std::cout << "Forward pass called, output created \n";

  Tensor output_t(output);

  std::cout << "Output tensor: " << output_t << std::endl;
}

void dataset_loading_test(const bool impose_limit,
                          const int& size_limit = 60000) {
  std::cout << "Before loading data \n";
  MNIST mnist_train =
      MNIST("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
            impose_limit, size_limit);

  int batch_size = 10;
  Dataloader mnist_train_loader(&mnist_train, batch_size);

  std::cout << "Dataset loaded with " << mnist_train_loader.n_samples()
            << " samples\n";
  std::cout << "Visualizing first batch of training data \n" << std::endl;

  for (auto batch : mnist_train_loader) {
    for (auto item : batch) {
      visualize_image(item.second);
      std::cout << mnist_train.label_to_class(item.first) << std::endl;
    }
    break;
  }
}

int main() {
  int size_limit = 1000;
  bool impose_limit = true;
  // dataset_loading_test(impose_limit, size_limit);
  train_new_mnist_model(impose_limit, size_limit);
  inference_on_saved_model(impose_limit, size_limit);
  return 0;
}
