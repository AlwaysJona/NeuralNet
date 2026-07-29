#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Neural.h"

// Test function to verify proper creation and working of NeuralNetwork object
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

// Test function to verify proper creation and working of Dataset and Dataloader
// objects
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
  int size_limit = 1000;  // Number of samples to train on and to test on.
                          // The higher the number, the longer the program is
                          // going to take, and the more accurate the model will be.
                          // WARNING: the full MNIST dataset contains 60k images,
                          // training on the full dataset can take from 30 minutes to
                          // 1 hour, while training on 1000 images takes about 1 minute
                          // RECOMMENDATION: set size_limit > 10k only if 
                          // you want to modify the model and the code

  bool impose_limit = true;  // True if you'd like to apply the above size limit 
                             // to the model
                             
  bool use_pre_trained = true; // True if you'd like to use the accurate pre-trained
                               // model, trained on 60k images

  // dataset_loading_test(impose_limit, size_limit);
  
  train_new_mnist_model(impose_limit, size_limit); // Declared in Neural.h and 
                                                   // defined in Neural.cpp.
                                                   // Call it to train a new model.
                                                   // Not needed if you want
                                                   // to use the pre-trained model

  inference_on_saved_model(impose_limit, size_limit, use_pre_trained); // use to test the model
                                                                       // on the test set and
                                                                       // visualize a few images, their
                                                                       // label, and the model's
                                                                       // guess
  return 0;
}
