#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "Neural.h"

NeuralNetwork::NeuralNetwork() {
  register_module("linear_1", m_lin1);
  register_module("linear_2", m_lin2);
  register_module("linear_3", m_lin3);
}

std::shared_ptr<Node> NeuralNetwork::forward(std::shared_ptr<Node> input) {
  Tensor t_input(std::move(input));
  std::shared_ptr<Node> flat = (*m_flatten)(t_input.node());
  std::shared_ptr<Node> linear_1 = (*m_lin1)(flat);
  std::shared_ptr<Node> relu_1 = (*m_relu)(linear_1);
  std::shared_ptr<Node> linear_2 = (*m_lin2)(relu_1);
  std::shared_ptr<Node> relu_2 = (*m_relu)(linear_2);
  std::shared_ptr<Node> linear_3 = (*m_lin3)(relu_2);
  return linear_3;
}

void train(Dataloader& dataloader, NeuralNetwork& model,
           Cross_Entropy_Loss& loss_fn, SGD& optimizer) {
  std::size_t log_interval = 10;
  std::size_t batch_n = 0;
  std::size_t seen_samples = 0;
  for (auto const& batch : dataloader) {
    std::shared_ptr<Node> total_loss = nullptr;
    Tensor total_loss_t(total_loss);
    std::size_t batch_size = batch.size();

    for (const auto& [label, tensor] : batch) {
      auto output = model(tensor);
      auto loss = loss_fn(output, label);
      if (total_loss_t.node() == nullptr) {
        total_loss_t.set_node(loss);
      } else {
        Tensor loss_t(loss);
        total_loss_t = total_loss_t + loss_t;
      }
      seen_samples++;
    }
    total_loss_t.item() /= batch_size;
    if (batch_n % log_interval == 0) {
      std::cout << "loss: " << std::fixed << std::setprecision(6)
                << total_loss_t.item() << " [" << seen_samples << "/"
                << dataloader.n_samples() << "]" << std::endl;
    }
    total_loss_t.backward();
    // std::cout << "Backprop completed \n";
    optimizer.step();
    // std::cout << "SGD step taken \n";
    optimizer.zero_grad();
    // std::cout << "Gradient of optimizer reset to 0 \n";
    batch_n++;
    // std::cout << "Batch number increased \n";
  }
}

void test(Dataloader& dataloader, NeuralNetwork& model,
          Cross_Entropy_Loss& loss_fn) {
  float running_loss = 0.0f;
  std::size_t correct = 0;
  std::size_t n_samples = 0;

  for (const auto& batch : dataloader) {
    for (const auto& [label, tensor] : batch) {
      auto output = model(tensor);
      Tensor output_t(output);
      // accuracy
      if (output_t.argmax() == label) {
        correct++;
      }
      Tensor loss_t(loss_fn(output_t.node(), label));
      running_loss += loss_t.item();
      n_samples++;
    }
  }
  float accuracy = static_cast<float>(correct) / static_cast<float>(n_samples);
  float avg_loss = running_loss / n_samples;

  std::cout << "Accuracy " << std::fixed << std::setprecision(1)
            << accuracy * 100.0 << "%\n"
            << "Avg loss: " << std::setprecision(6) << avg_loss << "\n";
}

void train_new_mnist_model(const bool impose_limit, const int size_limit) {
  MNIST mnist_train =
      MNIST("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
            impose_limit, size_limit);
  MNIST mnist_test =
      MNIST("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
            impose_limit, size_limit);

  std::cout << "Datasets loaded \n";
  int batch_size = 10;
  Dataloader train_dataloader(&mnist_train, batch_size);
  Dataloader test_dataloader(&mnist_test, batch_size);

  NeuralNetwork model;
  Cross_Entropy_Loss loss_fn;
  float learning_rate = 0.001f;
  SGD optimizer(model.parameters(), learning_rate);

  int n_epochs = 1;
  for (int epoch = 0; epoch < n_epochs; ++epoch) {
    std::cout << "[Epoch " << epoch << "/" << n_epochs << "] Training\n";
    train(train_dataloader, model, loss_fn, optimizer);
    std::cout << "[Epoch " << epoch << "/" << n_epochs << "] Testing\n";
    test(test_dataloader, model, loss_fn);
  }

  auto state_dict = model.state_dict();
  save(state_dict, "models/mnist.nn");
  std::cout << "Model saved \n";
}

void inference_on_saved_model(const bool impose_limit, const int size_limit,
                              bool use_pre_trained) {
  NeuralNetwork model;
  std::cout << "Loading model...\n";
  std::string pre_trained_path("models/pre_trained_mnist.nn");
  std::string new_model_path("models/mnist.nn");
  auto loaded_state_dict =
      use_pre_trained ? load(pre_trained_path) : load(new_model_path);
  model.load_state_dict(loaded_state_dict);

  std::cout << "Loading test set...\n";
  MNIST mnist_test =
      MNIST("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
            impose_limit, size_limit);

  int n_samples = 10;

  std::vector<int> all_indices(mnist_test.get_length());
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(all_indices.begin(), all_indices.end(), g);
  std::vector<int> indices(all_indices.begin(),
                           all_indices.begin() + n_samples);

  if (use_pre_trained) {
    std::cout << "Using pre-trained model \n";
  }
  for (int i = 0; i < n_samples; ++i) {
    std::cout << "Sample " << i << " of " << n_samples << std::endl;
    std::pair<int, std::shared_ptr<Node>> sample_image =
        mnist_test.get_item(indices[i]);
    visualize_image(sample_image.second);
    auto output = model(sample_image.second);
    Tensor output_t(output);
    int predicted_class = output_t.argmax();
    std::cout << "Predicted class "
              << mnist_test.label_to_class(predicted_class) << std::endl;
    std::cout << "True class " << mnist_test.label_to_class(sample_image.first)
              << std::endl;
    std::cout << "-------------------------------------------\n";
  }
}
