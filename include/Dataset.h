#ifndef DATASET_H
#define DATASET_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Tensor.h"

using Node = Tensor::Node;

class Dataset {
 public:
  virtual std::pair<int, std::shared_ptr<Node>> get_item(int index) = 0;
  virtual int get_length() = 0;
};

class MNIST : public Dataset {
 private:
  std::vector<std::vector<std::vector<float>>> m_images;  // vector of 2D images
  std::vector<int> m_labels;
  std::vector<std::string> classes = {"zero", "one", "two",   "three", "four",
                                      "five", "six", "seven", "eight", "nine"};

 public:
  MNIST(std::string data_path, std::string labels_path, const bool impose_limit,
        const int& size_limit = 60000);
  std::pair<int, std::shared_ptr<Node>> get_item(int index) override;
  int get_length() override;
  std::string label_to_class(int label);
};

void visualize_image(std::shared_ptr<Node> image);

#endif
