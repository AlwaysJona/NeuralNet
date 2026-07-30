#ifndef DATASET_H
#define DATASET_H

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "Tensor.h"

// Dataset class is used to read the MNIST dataset and properly write it
// into Tensor data. The MNIST dataset is in given in a very specific format:
// It starts with a MAGIC NUMBER, which is used to differentiate images from
// labels, and then in order there are: number of images, number of rows,
// number of columns, and then the pixel data, everything is in binary.
//
// Implementation in Dataset.cpp

using Node = Tensor::Node;

// Abstract base class
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
  // constructor
  MNIST(std::string data_path, std::string labels_path, const bool impose_limit,
        const int& size_limit = 60000);
  // getters
  // get a single image with its label
  std::pair<int, std::shared_ptr<Node>> get_item(int index) override;
  // returns the number of images in the dataset
  int get_length() override;
  std::string label_to_class(int label);
};

// visualize images in terminal with ascii characters
void visualize_image(std::shared_ptr<Node> image);

#endif
