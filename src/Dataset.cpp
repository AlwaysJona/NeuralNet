#include "Dataset.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "Tensor.h"

// reverse byte order
// not actually needed, std::byteswap in <bit> does the same
// implemented for educational purposes
int reverse_int(int i) {
  unsigned char c1, c2, c3, c4;
  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

float convert_to_float(unsigned char px) { return (float)px / 255.0f; }

// Reading images and saving them to Tensors, the output is a 
// collection of 2D vectors, the values of each pixel
std::vector<std::vector<std::vector<float>>> read_mnist(std::string path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<std::vector<std::vector<float>>> dataset;
  if (file.is_open()) {
    int magic_number = 0;
    int num_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    // mnist files are in big endian format, so need to reverse order of bytes
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
      throw std::runtime_error("Invalid MNIST image file \n");
    }
    file.read((char *)&num_of_images, sizeof(num_of_images));
    num_of_images = reverse_int(num_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    for (int i = 0; i < num_of_images; ++i) {
      std::vector<std::vector<float>> image;
      for (int r = 0; r < n_rows; ++r) {
        std::vector<float> row;
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          row.push_back(convert_to_float(temp));
        }
        image.push_back(row);
      }
      dataset.push_back(image);
    }
  }
  return dataset;
}

// overload for custom sized dataset
std::vector<std::vector<std::vector<float>>> read_mnist(
    std::string path, const int &image_limit) {
  std::ifstream file(path, std::ios::binary);
  std::vector<std::vector<std::vector<float>>> dataset;
  if (file.is_open()) {
    int magic_number = 0;
    int num_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    // mnist files are in big endian format, so need to reverse order of bytes
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
      throw std::runtime_error("Invalid MNIST image file \n");
    }
    file.read((char *)&num_of_images, sizeof(num_of_images));
    num_of_images = reverse_int(num_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);
    num_of_images = num_of_images < image_limit ? num_of_images : image_limit;
    for (int i = 0; i < num_of_images; ++i) {
      std::vector<std::vector<float>> image;
      for (int r = 0; r < n_rows; ++r) {
        std::vector<float> row;
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));
          row.push_back(convert_to_float(temp));
        }
        image.push_back(row);
      }
      dataset.push_back(image);
    }
  }
  return dataset;
}

// Reading labels and saving them to vectors,
// The output is just a collection of integers,
// corresponding to the label of each image
std::vector<int> read_mnist_labels(std::string path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<int> labels;
  if (file.is_open()) {
    int magic_number = 0;
    int num_of_items = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049) {
      throw std::runtime_error("Invalid MNIST label file \n");
    }
    file.read((char *)&num_of_items, sizeof(num_of_items));
    num_of_items = reverse_int(num_of_items);
    for (int i = 0; i < num_of_items; ++i) {
      unsigned char label = 0;
      file.read((char *)&label, sizeof(label));
      labels.push_back(label);
    }
    return labels;
  }
}

// overload for custom sized dataset
std::vector<int> read_mnist_labels(std::string path, const int &label_limit) {
  std::ifstream file(path, std::ios::binary);
  std::vector<int> labels;
  if (file.is_open()) {
    int magic_number = 0;
    int num_of_items = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049) {
      throw std::runtime_error("Invalid MNIST label file \n");
    }
    file.read((char *)&num_of_items, sizeof(num_of_items));
    num_of_items = reverse_int(num_of_items);
    num_of_items = num_of_items < label_limit ? num_of_items : label_limit;
    for (int i = 0; i < num_of_items; ++i) {
      unsigned char label = 0;
      file.read((char *)&label, sizeof(label));
      labels.push_back(label);
    }
    return labels;
  }
}

MNIST::MNIST(std::string data_path, std::string labels_path,
             const bool impose_limit, const int &size_limit) {
  if (impose_limit) {
    m_images = read_mnist(data_path, size_limit);
    m_labels = read_mnist_labels(labels_path, size_limit);
  } else {
    m_images = read_mnist(data_path);
    m_labels = read_mnist_labels(labels_path);
  }
}

std::pair<int, std::shared_ptr<Node>> MNIST::get_item(int index) {
  Tensor im(m_images[index]);
  return std::make_pair(m_labels[index], im.node());
}

int MNIST::get_length() { return m_images.size(); }

std::string MNIST::label_to_class(int label) { return classes[label]; }

void visualize_image(std::shared_ptr<Node> image) {
  Tensor im(image);
  for (int i = 0; i < im.shape()[0]; ++i) {
    for (int j = 0; j < im.shape()[1]; ++j) {
      float px = im(i, j);
      std::cout << (px > 0.75   ? '@'
                    : px > 0.5  ? '#'
                    : px > 0.25 ? '+'
                    : px > 0.1  ? '.'
                                : ' ');
    }
    std::cout << std::endl;
  }
}
