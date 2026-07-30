#ifndef DATALOADER_H
#define DATALOADER_H

#include <memory>
#include <utility>
#include <vector>

#include "Dataset.h"

// A Dataloader takes a Dataset object and prepares it for the neural network,
// by shuffling its items, and setting batches of images to be processed
//
// Implementation in Dataloader.h

class Dataloader {
 private:
  Dataset* m_dataset;
  int m_batch_size;
  std::vector<int> m_indices;

 public:
  Dataloader(Dataset* dataset, int batch_size, bool shuffle = true);
  
  // The iterator class implements a custom iteration method, such that
  // if obj is a Dataloader object, the loop: 
  //
  // for(auto batch : obj) { ... }
  //
  // loops directly over a collection of images (but not all of them,
  // just a few at once), and corresponding labels, and consequently:
  //
  // for(auto image : batch) { ... }
  //
  // loops over single images, and corresponding labels
  class Iterator {
   private:
    Dataloader* m_dataloader;
    int m_index;

   public:
    Iterator(Dataloader* dataloader, int index);
    void operator++();
    std::vector<std::pair<int, std::shared_ptr<Node>>> operator*();
    bool operator!=(const Iterator& other);
  };

  Iterator begin();
  Iterator end();
  int batch_size() const { return m_batch_size; }
  int n_samples() const { return m_dataset->get_length(); }
  int n_batches() const {
    return (n_samples() + batch_size() - 1) / batch_size();
  }
};

#endif
