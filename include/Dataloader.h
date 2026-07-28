#ifndef DATALOADER_H
#define DATALOADER_H

#include <memory>
#include <utility>
#include <vector>

#include "Dataset.h"

class Dataloader {
 private:
  Dataset* m_dataset;
  int m_batch_size;
  std::vector<int> m_indices;

 public:
  Dataloader(Dataset* dataset, int batch_size, bool shuffle = true);

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
