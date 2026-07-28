#include "Dataloader.h"

#include <algorithm>
#include <random>

#include "Dataset.h"

Dataloader::Dataloader(Dataset *dataset, int batch_size, bool shuffle)
    : m_dataset(dataset), m_batch_size(batch_size) {
  m_indices.resize(m_dataset->get_length());
  // populate m_indices vector with indices starting from 0 up to length of
  // dataset - 1
  std::iota(m_indices.begin(), m_indices.end(), 0);
  if (shuffle) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(m_indices.begin(), m_indices.end(), g);
  }
}

Dataloader::Iterator::Iterator(Dataloader *dataloader, int index)
    : m_dataloader(dataloader), m_index(index) {}

void Dataloader::Iterator::operator++() {
  m_index += m_dataloader->m_batch_size;
}

std::vector<std::pair<int, std::shared_ptr<Node>>>
Dataloader::Iterator::operator*() {
  std::vector<std::pair<int, std::shared_ptr<Node>>> batch;
  for (int i = 0; i < m_dataloader->m_batch_size; ++i) {
    batch.push_back(m_dataloader->m_dataset->get_item(
        m_dataloader->m_indices[m_index + i]));
  }
  return batch;
}

bool Dataloader::Iterator::operator!=(const Iterator &other) {
  return m_index != other.m_index;
}

Dataloader::Iterator Dataloader::begin() { return Iterator(this, 0); }
Dataloader::Iterator Dataloader::end() {
  return Iterator(this, m_dataset->get_length());
}
