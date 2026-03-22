#include "dataloader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <stdexcept>

void write_token_file(const std::string& path, const std::vector<uint16_t>& tokens) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write " + path);

  auto count = static_cast<uint64_t>(tokens.size());
  f.write(reinterpret_cast<const char*>(&count), sizeof(count));
  f.write(reinterpret_cast<const char*>(tokens.data()),
          static_cast<std::streamsize>(tokens.size() * sizeof(uint16_t)));
}

DataLoader::DataLoader(const std::string& path, int batch_size, int seq_len)
    : batch_size_(batch_size), seq_len_(seq_len) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("Cannot open " + path);

  struct stat st {};
  fstat(fd, &st);
  file_size_ = static_cast<size_t>(st.st_size);

  void* mapped = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);

  if (mapped == MAP_FAILED) {
    throw std::runtime_error("mmap failed for " + path);
  }

  // Read token count from header
  std::memcpy(&num_tokens_, mapped, sizeof(uint64_t));

  // Data starts after the uint64_t header
  data_ = reinterpret_cast<uint16_t*>(static_cast<char*>(mapped) + sizeof(uint64_t));
}

DataLoader::~DataLoader() {
  if (data_ != nullptr) {
    void* base = reinterpret_cast<char*>(data_) - sizeof(uint64_t);
    munmap(base, file_size_);
  }
}

DataLoader::DataLoader(DataLoader&& other) noexcept
    : data_(other.data_),
      num_tokens_(other.num_tokens_),
      file_size_(other.file_size_),
      cursor_(other.cursor_),
      batch_size_(other.batch_size_),
      seq_len_(other.seq_len_) {
  other.data_ = nullptr;
  other.num_tokens_ = 0;
}

DataLoader& DataLoader::operator=(DataLoader&& other) noexcept {
  if (this != &other) {
    if (data_ != nullptr) {
      void* base = reinterpret_cast<char*>(data_) - sizeof(uint64_t);
      munmap(base, file_size_);
    }
    data_ = other.data_;
    num_tokens_ = other.num_tokens_;
    file_size_ = other.file_size_;
    cursor_ = other.cursor_;
    batch_size_ = other.batch_size_;
    seq_len_ = other.seq_len_;
    other.data_ = nullptr;
    other.num_tokens_ = 0;
  }
  return *this;
}

bool DataLoader::has_next() const {
  // Need seq_len + 1 tokens per sequence (for the shifted target)
  size_t tokens_per_batch =
      static_cast<size_t>(batch_size_) * static_cast<size_t>(seq_len_ + 1);
  return cursor_ + tokens_per_batch <= num_tokens_;
}

Batch DataLoader::next_batch() {
  Batch batch;
  batch.tokens.resize(batch_size_, std::vector<uint16_t>(seq_len_));
  batch.targets.resize(batch_size_, std::vector<uint16_t>(seq_len_));

  for (int b = 0; b < batch_size_; ++b) {
    for (int s = 0; s < seq_len_; ++s) {
      batch.tokens[b][s] = data_[cursor_ + s];
      batch.targets[b][s] = data_[cursor_ + s + 1];
    }
    cursor_ += static_cast<size_t>(seq_len_);
  }

  return batch;
}

void DataLoader::reset() { cursor_ = 0; }
