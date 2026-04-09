#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Batch {
  std::vector<uint16_t> tokens;   // [batch_size * seq_len] flat contiguous
  std::vector<uint16_t> targets;  // [batch_size * seq_len] flat contiguous
};

// Write a token stream to a binary file: uint64_t header + uint16_t[] data
void write_token_file(const std::string& path, const std::vector<uint16_t>& tokens);

class DataLoader {
 public:
  DataLoader(const std::string& path, int batch_size, int seq_len);
  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;
  DataLoader(DataLoader&& other) noexcept;
  DataLoader& operator=(DataLoader&& other) noexcept;

  Batch next_batch();
  void reset();
  bool has_next() const;
  uint64_t num_tokens() const { return num_tokens_; }

 private:
  uint16_t* data_ = nullptr;
  uint64_t num_tokens_ = 0;
  size_t file_size_ = 0;
  size_t cursor_ = 0;
  int batch_size_;
  int seq_len_;
};
