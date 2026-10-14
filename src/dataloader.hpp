#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct Batch {
  std::vector<uint16_t> tokens;   // [batch_size * seq_len] flat contiguous
  std::vector<uint16_t> targets;  // [batch_size * seq_len] flat contiguous
};

// Write a token stream to a binary file: uint64_t header + uint16_t[] data
void write_token_file(const std::string& path, const std::vector<uint16_t>& tokens);

// Which tokenizer produced a data directory's token files (data_dir/meta.txt, key=value lines).
// Directories prepared before meta.txt existed read as the 512-token dad-jokes BPE.
struct DatasetMeta {
  std::string tokenizer = "dadbpe";  // "dadbpe" or "gpt2"
  int vocab_size = 512;
};
DatasetMeta read_dataset_meta(const std::string& data_dir);
void write_dataset_meta(const std::string& data_dir, const DatasetMeta& meta);

class DataLoader {
 public:
  DataLoader(const std::string& path, int batch_size, int seq_len);
  ~DataLoader();

  DataLoader(const DataLoader&) = delete;
  DataLoader& operator=(const DataLoader&) = delete;
  DataLoader(DataLoader&& other) noexcept;
  DataLoader& operator=(DataLoader&& other) noexcept;

  // Sequential windows (used for evaluation): row b starts where row b-1 ended
  Batch next_batch();
  // nanoGPT-style training batch: batch_size windows at independent random offsets
  Batch random_batch(std::mt19937_64& rng) const;
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
