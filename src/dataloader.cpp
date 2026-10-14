#include "dataloader.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {

// Read-only memory map of a whole file. Returns the base pointer and sets size.
void* map_file(const std::string& path, size_t& size) {
#ifdef _WIN32
  HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot open " + path);
  LARGE_INTEGER li{};
  if (!GetFileSizeEx(file, &li)) {
    CloseHandle(file);
    throw std::runtime_error("Cannot stat " + path);
  }
  size = static_cast<size_t>(li.QuadPart);
  HANDLE mapping = size ? CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr) : nullptr;
  CloseHandle(file);
  if (mapping == nullptr) throw std::runtime_error("mmap failed for " + path);
  void* mapped = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  CloseHandle(mapping);  // the view keeps the mapping alive
  if (mapped == nullptr) throw std::runtime_error("mmap failed for " + path);
  return mapped;
#else
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("Cannot open " + path);
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error("Cannot stat " + path);
  }
  size = static_cast<size_t>(st.st_size);
  void* mapped = size ? mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
  close(fd);
  if (mapped == MAP_FAILED) throw std::runtime_error("mmap failed for " + path);
  return mapped;
#endif
}

void unmap_file(void* base, size_t size) {
#ifdef _WIN32
  (void)size;
  UnmapViewOfFile(base);
#else
  munmap(base, size);
#endif
}

}  // namespace

void write_token_file(const std::string& path, const std::vector<uint16_t>& tokens) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write " + path);

  auto count = static_cast<uint64_t>(tokens.size());
  f.write(reinterpret_cast<const char*>(&count), sizeof(count));
  f.write(reinterpret_cast<const char*>(tokens.data()),
          static_cast<std::streamsize>(tokens.size() * sizeof(uint16_t)));
}

DatasetMeta read_dataset_meta(const std::string& data_dir) {
  DatasetMeta meta;
  std::ifstream f(data_dir + "/meta.txt");
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq), value = line.substr(eq + 1);
    if (key == "tokenizer") meta.tokenizer = value;
    if (key == "vocab_size") meta.vocab_size = std::stoi(value);
    if (key == "tokenizer_dir") meta.tokenizer_dir = value;
  }
  return meta;
}

void write_dataset_meta(const std::string& data_dir, const DatasetMeta& meta) {
  std::ofstream f(data_dir + "/meta.txt");
  if (!f) throw std::runtime_error("Cannot write " + data_dir + "/meta.txt");
  f << "tokenizer=" << meta.tokenizer << "\nvocab_size=" << meta.vocab_size << "\n";
  if (!meta.tokenizer_dir.empty()) f << "tokenizer_dir=" << meta.tokenizer_dir << "\n";
}

DataLoader::DataLoader(const std::string& path, int batch_size, int seq_len)
    : batch_size_(batch_size), seq_len_(seq_len) {
  void* mapped = map_file(path, file_size_);

  // Read token count from header and make sure the file actually holds that many tokens
  if (file_size_ < sizeof(uint64_t)) {
    unmap_file(mapped, file_size_);
    throw std::runtime_error("Token file too small: " + path);
  }
  std::memcpy(&num_tokens_, mapped, sizeof(uint64_t));
  if (num_tokens_ > (file_size_ - sizeof(uint64_t)) / sizeof(uint16_t)) {
    unmap_file(mapped, file_size_);
    throw std::runtime_error("Token file header count exceeds file size: " + path);
  }

  // Data starts after the uint64_t header
  data_ = reinterpret_cast<uint16_t*>(static_cast<char*>(mapped) + sizeof(uint64_t));
}

DataLoader::~DataLoader() {
  if (data_ != nullptr) {
    unmap_file(reinterpret_cast<char*>(data_) - sizeof(uint64_t), file_size_);
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
      unmap_file(reinterpret_cast<char*>(data_) - sizeof(uint64_t), file_size_);
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
  size_t total = static_cast<size_t>(batch_size_) * seq_len_;
  batch.tokens.resize(total);
  batch.targets.resize(total);

  for (int b = 0; b < batch_size_; ++b) {
    size_t dst = static_cast<size_t>(b) * seq_len_;
    for (int s = 0; s < seq_len_; ++s) {
      batch.tokens[dst + s] = data_[cursor_ + s];
      batch.targets[dst + s] = data_[cursor_ + s + 1];
    }
    cursor_ += static_cast<size_t>(seq_len_);
  }

  return batch;
}

Batch DataLoader::random_batch(std::mt19937_64& rng) const {
  // Each window needs seq_len + 1 tokens (inputs plus the shifted target)
  if (num_tokens_ < static_cast<uint64_t>(seq_len_) + 1) {
    throw std::runtime_error("Token file shorter than one sequence");
  }
  std::uniform_int_distribution<uint64_t> start_dist(0, num_tokens_ - seq_len_ - 1);

  Batch batch;
  size_t total = static_cast<size_t>(batch_size_) * seq_len_;
  batch.tokens.resize(total);
  batch.targets.resize(total);
  for (int b = 0; b < batch_size_; ++b) {
    uint64_t start = start_dist(rng);
    size_t dst = static_cast<size_t>(b) * seq_len_;
    for (int s = 0; s < seq_len_; ++s) {
      batch.tokens[dst + s] = data_[start + s];
      batch.targets[dst + s] = data_[start + s + 1];
    }
  }
  return batch;
}

void DataLoader::reset() { cursor_ = 0; }
