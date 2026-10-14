#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "dataloader.hpp"
#include "test_util.hpp"

namespace {

std::string create_test_token_file(const std::vector<uint16_t>& tokens) {
  std::string path = test_util::temp_path("tokens.bin");
  write_token_file(path, tokens);
  return path;
}

}  // namespace

TEST(DataLoader, BatchShape) {
  // Create a token file with enough data for 2 batches of (batch=2, seq=4)
  // Each batch needs batch_size * (seq_len + 1) = 2 * 5 = 10 tokens
  std::vector<uint16_t> tokens(20);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<uint16_t>(i % 100);
  }

  auto path = create_test_token_file(tokens);
  DataLoader loader(path, 2, 4);

  EXPECT_TRUE(loader.has_next());
  auto batch = loader.next_batch();

  // Flat: batch_size * seq_len = 2 * 4 = 8
  ASSERT_EQ(batch.tokens.size(), 8);
  ASSERT_EQ(batch.targets.size(), 8);

  std::remove(path.c_str());
}

TEST(DataLoader, TargetsAreShiftedInputs) {
  std::vector<uint16_t> tokens = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  auto path = create_test_token_file(tokens);
  DataLoader loader(path, 1, 4);

  auto batch = loader.next_batch();

  // tokens = [10, 20, 30, 40], targets = [20, 30, 40, 50]
  EXPECT_EQ(batch.tokens[0], 10);
  EXPECT_EQ(batch.tokens[3], 40);
  EXPECT_EQ(batch.targets[0], 20);
  EXPECT_EQ(batch.targets[3], 50);

  // Verify shift: targets[j] == tokens[j] + 10 (because source is 10,20,30,...)
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(batch.targets[j], batch.tokens[j] + 10) << "Shift mismatch at position " << j;
  }

  std::remove(path.c_str());
}

TEST(DataLoader, EpochBoundary) {
  // 10 tokens, batch=1, seq=4 → needs 5 tokens per batch → 2 batches
  std::vector<uint16_t> tokens(10);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<uint16_t>(i);
  }

  auto path = create_test_token_file(tokens);
  DataLoader loader(path, 1, 4);

  EXPECT_TRUE(loader.has_next());
  loader.next_batch();
  EXPECT_TRUE(loader.has_next());
  loader.next_batch();
  EXPECT_FALSE(loader.has_next());

  loader.reset();
  EXPECT_TRUE(loader.has_next());

  std::remove(path.c_str());
}

TEST(DataLoader, TokenCountHeader) {
  std::vector<uint16_t> tokens = {1, 2, 3, 4, 5};
  auto path = create_test_token_file(tokens);

  DataLoader loader(path, 1, 2);
  EXPECT_EQ(loader.num_tokens(), 5);

  std::remove(path.c_str());
}

TEST(DataLoader, RandomBatchWindowsAreShiftedAndInRange) {
  std::vector<uint16_t> tokens(500);
  for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = static_cast<uint16_t>(i);
  auto path = create_test_token_file(tokens);
  DataLoader loader(path, 4, 16);

  std::mt19937_64 rng(1);
  std::set<uint16_t> starts;
  for (int it = 0; it < 50; ++it) {
    Batch batch = loader.random_batch(rng);
    ASSERT_EQ(batch.tokens.size(), 64u);
    for (int b = 0; b < 4; ++b) {
      uint16_t start = batch.tokens[b * 16];
      starts.insert(start);
      for (int s = 0; s < 16; ++s) {
        // Token value == stream index, so windows must be contiguous and targets shifted by 1
        EXPECT_EQ(batch.tokens[b * 16 + s], start + s);
        EXPECT_EQ(batch.targets[b * 16 + s], start + s + 1);
      }
      EXPECT_LE(start + 16, 499);  // the last target must exist
    }
  }
  EXPECT_GT(starts.size(), 100u);  // offsets actually vary
}

TEST(DataLoader, RejectsHeaderLongerThanFile) {
  std::string path = test_util::temp_path("bad_tokens.bin");
  {
    std::ofstream f(path, std::ios::binary);
    uint64_t count = 1000;  // claims 1000 tokens but stores 2
    uint16_t data[2] = {1, 2};
    f.write(reinterpret_cast<const char*>(&count), sizeof(count));
    f.write(reinterpret_cast<const char*>(data), sizeof(data));
  }
  EXPECT_THROW(DataLoader(path, 1, 1), std::runtime_error);
  std::remove(path.c_str());
}

TEST(DataLoader, DatasetMetaRoundTrip) {
  auto dir = std::filesystem::path(test_util::temp_path("meta_dir"));
  std::filesystem::create_directories(dir);
  EXPECT_EQ(read_dataset_meta(dir.string()).tokenizer, "dadbpe");  // legacy default
  write_dataset_meta(dir.string(), {"gpt2", 50257});
  DatasetMeta meta = read_dataset_meta(dir.string());
  EXPECT_EQ(meta.tokenizer, "gpt2");
  EXPECT_EQ(meta.vocab_size, 50257);
  std::filesystem::remove_all(dir);
}
