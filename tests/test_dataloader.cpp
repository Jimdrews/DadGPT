#include <gtest/gtest.h>

#include <cstdio>
#include <vector>

#include "dataloader.hpp"

namespace {

std::string create_test_token_file(const std::vector<uint16_t>& tokens) {
  std::string path = "/tmp/test_tokens.bin";
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

  ASSERT_EQ(batch.tokens.size(), 2);
  ASSERT_EQ(batch.tokens[0].size(), 4);
  ASSERT_EQ(batch.targets.size(), 2);
  ASSERT_EQ(batch.targets[0].size(), 4);

  std::remove(path.c_str());
}

TEST(DataLoader, TargetsAreShiftedInputs) {
  std::vector<uint16_t> tokens = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  auto path = create_test_token_file(tokens);
  DataLoader loader(path, 1, 4);

  auto batch = loader.next_batch();

  // tokens[0] = [10, 20, 30, 40], targets[0] = [20, 30, 40, 50]
  EXPECT_EQ(batch.tokens[0][0], 10);
  EXPECT_EQ(batch.tokens[0][3], 40);
  EXPECT_EQ(batch.targets[0][0], 20);
  EXPECT_EQ(batch.targets[0][3], 50);

  // Verify shift: targets[i][j] == tokens[i][j+1] in the original data
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(batch.targets[0][j], batch.tokens[0][j] + 10)
        << "Shift mismatch at position " << j;
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
