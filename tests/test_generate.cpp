#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "generate.hpp"
#include "optimizer.hpp"

TEST(Sampling, GreedyPicksArgmax) {
  std::mt19937 rng(0);
  std::vector<float> logits = {0.1f, 2.5f, -1.0f, 2.4f};
  EXPECT_EQ(sample_token(logits, {}, 0.0f, 0, 1.0f, rng), 1);
  EXPECT_EQ(sample_token(logits, {}, 1.0f, 1, 1.0f, rng), 1);  // top-k 1 is greedy too
}

TEST(Sampling, TopKKeepsExactlyK) {
  // Ten distinct logits; with top_k = 3 only the three largest may ever be sampled.
  std::vector<float> logits = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  std::mt19937 rng(42);
  std::set<uint16_t> seen;
  for (int i = 0; i < 5000; ++i) {
    seen.insert(sample_token(logits, {}, 1.0f, 3, 1.0f, rng));
  }
  EXPECT_EQ(seen, (std::set<uint16_t>{7, 8, 9}));
}

TEST(Sampling, RepetitionPenaltyAppliesOncePerToken) {
  // Token 0 appears three times in history. A penalty of 2 applied once gives 2.0 / 2 = 1.0,
  // still above token 1's 0.9; applied three times (2 / 8 = 0.25) it would lose.
  std::vector<float> logits = {2.0f, 0.9f, -5.0f};
  std::mt19937 rng(0);
  EXPECT_EQ(sample_token(logits, {0, 0, 0}, 0.0f, 0, 2.0f, rng), 0);

  // Negative logits are multiplied (pushed further down), not divided toward zero.
  std::vector<float> neg = {-1.0f, -1.5f};
  EXPECT_EQ(sample_token(neg, {0}, 0.0f, 0, 2.0f, rng), 1);  // -1 * 2 = -2 < -1.5
}

TEST(Sampling, NeverReturnsZeroProbabilityToken) {
  // All mass on the last token: sampling must never fall back to token 0.
  std::vector<float> logits(50, -1e4f);
  logits.back() = 10.0f;
  std::mt19937 rng(5);
  for (int i = 0; i < 2000; ++i) {
    ASSERT_EQ(sample_token(logits, {}, 1.0f, 0, 1.0f, rng), 49);
  }
}

TEST(Schedule, WarmupThenCosineToTenPercent) {
  Adam opt;
  opt.max_lr = 1.0f;
  opt.warmup_steps = 10;
  opt.total_steps = 110;

  opt.step = 0;
  EXPECT_FLOAT_EQ(adam_lr(opt), 0.0f);
  opt.step = 5;
  EXPECT_FLOAT_EQ(adam_lr(opt), 0.5f);
  opt.step = 10;
  EXPECT_NEAR(adam_lr(opt), 1.0f, 1e-6);
  opt.step = 60;  // halfway through the decay
  EXPECT_NEAR(adam_lr(opt), 0.55f, 1e-5);
  opt.step = 110;
  EXPECT_NEAR(adam_lr(opt), 0.1f, 1e-6);
  opt.step = 500;  // clamped past the end
  EXPECT_NEAR(adam_lr(opt), 0.1f, 1e-6);
}
