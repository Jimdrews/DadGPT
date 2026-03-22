#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "model.hpp"

TEST(Model, DefaultConfig) {
  ModelConfig config;
  EXPECT_EQ(config.d_model, 256);
  EXPECT_EQ(config.n_layers, 4);
  EXPECT_EQ(config.n_heads, 4);
  EXPECT_EQ(config.d_ff, 1024);
  EXPECT_EQ(config.ctx_len, 128);
  EXPECT_EQ(config.vocab_size, 4096);
  EXPECT_EQ(config.d_head(), 64);
}

TEST(Model, ParameterCount) {
  ModelConfig config;
  Model model{};
  model_alloc(model, config);

  // Should be approximately 4.2M params
  EXPECT_GT(model.param_offsets.total, 4000000u);
  EXPECT_LT(model.param_offsets.total, 5000000u);

  model_free(model);
}

TEST(Model, WeightInitNonZero) {
  ModelConfig config;
  Model model{};
  model_alloc(model, config);

  // Token embeddings should be non-zero
  float sum = 0.0f;
  int emb_size = config.vocab_size * config.d_model;
  for (int i = 0; i < emb_size; ++i) {
    sum += std::abs(model.param(model.param_offsets.tok_emb)[i]);
  }
  EXPECT_GT(sum, 0.0f);

  // Biases should be zero
  float bias_sum = 0.0f;
  for (int i = 0; i < config.d_model; ++i) {
    bias_sum += std::abs(model.param(model.param_offsets.layers[0].bq)[i]);
  }
  EXPECT_NEAR(bias_sum, 0.0f, 1e-10);

  model_free(model);
}

TEST(Model, ForwardPassOutputShape) {
  // Use a tiny model for speed
  ModelConfig config;
  config.d_model = 32;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 64;
  config.ctx_len = 16;
  config.vocab_size = 100;

  Model model{};
  model_alloc(model, config);

  uint16_t tokens[] = {4, 5, 6, 7, 8};
  int seq_len = 5;

  model_forward(model, tokens, seq_len);

  // Check logits are finite and non-zero
  float* logits = model.act(model.act_offsets.logits);
  float sum = 0.0f;
  bool all_finite = true;
  for (int i = 0; i < seq_len * config.vocab_size; ++i) {
    sum += std::abs(logits[i]);
    if (!std::isfinite(logits[i])) all_finite = false;
  }
  EXPECT_TRUE(all_finite);
  EXPECT_GT(sum, 0.0f);

  model_free(model);
}

TEST(Model, CausalMasking) {
  // Run forward pass twice with different future tokens — results at earlier positions should match
  ModelConfig config;
  config.d_model = 32;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 64;
  config.ctx_len = 16;
  config.vocab_size = 100;

  Model model{};
  model_alloc(model, config);

  uint16_t tokens1[] = {4, 5, 6, 10, 20};
  uint16_t tokens2[] = {4, 5, 6, 30, 40};

  model_forward(model, tokens1, 5);
  std::vector<float> logits1(model.act(model.act_offsets.logits),
                             model.act(model.act_offsets.logits) + 3 * config.vocab_size);

  model_forward(model, tokens2, 5);
  std::vector<float> logits2(model.act(model.act_offsets.logits),
                             model.act(model.act_offsets.logits) + 3 * config.vocab_size);

  // First 3 positions should produce identical logits (causal = no future leaking)
  for (int i = 0; i < 3 * config.vocab_size; ++i) {
    EXPECT_NEAR(logits1[i], logits2[i], 1e-5) << "Mismatch at logit index " << i;
  }

  model_free(model);
}

TEST(Model, SaveLoadRoundTrip) {
  ModelConfig config;
  config.d_model = 32;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 64;
  config.ctx_len = 16;
  config.vocab_size = 100;

  Model model{};
  model_alloc(model, config);

  std::string path = "/tmp/test_model.bin";
  model_save(model, path);

  Model loaded{};
  model_load(loaded, path);

  EXPECT_EQ(loaded.config.d_model, config.d_model);
  EXPECT_EQ(loaded.config.n_layers, config.n_layers);
  EXPECT_EQ(loaded.param_offsets.total, model.param_offsets.total);

  for (size_t i = 0; i < model.param_offsets.total; ++i) {
    EXPECT_NEAR(loaded.params[i], model.params[i], 1e-7) << "Param mismatch at index " << i;
  }

  model_free(model);
  model_free(loaded);
  std::remove(path.c_str());
}
