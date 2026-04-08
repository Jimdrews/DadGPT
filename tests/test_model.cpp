#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#include "gpu_ops.cuh"
#include "model.hpp"

TEST(Model, DefaultConfig) {
  ModelConfig config;
  EXPECT_EQ(config.d_model, 512);
  EXPECT_EQ(config.n_layers, 8);
  EXPECT_EQ(config.n_heads, 8);
  EXPECT_EQ(config.d_ff, 2048);
  EXPECT_EQ(config.ctx_len, 128);
  EXPECT_EQ(config.vocab_size, 512);
  EXPECT_EQ(config.d_head(), 64);
}

TEST(Model, ParameterCount) {
  ModelConfig config;
  Model model{};
  model_alloc(model, config);

  EXPECT_GT(model.param_offsets.total, 25000000u);
  EXPECT_LT(model.param_offsets.total, 30000000u);

  model_free(model);
}

TEST(Model, WeightInitNonZero) {
  ModelConfig config;
  Model model{};
  model_alloc(model, config);

  int emb_size = config.vocab_size * config.d_model;
  std::vector<float> h_emb(emb_size);
  gpu_copy_to_host(h_emb.data(), model.param(model.param_offsets.tok_emb), emb_size);

  float sum = 0.0f;
  for (float v : h_emb) sum += std::abs(v);
  EXPECT_GT(sum, 0.0f);

  std::vector<float> h_bias(config.d_model);
  gpu_copy_to_host(h_bias.data(), model.param(model.param_offsets.layers[0].bq), config.d_model);

  float bias_sum = 0.0f;
  for (float v : h_bias) bias_sum += std::abs(v);
  EXPECT_NEAR(bias_sum, 0.0f, 1e-10);

  model_free(model);
}

TEST(Model, ForwardPassOutputShape) {
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

  int logit_count = seq_len * config.vocab_size;
  std::vector<float> h_logits(logit_count);
  gpu_copy_to_host(h_logits.data(), model.act(model.act_offsets.logits), logit_count);

  float sum = 0.0f;
  bool all_finite = true;
  for (float v : h_logits) {
    sum += std::abs(v);
    if (!std::isfinite(v)) all_finite = false;
  }
  EXPECT_TRUE(all_finite);
  EXPECT_GT(sum, 0.0f);

  model_free(model);
}

TEST(Model, CausalMasking) {
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

  int logit_count = 3 * config.vocab_size;

  model_forward(model, tokens1, 5);
  std::vector<float> logits1(logit_count);
  gpu_copy_to_host(logits1.data(), model.act(model.act_offsets.logits), logit_count);

  model_forward(model, tokens2, 5);
  std::vector<float> logits2(logit_count);
  gpu_copy_to_host(logits2.data(), model.act(model.act_offsets.logits), logit_count);

  for (int i = 0; i < logit_count; ++i) {
    EXPECT_NEAR(logits1[i], logits2[i], 1e-4) << "Mismatch at logit index " << i;
  }

  model_free(model);
}

TEST(Model, FusedBiasAdd) {
  int rows = 4, cols = 3;
  std::vector<float> h_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> h_bias = {0.1f, 0.2f, 0.3f};

  float* d_data = gpu_alloc(rows * cols);
  float* d_bias = gpu_alloc(cols);
  gpu_copy_to_device(d_data, h_data.data(), rows * cols);
  gpu_copy_to_device(d_bias, h_bias.data(), cols);

  gpu_add_bias(d_data, d_bias, rows, cols);

  std::vector<float> result(rows * cols);
  gpu_copy_to_host(result.data(), d_data, rows * cols);

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float expected = h_data[r * cols + c] + h_bias[c];
      EXPECT_NEAR(result[r * cols + c], expected, 1e-5)
          << "Mismatch at (" << r << "," << c << ")";
    }
  }

  gpu_free(d_data);
  gpu_free(d_bias);
}

TEST(Model, ReduceRows) {
  int rows = 4, cols = 3;
  std::vector<float> h_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  float* d_data = gpu_alloc(rows * cols);
  float* d_dst = gpu_alloc(cols);
  gpu_copy_to_device(d_data, h_data.data(), rows * cols);
  gpu_zero(d_dst, cols);

  gpu_reduce_rows(d_dst, d_data, rows, cols);

  std::vector<float> result(cols);
  gpu_copy_to_host(result.data(), d_dst, cols);

  // Column sums: {1+4+7+10, 2+5+8+11, 3+6+9+12} = {22, 26, 30}
  EXPECT_NEAR(result[0], 22.0f, 1e-5);
  EXPECT_NEAR(result[1], 26.0f, 1e-5);
  EXPECT_NEAR(result[2], 30.0f, 1e-5);

  gpu_free(d_data);
  gpu_free(d_dst);
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

  std::vector<float> h_orig(model.param_offsets.total);
  std::vector<float> h_loaded(model.param_offsets.total);
  gpu_copy_to_host(h_orig.data(), model.params, model.param_offsets.total);
  gpu_copy_to_host(h_loaded.data(), loaded.params, model.param_offsets.total);

  for (size_t i = 0; i < model.param_offsets.total; ++i) {
    EXPECT_NEAR(h_loaded[i], h_orig[i], 1e-7) << "Param mismatch at index " << i;
  }

  model_free(model);
  model_free(loaded);
  std::remove(path.c_str());
}
