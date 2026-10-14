#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <numeric>
#include <vector>

#include "gpu_ops.cuh"
#include "model.hpp"
#include "optimizer.hpp"
#include "test_util.hpp"

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
  model.training = false;

  EXPECT_GT(model.param_offsets.total, 25000000u);
  EXPECT_LT(model.param_offsets.total, 30000000u);

  model_free(model);
}

TEST(Model, WeightInitNonZero) {
  ModelConfig config;
  Model model{};
  model_alloc(model, config);
  model.training = false;

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
  model.training = false;

  uint16_t tokens[] = {4, 5, 6, 7, 8};
  int seq_len = 5;

  model_forward(model, tokens, 1, seq_len);

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
  model.training = false;

  uint16_t tokens1[] = {4, 5, 6, 10, 20};
  uint16_t tokens2[] = {4, 5, 6, 30, 40};

  int logit_count = 3 * config.vocab_size;

  model_forward(model, tokens1, 1, 5);
  std::vector<float> logits1(logit_count);
  gpu_copy_to_host(logits1.data(), model.act(model.act_offsets.logits), logit_count);

  model_forward(model, tokens2, 1, 5);
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
  model.training = false;

  std::string path = test_util::temp_path("model.bin");
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

TEST(Model, KVCacheMatchesFullForward) {
  // Incremental decoding through the KV cache must reproduce the full causal forward pass
  ModelConfig config;
  config.d_model = 32;
  config.n_layers = 2;
  config.n_heads = 4;
  config.d_ff = 64;
  config.ctx_len = 16;
  config.vocab_size = 100;

  Model model{};
  model_alloc(model, config);
  model.training = false;

  std::vector<uint16_t> tokens = {4, 17, 33, 4, 90, 12, 55, 61, 2, 8, 8, 70, 1, 99, 5, 40};
  int seq_len = static_cast<int>(tokens.size());  // the full context, to cover the last slot

  model_forward(model, tokens.data(), 1, seq_len);
  std::vector<float> full(static_cast<size_t>(seq_len) * config.vocab_size);
  gpu_copy_to_host(full.data(), model.act(model.act_offsets.logits), full.size());

  KVCache cache;
  kv_cache_alloc(cache, config);
  std::vector<float> step(config.vocab_size);
  for (int pos = 0; pos < seq_len; ++pos) {
    model_forward_cached(model, cache, tokens[pos], pos);
    gpu_copy_to_host(step.data(), model.act(model.act_offsets.logits), step.size());
    for (int v = 0; v < config.vocab_size; ++v) {
      ASSERT_NEAR(step[v], full[static_cast<size_t>(pos) * config.vocab_size + v], 1e-4)
          << "pos " << pos << " vocab " << v;
    }
  }
  EXPECT_EQ(cache.length, seq_len);

  kv_cache_free(cache);
  model_free(model);
}

TEST(Model, RejectsShapesBeyondBuffers) {
  ModelConfig config;
  config.d_model = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 32;
  config.ctx_len = 8;
  config.vocab_size = 20;
  config.batch_size = 2;

  Model model{};
  model_alloc(model, config);
  std::vector<uint16_t> tokens(3 * 9, 4);

  EXPECT_THROW(model_forward(model, tokens.data(), 1, 9), std::out_of_range);  // seq > ctx
  EXPECT_THROW(model_forward(model, tokens.data(), 3, 4), std::out_of_range);  // batch > alloc
  EXPECT_NO_THROW(model_forward(model, tokens.data(), 2, 8));

  KVCache cache;
  kv_cache_alloc(cache, config);
  EXPECT_THROW(model_forward_cached(model, cache, 4, config.ctx_len), std::out_of_range);
  kv_cache_free(cache);
  model_free(model);
}

TEST(Model, SaveLoadWithOptimizerRestoresState) {
  ModelConfig config;
  config.d_model = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 32;
  config.ctx_len = 8;
  config.vocab_size = 20;
  config.batch_size = 4;

  Model model{};
  model_alloc(model, config);
  size_t n = model.param_offsets.total;

  Adam opt;
  opt.step = 1234;
  opt.m = gpu_alloc(n);
  opt.v = gpu_alloc(n);
  std::vector<float> h_m(n), h_v(n);
  for (size_t i = 0; i < n; ++i) {
    h_m[i] = 0.001f * static_cast<float>(i % 97);
    h_v[i] = 0.002f * static_cast<float>(i % 89);
  }
  gpu_copy_to_device(opt.m, h_m.data(), n);
  gpu_copy_to_device(opt.v, h_v.data(), n);

  std::string path = test_util::temp_path("ckpt_opt.bin");
  model_save_with_optimizer(model, opt, path);

  // Resume with a different batch size: buffers must follow the override
  Model loaded{};
  Adam opt2;
  model_load_with_optimizer(loaded, opt2, path, /*batch_size=*/7);
  EXPECT_EQ(loaded.config.batch_size, 7);
  EXPECT_EQ(opt2.step, 1234);
  ASSERT_NE(opt2.m, nullptr);
  ASSERT_NE(opt2.v, nullptr);

  std::vector<float> got(n);
  gpu_copy_to_host(got.data(), opt2.m, n);
  EXPECT_EQ(got, h_m);
  gpu_copy_to_host(got.data(), opt2.v, n);
  EXPECT_EQ(got, h_v);

  std::vector<float> orig(n);
  gpu_copy_to_host(orig.data(), model.params, n);
  gpu_copy_to_host(got.data(), loaded.params, n);
  EXPECT_EQ(got, orig);

  // A batch of 7 must now fit
  std::vector<uint16_t> tokens(7 * 8, 4);
  EXPECT_NO_THROW(model_forward(loaded, tokens.data(), 7, 8));

  gpu_free(opt.m);
  gpu_free(opt.v);
  gpu_free(opt2.m);
  gpu_free(opt2.v);
  model_free(model);
  model_free(loaded);
  std::remove(path.c_str());
}

TEST(Model, LoadRejectsTruncatedCheckpoint) {
  ModelConfig config;
  config.d_model = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 32;
  config.ctx_len = 8;
  config.vocab_size = 20;

  Model model{};
  model_alloc(model, config);
  std::string path = test_util::temp_path("ckpt_trunc.bin");
  model_save(model, path);
  model_free(model);

  std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
  Model loaded{};
  EXPECT_THROW(model_load(loaded, path), std::runtime_error);
  model_free(loaded);
  std::remove(path.c_str());
}

TEST(Model, PresetsMatchGpt2Shapes) {
  ModelConfig c = model_preset("gpt2-124m");
  c.vocab_size = 50257;
  EXPECT_EQ(c.d_head(), 64);
  // GPT-2 small has 124,439,808 parameters (tied lm_head, no separate output matrix)
  Model model{};
  model_alloc(model, c);
  EXPECT_EQ(model.param_offsets.total, 124439808u);
  model_free(model);
  EXPECT_THROW(model_preset("gpt5"), std::invalid_argument);
}

TEST(Model, PrefillThenCachedMatchesFullForward) {
  // Batched prefill of a prompt, then incremental decoding, must match the full forward pass
  ModelConfig config;
  config.d_model = 64;
  config.n_layers = 2;
  config.n_heads = 2;
  config.d_ff = 128;
  config.ctx_len = 24;
  config.vocab_size = 100;

  Model model{};
  model_alloc(model, config);
  model.training = false;
  std::vector<uint16_t> tokens = {4, 17, 33, 4, 90, 12, 55, 61, 2, 8, 8, 70, 1, 99, 5, 40, 3, 3, 9};
  int n = static_cast<int>(tokens.size()), prompt = 11;

  model_forward(model, tokens.data(), 1, n);
  std::vector<float> full(static_cast<size_t>(n) * config.vocab_size);
  gpu_copy_to_host(full.data(), model.act(model.act_offsets.logits), full.size());

  KVCache cache;
  kv_cache_alloc(cache, config);
  std::vector<float> step(config.vocab_size);
  auto check_row = [&](int pos) {
    gpu_copy_to_host(step.data(), model.act(model.act_offsets.logits), step.size());
    for (int v = 0; v < config.vocab_size; ++v) {
      ASSERT_NEAR(step[v], full[static_cast<size_t>(pos) * config.vocab_size + v], 1e-4)
          << "pos " << pos << " vocab " << v;
    }
  };
  model_prefill(model, cache, tokens.data(), prompt);
  EXPECT_EQ(cache.length, prompt);
  check_row(prompt - 1);
  for (int pos = prompt; pos < n; ++pos) {
    model_forward_cached(model, cache, tokens[pos], pos);
    check_row(pos);
  }
  kv_cache_free(cache);
  model_free(model);
}
