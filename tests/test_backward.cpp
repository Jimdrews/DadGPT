#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "backward.hpp"
#include "gpu_ops.cuh"
#include "model.hpp"
#include "tensor_ops.hpp"

namespace {

// GPU-aware finite-difference gradient check.
// Perturbs params on GPU one at a time, runs forward+loss, compares against analytical grads.
double gpu_max_relative_error(std::function<float()> forward_loss, float* d_params,
                              const float* d_grads, int n, float h = 1e-3f) {
  // Copy analytical gradients to host
  std::vector<float> h_grads(n);
  gpu_copy_to_host(h_grads.data(), d_grads, n);

  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    // Read original param value
    float orig = 0;
    gpu_copy_to_host(&orig, d_params + i, 1);

    // Perturb +h
    float plus_val = orig + h;
    gpu_copy_to_device(d_params + i, &plus_val, 1);
    float loss_plus = forward_loss();

    // Perturb -h
    float minus_val = orig - h;
    gpu_copy_to_device(d_params + i, &minus_val, 1);
    float loss_minus = forward_loss();

    // Restore
    gpu_copy_to_device(d_params + i, &orig, 1);

    double numerical = static_cast<double>(loss_plus - loss_minus) / (2.0 * h);
    double analytical = h_grads[i];

    if (std::abs(numerical) < 1e-6 && std::abs(analytical) < 1e-3) continue;

    double denom = std::max(std::abs(numerical) + std::abs(analytical), 1e-7);
    double rel_err = std::abs(numerical - analytical) / denom;
    max_err = std::max(max_err, rel_err);
  }
  return max_err;
}

}  // namespace

TEST(Backward, MatmulGradientCheck) {
  // CPU-only test using the CPU backward functions
  float a[] = {1, 2, 3, 4, 5, 6};
  float b[] = {7, 8, 9, 10, 11, 12};
  float dc[] = {1, 0, 0, 1};
  float da[6] = {};
  float db[6] = {};

  matmul_backward(dc, a, b, da, db, 2, 3, 2);

  float da_sum = 0, db_sum = 0;
  for (float v : da) da_sum += std::abs(v);
  for (float v : db) db_sum += std::abs(v);
  EXPECT_GT(da_sum, 0);
  EXPECT_GT(db_sum, 0);
}

TEST(Backward, CrossEntropyLoss) {
  // CPU-only test
  float logits[] = {2.0f, 1.0f, 0.1f, -1.0f, 0.5f, 3.0f};
  uint16_t targets[] = {0, 2};
  float dlogits[6] = {};

  float loss = cross_entropy_loss(logits, targets, dlogits, 2, 3);

  EXPECT_GT(loss, 0.0f);
  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_LT(dlogits[0], 0.0f);
  EXPECT_LT(dlogits[5], 0.0f);
}

TEST(Backward, DebugGradientValues) {
  ModelConfig config;
  config.d_model = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 32;
  config.ctx_len = 8;
  config.vocab_size = 20;

  Model model{};
  model_alloc(model, config);

  uint16_t tokens[] = {4, 5, 6, 7};
  uint16_t targets[] = {5, 6, 7, 8};
  int seq_len = 4;

  uint16_t* d_targets = gpu_alloc_u16(seq_len);
  gpu_copy_u16_to_device(d_targets, targets, seq_len);

  model_zero_grad(model);
  model_forward(model, tokens, seq_len);

  float loss = gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                                      model.act(model.act_offsets.logits), d_targets, seq_len,
                                      config.vocab_size);

  EXPECT_GT(loss, 0) << "loss is zero";

  model_backward(model, tokens, seq_len);

  // Check various parameter gradients are non-zero (copy to host first)
  auto check_grad = [&](size_t offset, int size, const char* name) {
    std::vector<float> h_grad(size);
    gpu_copy_to_host(h_grad.data(), model.grad(offset), size);
    float sum = 0;
    for (float v : h_grad) sum += std::abs(v);
    EXPECT_GT(sum, 0) << name << " gradients are all zeros";
  };

  check_grad(model.param_offsets.lnf_w, config.d_model, "lnf_w");
  check_grad(model.param_offsets.lnf_b, config.d_model, "lnf_b");
  check_grad(model.param_offsets.layers[0].fc2_w, config.d_ff * config.d_model, "fc2_w");
  check_grad(model.param_offsets.layers[0].fc1_w, config.d_model * config.d_ff, "fc1_w");
  check_grad(model.param_offsets.layers[0].Wo, config.d_model * config.d_model, "Wo");
  check_grad(model.param_offsets.layers[0].Wq, config.d_model * config.d_model, "Wq");
  check_grad(model.param_offsets.tok_emb, config.vocab_size * config.d_model, "tok_emb");

  gpu_free(d_targets);
  model_free(model);
}

TEST(Backward, FullModelGradientCheck) {
  // Gradient check with relaxed thresholds for float32 precision.
  ModelConfig config;
  config.d_model = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.d_ff = 32;
  config.ctx_len = 8;
  config.vocab_size = 20;

  Model model{};
  model_alloc(model, config);

  uint16_t tokens[] = {4, 5, 6, 7};
  uint16_t targets[] = {5, 6, 7, 8};
  int seq_len = 4;

  uint16_t* d_targets = gpu_alloc_u16(seq_len);
  gpu_copy_u16_to_device(d_targets, targets, seq_len);

  auto compute_loss = [&]() {
    gpu_zero(model.activations, model.act_offsets.total);
    model_forward(model, tokens, seq_len);
    float* dummy_dlogits = gpu_alloc(static_cast<size_t>(seq_len) * config.vocab_size);
    float loss = gpu_cross_entropy_loss(dummy_dlogits, model.act(model.act_offsets.logits),
                                        d_targets, seq_len, config.vocab_size);
    gpu_free(dummy_dlogits);
    return loss;
  };

  // Compute analytical gradients
  model_zero_grad(model);
  gpu_zero(model.activations, model.act_offsets.total);
  model_forward(model, tokens, seq_len);
  gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                         model.act(model.act_offsets.logits), d_targets, seq_len, config.vocab_size);
  model_backward(model, tokens, seq_len);

  // Check gradients via finite differences on GPU
  double err = gpu_max_relative_error(compute_loss,
                                      model.param(model.param_offsets.layers[0].fc1_w),
                                      model.grad(model.param_offsets.layers[0].fc1_w), 32);
  EXPECT_LT(err, 0.5) << "fc1_w gradient error: " << err;

  err = gpu_max_relative_error(compute_loss, model.param(model.param_offsets.lnf_w),
                               model.grad(model.param_offsets.lnf_w), config.d_model);
  EXPECT_LT(err, 0.5) << "lnf_w gradient error: " << err;

  gpu_free(d_targets);
  model_free(model);
}
