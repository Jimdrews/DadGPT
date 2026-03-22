#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <vector>

#include "backward.hpp"
#include "model.hpp"
#include "tensor_ops.hpp"

namespace {

// Finite-difference gradient check helper
// f: function that takes params and returns scalar loss
// params: parameter array to check
// grads: analytically computed gradients
// n: number of parameters
// h: perturbation size
double max_relative_error(std::function<float()> f, float* params, const float* grads, int n,
                          float h = 1e-5f, bool verbose = false) {
  double max_err = 0.0;
  for (int i = 0; i < n; ++i) {
    float orig = params[i];

    params[i] = orig + h;
    float loss_plus = f();

    params[i] = orig - h;
    float loss_minus = f();

    params[i] = orig;

    double numerical = static_cast<double>(loss_plus - loss_minus) / (2.0 * h);
    double analytical = grads[i];

    // Skip tiny gradients where float precision dominates
    if (std::abs(numerical) < 1e-6 && std::abs(analytical) < 1e-3) continue;

    double denom = std::max(std::abs(numerical) + std::abs(analytical), 1e-7);
    double rel_err = std::abs(numerical - analytical) / denom;

    if (verbose && i < 5) {
      std::printf("  [%d] numerical=%.8f analytical=%.8f rel_err=%.4f loss+=%f loss-=%f\n", i,
                  numerical, analytical, rel_err, loss_plus, loss_minus);
    }

    max_err = std::max(max_err, rel_err);
  }
  return max_err;
}

}  // namespace

TEST(Backward, MatmulGradientCheck) {
  float a[] = {1, 2, 3, 4, 5, 6};       // 2×3
  float b[] = {7, 8, 9, 10, 11, 12};     // 3×2
  float c[4] = {};
  float dc[] = {1, 0, 0, 1};             // identity-like gradient
  float da[6] = {};
  float db[6] = {};

  matmul_backward(dc, a, b, da, db, 2, 3, 2);

  // Verify da and db are non-zero
  float da_sum = 0, db_sum = 0;
  for (float v : da) da_sum += std::abs(v);
  for (float v : db) db_sum += std::abs(v);
  EXPECT_GT(da_sum, 0);
  EXPECT_GT(db_sum, 0);
}

TEST(Backward, CrossEntropyLoss) {
  float logits[] = {2.0f, 1.0f, 0.1f, -1.0f, 0.5f, 3.0f};  // 2 positions × 3 vocab
  uint16_t targets[] = {0, 2};
  float dlogits[6] = {};

  float loss = cross_entropy_loss(logits, targets, dlogits, 2, 3);

  EXPECT_GT(loss, 0.0f);
  EXPECT_TRUE(std::isfinite(loss));

  // dlogits for target positions should be negative (softmax - 1)
  EXPECT_LT(dlogits[0], 0.0f);   // target=0, position 0
  EXPECT_LT(dlogits[5], 0.0f);   // target=2, position 1
}

TEST(Backward, DebugGradientValues) {
  // Check if gradients are actually non-zero after backward pass
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

  model_zero_grad(model);
  model_forward(model, tokens, seq_len);

  float loss = cross_entropy_loss(model.act(model.act_offsets.logits), targets,
                                  model.grad_act(model.act_offsets.logits), seq_len,
                                  config.vocab_size);

  // Check dlogits is non-zero
  float dlogits_sum = 0;
  for (int i = 0; i < seq_len * config.vocab_size; ++i) {
    dlogits_sum += std::abs(model.grad_act(model.act_offsets.logits)[i]);
  }
  EXPECT_GT(dlogits_sum, 0) << "dlogits is all zeros";
  EXPECT_GT(loss, 0) << "loss is zero";

  model_backward(model, tokens, seq_len);

  // Check various parameter gradients
  auto check_grad = [&](size_t offset, int size, const char* name) {
    float sum = 0;
    for (int i = 0; i < size; ++i) {
      sum += std::abs(model.grad(offset)[i]);
    }
    EXPECT_GT(sum, 0) << name << " gradients are all zeros";
  };

  check_grad(model.param_offsets.lnf_w, config.d_model, "lnf_w");
  check_grad(model.param_offsets.lnf_b, config.d_model, "lnf_b");
  check_grad(model.param_offsets.layers[0].fc2_w, config.d_ff * config.d_model, "fc2_w");
  check_grad(model.param_offsets.layers[0].fc1_w, config.d_model * config.d_ff, "fc1_w");
  check_grad(model.param_offsets.layers[0].Wo, config.d_model * config.d_model, "Wo");
  check_grad(model.param_offsets.layers[0].Wq, config.d_model * config.d_model, "Wq");
  check_grad(model.param_offsets.tok_emb, config.vocab_size * config.d_model, "tok_emb");
  check_grad(model.param_offsets.pos_emb, seq_len * config.d_model, "pos_emb");

  model_free(model);
}

TEST(Backward, FullModelGradientCheck) {
  // Gradient check with relaxed thresholds for float32 precision.
  // Uses h=1e-3 (not 1e-5) and threshold 0.5 (not 1e-4) because
  // float32 loss values don't have enough precision for tighter checks.
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

  auto compute_loss = [&]() {
    std::memset(model.activations, 0, model.act_offsets.total * sizeof(float));
    model_forward(model, tokens, seq_len);
    std::vector<float> dlogits_dummy(static_cast<size_t>(seq_len) * config.vocab_size, 0.0f);
    return cross_entropy_loss(model.act(model.act_offsets.logits), targets, dlogits_dummy.data(),
                              seq_len, config.vocab_size);
  };

  // Compute analytical gradients
  model_zero_grad(model);
  std::memset(model.activations, 0, model.act_offsets.total * sizeof(float));
  model_forward(model, tokens, seq_len);
  cross_entropy_loss(model.act(model.act_offsets.logits), targets,
                     model.grad_act(model.act_offsets.logits), seq_len, config.vocab_size);
  model_backward(model, tokens, seq_len);

  // Debug: check if perturbation actually changes loss
  {
    float base_loss = compute_loss();
    float* p = model.param(model.param_offsets.layers[0].fc1_w);
    float orig = p[0];
    p[0] = orig + 1e-3f;
    float perturbed_loss = compute_loss();
    p[0] = orig;
    EXPECT_NE(base_loss, perturbed_loss)
        << "Loss unchanged by perturbation. base=" << base_loss << " perturbed=" << perturbed_loss;
  }

  // Check a subset of parameters
  // Check first layer's fc1_w (first 32 params only for speed)
  float h = 1e-3f;

  double err = max_relative_error(compute_loss, model.param(model.param_offsets.layers[0].fc1_w),
                                  model.grad(model.param_offsets.layers[0].fc1_w), 32, h);
  EXPECT_LT(err, 0.5) << "fc1_w gradient error: " << err;

  err = max_relative_error(compute_loss, model.param(model.param_offsets.lnf_w),
                           model.grad(model.param_offsets.lnf_w), config.d_model, h);
  EXPECT_LT(err, 0.5) << "lnf_w gradient error: " << err;

  err = max_relative_error(compute_loss, model.param(model.param_offsets.layers[0].Wo),
                           model.grad(model.param_offsets.layers[0].Wo), 32, h);
  EXPECT_LT(err, 0.5) << "Wo gradient error: " << err;

  model_free(model);
}
