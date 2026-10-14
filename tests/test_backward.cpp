#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "backward.hpp"
#include "gpu_ops.cuh"
#include "model.hpp"
#include "tensor_ops.hpp"

namespace {

// Small but complete model: 2 layers so residual chaining between layers is exercised.
// d_head 8 runs attention on the materialized path; d_model 64 (d_head 32) on the fused one.
ModelConfig tiny_config(int batch_size, int d_model = 16) {
  ModelConfig config;
  config.d_model = d_model;
  config.n_layers = 2;
  config.n_heads = 2;
  config.d_ff = 2 * d_model;
  config.ctx_len = 8;
  config.vocab_size = 20;
  config.batch_size = batch_size;
  return config;
}

// Forward + mean cross-entropy over all B*S positions. Writes dlogits into the model's
// logits gradient buffer so model_backward can follow.
float forward_loss(Model& model, const uint16_t* tokens, const uint16_t* d_targets, int batch,
                   int seq_len) {
  gpu_zero(model.activations, model.act_offsets.total);
  gpu_zero(model.grad_activations, model.act_offsets.total);
  model_forward(model, tokens, batch, seq_len);
  return gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                                model.act(model.act_offsets.logits), d_targets, batch * seq_len,
                                model.config.vocab_size);
}

struct ParamGroup {
  std::string name;
  size_t offset;
  size_t size;
};

std::vector<ParamGroup> all_param_groups(const Model& model) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  size_t d = c.d_model, ff = c.d_ff;
  std::vector<ParamGroup> groups = {
      {"tok_emb", po.tok_emb, c.vocab_size * d},
      {"pos_emb", po.pos_emb, c.ctx_len * d},
      {"lnf_w", po.lnf_w, d},
      {"lnf_b", po.lnf_b, d},
  };
  for (int l = 0; l < c.n_layers; ++l) {
    const auto& ly = po.layers[l];
    auto L = [&](const char* n) { return "L" + std::to_string(l) + "." + n; };
    groups.push_back({L("ln1_w"), ly.ln1_w, d});
    groups.push_back({L("ln1_b"), ly.ln1_b, d});
    groups.push_back({L("Wq"), ly.Wq, d * d});
    groups.push_back({L("bq"), ly.bq, d});
    groups.push_back({L("Wk"), ly.Wk, d * d});
    groups.push_back({L("bk"), ly.bk, d});
    groups.push_back({L("Wv"), ly.Wv, d * d});
    groups.push_back({L("bv"), ly.bv, d});
    groups.push_back({L("Wo"), ly.Wo, d * d});
    groups.push_back({L("bo"), ly.bo, d});
    groups.push_back({L("ln2_w"), ly.ln2_w, d});
    groups.push_back({L("ln2_b"), ly.ln2_b, d});
    groups.push_back({L("fc1_w"), ly.fc1_w, d * ff});
    groups.push_back({L("fc1_b"), ly.fc1_b, ff});
    groups.push_back({L("fc2_w"), ly.fc2_w, ff * d});
    groups.push_back({L("fc2_b"), ly.fc2_b, d});
  }
  return groups;
}

bool is_key_bias(const ParamGroup& g) {
  return g.name.size() >= 3 && g.name.compare(g.name.size() - 3, 3, ".bk") == 0;
}

// Finite differences on sampled elements of one parameter group. Uses Richardson
// extrapolation of two central differences, (4*D(h/2) - D(h)) / 3, which cancels the h^2
// truncation term so h can stay large enough to keep float32 loss noise small.
// Returns ||numerical - analytical|| / ||numerical|| over the sampled elements.
double fd_relative_error(Model& model, const ParamGroup& g, const std::vector<float>& analytical,
                         const std::function<float()>& loss_fn, int samples, std::mt19937& rng,
                         float h = 1e-2f) {
  std::vector<size_t> idx(g.size);
  for (size_t i = 0; i < g.size; ++i) idx[i] = i;
  std::shuffle(idx.begin(), idx.end(), rng);
  idx.resize(std::min<size_t>(idx.size(), samples));

  double diff_sq = 0.0, ref_sq = 0.0;
  for (size_t i : idx) {
    float* p = model.param(g.offset + i);
    float orig = 0.0f;
    gpu_copy_to_host(&orig, p, 1);

    auto central = [&](float step) {
      float plus = orig + step, minus = orig - step;
      gpu_copy_to_device(p, &plus, 1);
      double lp = loss_fn();
      gpu_copy_to_device(p, &minus, 1);
      double lm = loss_fn();
      gpu_copy_to_device(p, &orig, 1);
      return (lp - lm) / (2.0 * static_cast<double>(step));
    };
    double numerical = (4.0 * central(h / 2) - central(h)) / 3.0;
    double a = analytical[g.offset + i];
    diff_sq += (numerical - a) * (numerical - a);
    ref_sq += numerical * numerical;
  }
  return std::sqrt(diff_sq / std::max(ref_sq, 1e-20));
}

}  // namespace

TEST(Backward, MatmulGradientCheck) {
  // CPU reference: C = A(2x3) x B(3x2). dA = dC x B^T, dB = A^T x dC.
  float a[] = {1, 2, 3, 4, 5, 6};
  float b[] = {7, 8, 9, 10, 11, 12};
  float dc[] = {1, 0, 0, 1};
  float da[6] = {};
  float db[6] = {};

  matmul_backward(dc, a, b, da, db, 2, 3, 2);

  // dA[i][k] = sum_j dC[i][j] * B[k][j]
  float expected_da[] = {7, 9, 11, 8, 10, 12};
  // dB[k][j] = sum_i A[i][k] * dC[i][j]
  float expected_db[] = {1, 4, 2, 5, 3, 6};
  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(da[i], expected_da[i]) << "da[" << i << "]";
    EXPECT_FLOAT_EQ(db[i], expected_db[i]) << "db[" << i << "]";
  }
}

TEST(Backward, CrossEntropyLoss) {
  // CPU reference against a hand-computed softmax
  float logits[] = {2.0f, 1.0f, 0.1f, -1.0f, 0.5f, 3.0f};
  uint16_t targets[] = {0, 2};
  float dlogits[6] = {};

  float loss = cross_entropy_loss(logits, targets, dlogits, 2, 3);

  auto row_loss = [](const float* z, int t) {
    double m = *std::max_element(z, z + 3), s = 0;
    for (int i = 0; i < 3; ++i) s += std::exp(z[i] - m);
    return -(z[t] - m - std::log(s));
  };
  double expected = 0.5 * (row_loss(logits, 0) + row_loss(logits + 3, 2));
  EXPECT_NEAR(loss, expected, 1e-5);
  EXPECT_LT(dlogits[0], 0.0f);
  EXPECT_LT(dlogits[5], 0.0f);
}

TEST(Backward, GpuCrossEntropyMatchesReference) {
  // Wide rows exercise the multi-warp reductions in the GPU kernel
  const int rows = 6, vocab = 1000;
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0f, 4.0f);
  std::vector<float> logits(rows * vocab);
  for (float& v : logits) v = dist(rng);
  std::vector<uint16_t> targets = {0, 17, 999, 500, 3, 250};

  float* d_logits = gpu_alloc(logits.size());
  float* d_dlogits = gpu_alloc(logits.size());
  uint16_t* d_targets = gpu_alloc_u16(rows);
  gpu_copy_to_device(d_logits, logits.data(), logits.size());
  gpu_copy_u16_to_device(d_targets, targets.data(), rows);

  float loss = gpu_cross_entropy_loss(d_dlogits, d_logits, d_targets, rows, vocab);
  std::vector<float> dlogits(logits.size());
  gpu_copy_to_host(dlogits.data(), d_dlogits, dlogits.size());

  double expected_loss = 0.0;
  for (int r = 0; r < rows; ++r) {
    const float* z = logits.data() + r * vocab;
    double m = *std::max_element(z, z + vocab), s = 0;
    for (int i = 0; i < vocab; ++i) s += std::exp(z[i] - m);
    double lse = m + std::log(s);
    expected_loss += lse - z[targets[r]];
    for (int i = 0; i < vocab; ++i) {
      double p = std::exp(z[i] - lse);
      double g = (p - (i == targets[r] ? 1.0 : 0.0)) / rows;
      ASSERT_NEAR(dlogits[r * vocab + i], g, 1e-6) << "row " << r << " col " << i;
    }
  }
  EXPECT_NEAR(loss, expected_loss / rows, 1e-4);

  gpu_free(d_logits);
  gpu_free(d_dlogits);
  gpu_free(d_targets);
}

TEST(Backward, GpuSoftmaxRowsSumToOne) {
  // 128 columns -> 4 warps per row, the configuration that exposed the max/sum sdata race
  const int rows = 512, cols = 128;
  std::mt19937 rng(3);
  std::normal_distribution<float> dist(0.0f, 10.0f);
  std::vector<float> in(rows * cols);
  for (float& v : in) v = dist(rng);

  float* d_in = gpu_alloc(in.size());
  float* d_out = gpu_alloc(in.size());
  gpu_copy_to_device(d_in, in.data(), in.size());
  gpu_softmax(d_out, d_in, rows, cols);
  std::vector<float> out(in.size());
  gpu_copy_to_host(out.data(), d_out, out.size());

  for (int r = 0; r < rows; ++r) {
    const float* z = in.data() + r * cols;
    double m = *std::max_element(z, z + cols), s = 0;
    for (int c = 0; c < cols; ++c) s += std::exp(z[c] - m);
    for (int c = 0; c < cols; ++c) {
      ASSERT_NEAR(out[r * cols + c], std::exp(z[c] - m) / s, 1e-5) << "row " << r;
    }
  }
  gpu_free(d_in);
  gpu_free(d_out);
}

// Finite differences over every parameter group (attention, LayerNorms, MLP, biases,
// tied token embedding, positional embedding) with batch 2 and seq_len < ctx_len.
// With dropout, every forward re-seeds the RNG so all passes see identical masks and the
// loss stays a deterministic, smooth function of the parameters.
void full_model_gradient_check(float dropout_rate, int d_model = 16,
                               AttentionImpl attention = AttentionImpl::Auto) {
  ModelConfig config = tiny_config(2, d_model);
  Model model{};
  model_alloc(model, config);
  model.attention = attention;
  model.training = dropout_rate > 0.0f;
  model.dropout_rate = dropout_rate;

  const int batch = 2, seq_len = 6;
  // Repeated tokens exercise the embedding scatter-add
  uint16_t tokens[] = {4, 5, 6, 4, 7, 8, 9, 9, 10, 11, 4, 12};
  uint16_t targets[] = {5, 6, 4, 7, 8, 1, 9, 10, 11, 4, 12, 13};
  uint16_t* d_targets = gpu_alloc_u16(batch * seq_len);
  gpu_copy_u16_to_device(d_targets, targets, batch * seq_len);

  // Training init (N(0, 0.02)) leaves attention near-uniform and the residual stream tiny,
  // so some gradients sit at float32 noise level. Gradient checks want O(1)-conditioned
  // random parameters instead: everything ~N(0, 0.3), LayerNorm weights around 1.
  {
    std::mt19937 prng(77);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> h_params(model.param_offsets.total);
    for (float& p : h_params) p = dist(prng);
    auto ones_ish = [&](size_t offset) {
      for (int i = 0; i < config.d_model; ++i) h_params[offset + i] += 1.0f;
    };
    ones_ish(model.param_offsets.lnf_w);
    for (const auto& ly : model.param_offsets.layers) {
      ones_ish(ly.ln1_w);
      ones_ish(ly.ln2_w);
    }
    gpu_copy_to_device(model.params, h_params.data(), h_params.size());
  }

  auto loss_fn = [&]() {
    gpu_dropout_init(2024, 0);
    return forward_loss(model, tokens, d_targets, batch, seq_len);
  };

  model_zero_grad(model);
  loss_fn();
  model_backward(model, tokens, batch, seq_len);
  std::vector<float> analytical(model.param_offsets.total);
  gpu_copy_to_host(analytical.data(), model.gradients, analytical.size());

  std::mt19937 rng(1234);
  for (const auto& g : all_param_groups(model)) {
    if (is_key_bias(g)) {
      // Softmax is shift-invariant per row and bk shifts every score in a row by q.bk,
      // so the exact gradient is zero: check that instead of a meaningless relative error.
      for (size_t i = g.offset; i < g.offset + g.size; ++i) {
        EXPECT_NEAR(analytical[i], 0.0f, 1e-5f) << g.name << "[" << i - g.offset << "]";
      }
      continue;
    }
    // h = 3e-2 from a sweep over {3e-3 .. 1e-1}: worst group error 0.26% (no dropout) and
    // 0.08% (dropout 0.3). A genuinely wrong gradient term shows up as >= 30% error.
    double err = fd_relative_error(model, g, analytical, loss_fn, 24, rng, 3e-2f);
    EXPECT_LT(err, 1e-2) << g.name << " gradient relative error " << err;
  }

  gpu_free(d_targets);
  model_free(model);
}

TEST(Backward, FullModelGradientCheck) { full_model_gradient_check(0.0f); }

// d_head 32 with dropout off: attention goes through the fused (flash) kernels
TEST(Backward, FullModelGradientCheckFusedAttention) {
  ASSERT_TRUE(gpu_flash_attention_supported(32));
  full_model_gradient_check(0.0f, 64, AttentionImpl::Fused);
}

// Covers the dropout backward at all four sites, including attention probabilities.
TEST(Backward, FullModelGradientCheckWithDropout) { full_model_gradient_check(0.3f); }

TEST(Backward, AccumulatedMicroBatchesMatchFullBatch) {
  // Two batch-1 backward passes, summed and averaged, must equal one batch-2 pass for
  // *every* gradient. This is what gradient accumulation relies on; it caught the tied
  // tok_emb gradient being overwritten (beta=0) instead of accumulated.
  ModelConfig config = tiny_config(2);
  Model model{};
  model_alloc(model, config);
  model.training = false;

  const int seq_len = 4;
  uint16_t tokens[] = {4, 5, 6, 7, 8, 9, 10, 11};
  uint16_t targets[] = {5, 6, 7, 8, 9, 10, 11, 12};
  uint16_t* d_targets = gpu_alloc_u16(8);
  size_t n = model.param_offsets.total;

  // Accumulated path: no zero_grad between micro-batches, exactly as train.cpp does
  model_zero_grad(model);
  for (int b = 0; b < 2; ++b) {
    gpu_copy_u16_to_device(d_targets, targets + b * seq_len, seq_len);
    forward_loss(model, tokens + b * seq_len, d_targets, 1, seq_len);
    model_backward(model, tokens + b * seq_len, 1, seq_len);
  }
  gpu_gradient_scale(model.gradients, n, 0.5f);
  std::vector<float> accumulated(n);
  gpu_copy_to_host(accumulated.data(), model.gradients, n);

  // Full-batch path
  model_zero_grad(model);
  gpu_copy_u16_to_device(d_targets, targets, 8);
  forward_loss(model, tokens, d_targets, 2, seq_len);
  model_backward(model, tokens, 2, seq_len);
  std::vector<float> full(n);
  gpu_copy_to_host(full.data(), model.gradients, n);

  for (const auto& g : all_param_groups(model)) {
    if (is_key_bias(g)) continue;  // exact gradient is zero; only float noise to compare
    double diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = g.offset; i < g.offset + g.size; ++i) {
      diff_sq += (accumulated[i] - full[i]) * static_cast<double>(accumulated[i] - full[i]);
      ref_sq += static_cast<double>(full[i]) * full[i];
    }
    double rel = std::sqrt(diff_sq / std::max(ref_sq, 1e-20));
    EXPECT_LT(rel, 1e-4) << g.name << " accumulated vs full-batch relative error " << rel;
  }

  gpu_free(d_targets);
  model_free(model);
}

TEST(Backward, DropoutGradientsAreConsistent) {
  // With dropout on, a single forward/backward must still produce finite, nonzero grads,
  // and running the same masks twice (same seed) must give identical losses.
  ModelConfig config = tiny_config(1);
  Model model{};
  model_alloc(model, config);
  model.training = true;
  model.dropout_rate = 0.5f;

  uint16_t tokens[] = {4, 5, 6, 7};
  uint16_t targets[] = {5, 6, 7, 8};
  uint16_t* d_targets = gpu_alloc_u16(4);
  gpu_copy_u16_to_device(d_targets, targets, 4);

  gpu_dropout_init(99, 0);
  float loss1 = forward_loss(model, tokens, d_targets, 1, 4);
  gpu_dropout_init(99, 0);
  float loss2 = forward_loss(model, tokens, d_targets, 1, 4);
  EXPECT_FLOAT_EQ(loss1, loss2);

  model_zero_grad(model);
  forward_loss(model, tokens, d_targets, 1, 4);
  model_backward(model, tokens, 1, 4);
  std::vector<float> grads(model.param_offsets.total);
  gpu_copy_to_host(grads.data(), model.gradients, grads.size());
  double sum = 0.0;
  for (float g : grads) {
    ASSERT_TRUE(std::isfinite(g));
    sum += std::abs(g);
  }
  EXPECT_GT(sum, 0.0);

  gpu_free(d_targets);
  model_free(model);
}

TEST(Backward, OverfitsSingleBatch) {
  // End-to-end smoke test of forward + backward + AdamW: a tiny model must be able to
  // memorize one short sequence.
  ModelConfig config = tiny_config(1);
  Model model{};
  model_alloc(model, config);
  model.training = false;

  uint16_t tokens[] = {4, 9, 13, 2, 7, 16, 5, 11};
  uint16_t targets[] = {9, 13, 2, 7, 16, 5, 11, 3};
  uint16_t* d_targets = gpu_alloc_u16(8);
  gpu_copy_u16_to_device(d_targets, targets, 8);

  size_t n = model.param_offsets.total;
  float* m = gpu_alloc(n);
  float* v = gpu_alloc(n);
  gpu_zero(m, n);
  gpu_zero(v, n);

  float first = 0.0f, loss = 0.0f;
  for (int step = 1; step <= 300; ++step) {
    model_zero_grad(model);
    loss = forward_loss(model, tokens, d_targets, 1, 8);
    if (step == 1) first = loss;
    model_backward(model, tokens, 1, 8);
    gpu_adamw_step(model.params, model.gradients, m, v, model.decay_flags, n, 3e-3f, 0.9f, 0.95f,
                   1e-8f, 0.0f, step);
  }
  EXPECT_GT(first, 2.0f);
  EXPECT_LT(loss, 0.05f) << "loss went " << first << " -> " << loss;

  gpu_free(m);
  gpu_free(v);
  gpu_free(d_targets);
  model_free(model);
}

TEST(Backward, FusedAttentionMatchesMaterialized) {
  // Several 32-row tiles with a ragged last tile (S = 70), batch 2, d_head 64: logits and every
  // parameter gradient must agree between the fused kernels and the materialized path.
  ModelConfig config = tiny_config(2, 128);
  config.ctx_len = 80;
  const int batch = 2, seq_len = 70, rows = batch * seq_len;
  std::vector<uint16_t> tokens(rows), targets(rows);
  for (int i = 0; i < rows; ++i) {
    tokens[i] = static_cast<uint16_t>((i * 7 + 3) % config.vocab_size);
    targets[i] = static_cast<uint16_t>((i * 11 + 5) % config.vocab_size);
  }
  uint16_t* d_targets = gpu_alloc_u16(rows);
  gpu_copy_u16_to_device(d_targets, targets.data(), rows);

  auto run = [&](bool materialized, std::vector<float>& logits, std::vector<float>& grads) {
    Model model{};
    model_alloc(model, config);  // deterministic init: both models get identical weights
    model.training = false;
    model.attention = materialized ? AttentionImpl::Materialized : AttentionImpl::Fused;
    model_zero_grad(model);
    forward_loss(model, tokens.data(), d_targets, batch, seq_len);
    logits.resize(static_cast<size_t>(rows) * config.vocab_size);
    gpu_copy_to_host(logits.data(), model.act(model.act_offsets.logits), logits.size());
    model_backward(model, tokens.data(), batch, seq_len);
    grads.resize(model.param_offsets.total);
    gpu_copy_to_host(grads.data(), model.gradients, grads.size());
    auto groups = all_param_groups(model);
    model_free(model);
    return groups;
  };
  std::vector<float> logits_m, grads_m, logits_f, grads_f;
  run(true, logits_m, grads_m);
  auto groups = run(false, logits_f, grads_f);

  for (size_t i = 0; i < logits_m.size(); ++i) {
    ASSERT_NEAR(logits_f[i], logits_m[i], 1e-4) << "logit " << i;
  }
  for (const auto& g : groups) {
    if (is_key_bias(g)) continue;  // exact gradient is zero; only float noise to compare
    double diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = g.offset; i < g.offset + g.size; ++i) {
      diff_sq += static_cast<double>(grads_f[i] - grads_m[i]) * (grads_f[i] - grads_m[i]);
      ref_sq += static_cast<double>(grads_m[i]) * grads_m[i];
    }
    double rel = std::sqrt(diff_sq / std::max(ref_sq, 1e-20));
    EXPECT_LT(rel, 1e-4) << g.name << " fused vs materialized relative error " << rel;
  }
  gpu_free(d_targets);
}
