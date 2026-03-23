#include "model.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "gpu_ops.cuh"

namespace {

ParameterOffsets compute_param_offsets(const ModelConfig& c) {
  ParameterOffsets p{};
  size_t offset = 0;

  auto alloc = [&](size_t count) {
    size_t o = offset;
    offset += count;
    return o;
  };

  p.tok_emb = alloc(static_cast<size_t>(c.vocab_size) * c.d_model);
  p.pos_emb = alloc(static_cast<size_t>(c.ctx_len) * c.d_model);

  p.layers.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) {
    auto& ly = p.layers[l];
    ly.ln1_w = alloc(c.d_model);
    ly.ln1_b = alloc(c.d_model);
    ly.Wq = alloc(static_cast<size_t>(c.d_model) * c.d_model);
    ly.bq = alloc(c.d_model);
    ly.Wk = alloc(static_cast<size_t>(c.d_model) * c.d_model);
    ly.bk = alloc(c.d_model);
    ly.Wv = alloc(static_cast<size_t>(c.d_model) * c.d_model);
    ly.bv = alloc(c.d_model);
    ly.Wo = alloc(static_cast<size_t>(c.d_model) * c.d_model);
    ly.bo = alloc(c.d_model);
    ly.ln2_w = alloc(c.d_model);
    ly.ln2_b = alloc(c.d_model);
    ly.fc1_w = alloc(static_cast<size_t>(c.d_model) * c.d_ff);
    ly.fc1_b = alloc(c.d_ff);
    ly.fc2_w = alloc(static_cast<size_t>(c.d_ff) * c.d_model);
    ly.fc2_b = alloc(c.d_model);
  }

  p.lnf_w = alloc(c.d_model);
  p.lnf_b = alloc(c.d_model);

  p.total = offset;
  return p;
}

ActivationOffsets compute_act_offsets(const ModelConfig& c) {
  ActivationOffsets a{};
  size_t offset = 0;

  auto alloc = [&](size_t count) {
    size_t o = offset;
    offset += count;
    return o;
  };

  size_t sd = static_cast<size_t>(c.ctx_len) * c.d_model;

  a.embedded = alloc(sd);

  a.layers.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) {
    auto& ly = a.layers[l];
    ly.ln1_out = alloc(sd);
    ly.q = alloc(sd);
    ly.k = alloc(sd);
    ly.v = alloc(sd);
    ly.q_heads = alloc(sd);  // [n_heads × seq_len × d_head]
    ly.k_heads = alloc(sd);
    ly.v_heads = alloc(sd);
    ly.attn_scores = alloc(static_cast<size_t>(c.n_heads) * c.ctx_len * c.ctx_len);
    ly.attn_out_heads = alloc(sd);  // [n_heads × seq_len × d_head]
    ly.attn_out = alloc(sd);
    ly.attn_proj = alloc(sd);
    ly.residual1 = alloc(sd);
    ly.ln2_out = alloc(sd);
    ly.fc1_out = alloc(static_cast<size_t>(c.ctx_len) * c.d_ff);
    ly.gelu_out = alloc(static_cast<size_t>(c.ctx_len) * c.d_ff);
    ly.fc2_out = alloc(sd);
    ly.residual2 = alloc(sd);
  }

  a.lnf_out = alloc(sd);
  a.logits = alloc(static_cast<size_t>(c.ctx_len) * c.vocab_size);

  a.total = offset;
  return a;
}

// Generate weights on CPU, then copy to GPU
void init_weights(Model& model) {
  const auto& c = model.config;
  size_t total = model.param_offsets.total;
  std::vector<float> h_params(total, 0.0f);

  std::mt19937 rng(42);

  auto xavier_init = [&](size_t offset, int fan_in, int fan_out, int count) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (int i = 0; i < count; ++i) h_params[offset + i] = dist(rng);
  };

  auto normal_init = [&](size_t offset, int count, float std_dev) {
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (int i = 0; i < count; ++i) h_params[offset + i] = dist(rng);
  };

  const auto& po = model.param_offsets;

  normal_init(po.tok_emb, c.vocab_size * c.d_model, 0.02f);
  normal_init(po.pos_emb, c.ctx_len * c.d_model, 0.02f);

  for (int l = 0; l < c.n_layers; ++l) {
    const auto& ly = po.layers[l];

    for (int i = 0; i < c.d_model; ++i) {
      h_params[ly.ln1_w + i] = 1.0f;
      h_params[ly.ln2_w + i] = 1.0f;
    }

    xavier_init(ly.Wq, c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(ly.Wk, c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(ly.Wv, c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(ly.Wo, c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(ly.fc1_w, c.d_model, c.d_ff, c.d_model * c.d_ff);
    xavier_init(ly.fc2_w, c.d_ff, c.d_model, c.d_ff * c.d_model);
  }

  for (int i = 0; i < c.d_model; ++i) {
    h_params[po.lnf_w + i] = 1.0f;
  }

  gpu_copy_to_device(model.params, h_params.data(), total);
}

}  // namespace

void model_alloc(Model& model, const ModelConfig& config) {
  model.config = config;
  model.param_offsets = compute_param_offsets(config);
  model.act_offsets = compute_act_offsets(config);

  model.params = gpu_alloc(model.param_offsets.total);
  model.activations = gpu_alloc(model.act_offsets.total);
  model.gradients = gpu_alloc(model.param_offsets.total);
  model.grad_activations = gpu_alloc(model.act_offsets.total);
  model.d_tokens = gpu_alloc_u16(config.ctx_len);

  gpu_zero(model.params, model.param_offsets.total);
  gpu_zero(model.activations, model.act_offsets.total);
  gpu_zero(model.gradients, model.param_offsets.total);
  gpu_zero(model.grad_activations, model.act_offsets.total);

  CUBLAS_CHECK(cublasCreate(&model.cublas_handle));

  init_weights(model);
}

void model_free(Model& model) {
  if (model.cublas_handle) cublasDestroy(model.cublas_handle);
  gpu_free(model.params);
  gpu_free(model.activations);
  gpu_free(model.gradients);
  gpu_free(model.grad_activations);
  gpu_free(model.d_tokens);
  model.params = nullptr;
  model.activations = nullptr;
  model.gradients = nullptr;
  model.grad_activations = nullptr;
  model.d_tokens = nullptr;
  model.cublas_handle = nullptr;
}

void model_zero_grad(Model& model) {
  gpu_zero(model.gradients, model.param_offsets.total);
  gpu_zero(model.grad_activations, model.act_offsets.total);
}

void model_forward(Model& model, const uint16_t* tokens, int seq_len) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  const auto& ao = model.act_offsets;
  auto h = model.cublas_handle;
  int d = c.d_model;

  // Copy tokens to device
  gpu_copy_u16_to_device(model.d_tokens, tokens, seq_len);

  // 1. Embedding + positional
  gpu_embedding_lookup(model.act(ao.embedded), model.d_tokens, model.param(po.tok_emb), seq_len, d);
  gpu_add_vectors(model.act(ao.embedded), model.act(ao.embedded), model.param(po.pos_emb),
                  seq_len * d);

  float* residual = model.act(ao.embedded);

  // 2. Transformer blocks
  for (int l = 0; l < c.n_layers; ++l) {
    const auto& lp = po.layers[l];
    const auto& la = ao.layers[l];

    // LayerNorm 1
    gpu_layernorm(model.act(la.ln1_out), residual, model.param(lp.ln1_w), model.param(lp.ln1_b),
                  seq_len, d);

    // Q, K, V projections + bias
    gpu_matmul(h, model.act(la.q), model.act(la.ln1_out), model.param(lp.Wq), seq_len, d, d);
    gpu_matmul(h, model.act(la.k), model.act(la.ln1_out), model.param(lp.Wk), seq_len, d, d);
    gpu_matmul(h, model.act(la.v), model.act(la.ln1_out), model.param(lp.Wv), seq_len, d, d);

    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.act(la.q) + s * d, model.act(la.q) + s * d, model.param(lp.bq), d);
      gpu_add_vectors(model.act(la.k) + s * d, model.act(la.k) + s * d, model.param(lp.bk), d);
      gpu_add_vectors(model.act(la.v) + s * d, model.act(la.v) + s * d, model.param(lp.bv), d);
    }

    // Multi-head attention using per-head contiguous layout
    int dh = c.d_head();

    // Split heads: [seq_len × d_model] → [n_heads × seq_len × d_head]
    gpu_split_heads(model.act(la.q_heads), model.act(la.q), seq_len, c.n_heads, dh);
    gpu_split_heads(model.act(la.k_heads), model.act(la.k), seq_len, c.n_heads, dh);
    gpu_split_heads(model.act(la.v_heads), model.act(la.v), seq_len, c.n_heads, dh);

    // Per-head attention: scores = Q_h × K_h^T / sqrt(d_head), then mask + softmax, then attn × V_h
    for (int head = 0; head < c.n_heads; ++head) {
      float* q_h = model.act(la.q_heads) + static_cast<size_t>(head) * seq_len * dh;
      float* k_h = model.act(la.k_heads) + static_cast<size_t>(head) * seq_len * dh;
      float* v_h = model.act(la.v_heads) + static_cast<size_t>(head) * seq_len * dh;
      float* scores = model.act(la.attn_scores) + static_cast<size_t>(head) * seq_len * seq_len;
      float* out_h = model.act(la.attn_out_heads) + static_cast<size_t>(head) * seq_len * dh;

      // scores = Q_h × K_h^T  [seq_len × seq_len]
      gpu_matmul_transposed_b(h, scores, q_h, k_h, seq_len, dh, seq_len);

      // Scale by 1/sqrt(d_head)
      float scale = 1.0f / std::sqrt(static_cast<float>(dh));
      gpu_scale(scores, seq_len * seq_len, scale);
    }

    // Causal mask all heads at once
    gpu_causal_mask(model.act(la.attn_scores), c.n_heads, seq_len);

    // Softmax per head (each head is seq_len rows of seq_len cols)
    gpu_softmax(model.act(la.attn_scores), model.act(la.attn_scores), c.n_heads * seq_len,
                seq_len);

    // attn_out = softmax_scores × V_h per head
    for (int head = 0; head < c.n_heads; ++head) {
      float* scores = model.act(la.attn_scores) + static_cast<size_t>(head) * seq_len * seq_len;
      float* v_h = model.act(la.v_heads) + static_cast<size_t>(head) * seq_len * dh;
      float* out_h = model.act(la.attn_out_heads) + static_cast<size_t>(head) * seq_len * dh;

      gpu_matmul(h, out_h, scores, v_h, seq_len, seq_len, dh);
    }

    // Merge heads: [n_heads × seq_len × d_head] → [seq_len × d_model]
    gpu_merge_heads(model.act(la.attn_out), model.act(la.attn_out_heads), seq_len, c.n_heads, dh);

    // Output projection
    gpu_matmul(h, model.act(la.attn_proj), model.act(la.attn_out), model.param(lp.Wo), seq_len, d,
               d);
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.act(la.attn_proj) + s * d, model.act(la.attn_proj) + s * d,
                      model.param(lp.bo), d);
    }

    // Residual 1
    gpu_add_vectors(model.act(la.residual1), residual, model.act(la.attn_proj), seq_len * d);

    // LayerNorm 2
    gpu_layernorm(model.act(la.ln2_out), model.act(la.residual1), model.param(lp.ln2_w),
                  model.param(lp.ln2_b), seq_len, d);

    // MLP
    gpu_matmul(h, model.act(la.fc1_out), model.act(la.ln2_out), model.param(lp.fc1_w), seq_len, d,
               c.d_ff);
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.act(la.fc1_out) + s * c.d_ff, model.act(la.fc1_out) + s * c.d_ff,
                      model.param(lp.fc1_b), c.d_ff);
    }

    gpu_gelu(model.act(la.gelu_out), model.act(la.fc1_out), seq_len * c.d_ff);

    gpu_matmul(h, model.act(la.fc2_out), model.act(la.gelu_out), model.param(lp.fc2_w), seq_len,
               c.d_ff, d);
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.act(la.fc2_out) + s * d, model.act(la.fc2_out) + s * d,
                      model.param(lp.fc2_b), d);
    }

    // Residual 2
    gpu_add_vectors(model.act(la.residual2), model.act(la.residual1), model.act(la.fc2_out),
                    seq_len * d);

    residual = model.act(la.residual2);
  }

  // 3. Final layernorm
  gpu_layernorm(model.act(ao.lnf_out), residual, model.param(po.lnf_w), model.param(po.lnf_b),
                seq_len, d);

  // 4. Logits (weight-tied)
  gpu_matmul_transposed_b(h, model.act(ao.logits), model.act(ao.lnf_out), model.param(po.tok_emb),
                          seq_len, d, c.vocab_size);
}

void model_backward(Model& model, const uint16_t* tokens, int seq_len) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  const auto& ao = model.act_offsets;
  auto h = model.cublas_handle;
  int d = c.d_model;

  // 4. Backward logits: logits = lnf_out × tok_emb^T
  gpu_matmul(h, model.grad_act(ao.lnf_out), model.grad_act(ao.logits), model.param(po.tok_emb),
             seq_len, c.vocab_size, d);
  // d_tok_emb += dlogits^T × lnf_out
  gpu_matmul_transposed_a(h, model.grad(po.tok_emb), model.grad_act(ao.logits),
                          model.act(ao.lnf_out), c.vocab_size, seq_len, d);

  // 3. Backward final layernorm
  size_t lnf_in = (c.n_layers > 0) ? ao.layers[c.n_layers - 1].residual2 : ao.embedded;
  gpu_layernorm_backward(model.grad_act(lnf_in), model.grad(po.lnf_w), model.grad(po.lnf_b),
                         model.grad_act(ao.lnf_out), model.act(lnf_in), model.param(po.lnf_w),
                         seq_len, d);

  // 2. Backward transformer blocks
  for (int l = c.n_layers - 1; l >= 0; --l) {
    const auto& lp = po.layers[l];
    const auto& la = ao.layers[l];

    // Residual 2 backward
    gpu_add_vectors(model.grad_act(la.residual1), model.grad_act(la.residual1),
                    model.grad_act(la.residual2), seq_len * d);
    gpu_add_vectors(model.grad_act(la.fc2_out), model.grad_act(la.fc2_out),
                    model.grad_act(la.residual2), seq_len * d);

    // fc2 bias backward
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.grad(lp.fc2_b), model.grad(lp.fc2_b),
                      model.grad_act(la.fc2_out) + s * d, d);
    }

    // fc2 matmul backward
    gpu_matmul_backward(h, model.grad_act(la.fc2_out), model.act(la.gelu_out),
                        model.param(lp.fc2_w), model.grad_act(la.gelu_out), model.grad(lp.fc2_w),
                        seq_len, c.d_ff, d);

    // GELU backward
    gpu_gelu_backward(model.grad_act(la.fc1_out), model.grad_act(la.gelu_out),
                      model.act(la.fc1_out), seq_len * c.d_ff);

    // fc1 bias backward
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.grad(lp.fc1_b), model.grad(lp.fc1_b),
                      model.grad_act(la.fc1_out) + s * c.d_ff, c.d_ff);
    }

    // fc1 matmul backward
    gpu_matmul_backward(h, model.grad_act(la.fc1_out), model.act(la.ln2_out),
                        model.param(lp.fc1_w), model.grad_act(la.ln2_out), model.grad(lp.fc1_w),
                        seq_len, d, c.d_ff);

    // Layernorm 2 backward
    gpu_layernorm_backward(model.grad_act(la.residual1), model.grad(lp.ln2_w),
                           model.grad(lp.ln2_b), model.grad_act(la.ln2_out),
                           model.act(la.residual1), model.param(lp.ln2_w), seq_len, d);

    // Residual 1 backward
    size_t prev = (l > 0) ? ao.layers[l - 1].residual2 : ao.embedded;
    gpu_add_vectors(model.grad_act(prev), model.grad_act(prev), model.grad_act(la.residual1),
                    seq_len * d);
    gpu_add_vectors(model.grad_act(la.attn_proj), model.grad_act(la.attn_proj),
                    model.grad_act(la.residual1), seq_len * d);

    // Output projection bias backward
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.grad(lp.bo), model.grad(lp.bo),
                      model.grad_act(la.attn_proj) + s * d, d);
    }

    // Output projection matmul backward
    gpu_matmul_backward(h, model.grad_act(la.attn_proj), model.act(la.attn_out),
                        model.param(lp.Wo), model.grad_act(la.attn_out), model.grad(lp.Wo),
                        seq_len, d, d);

    // Attention backward (per-head layout)
    int dh = c.d_head();

    // d_attn_out is in [seq_len × d_model] — split to per-head
    gpu_split_heads(model.grad_act(la.attn_out_heads), model.grad_act(la.attn_out), seq_len,
                    c.n_heads, dh);

    for (int head = 0; head < c.n_heads; ++head) {
      size_t h_off = static_cast<size_t>(head) * seq_len * dh;
      size_t s_off = static_cast<size_t>(head) * seq_len * seq_len;

      float* d_out_h = model.grad_act(la.attn_out_heads) + h_off;
      float* scores = model.act(la.attn_scores) + s_off;
      float* v_h = model.act(la.v_heads) + h_off;

      // d_scores += d_out_h × V_h^T
      float* d_scores_h = model.grad_act(la.attn_scores) + s_off;
      gpu_matmul_transposed_b(model.cublas_handle, d_scores_h, d_out_h, v_h, seq_len, dh,
                              seq_len);

      // d_v_h += scores^T × d_out_h  (scores is [S×S], d_out_h is [S×dh], result is [S×dh])
      gpu_matmul_transposed_a(model.cublas_handle, model.grad_act(la.v_heads) + h_off, scores,
                              d_out_h, seq_len, seq_len, dh);
    }

    // Softmax backward — need a temp buffer since din==dout would cause read/write aliasing
    size_t scores_size = static_cast<size_t>(c.n_heads) * seq_len * seq_len;
    float* d_prescores = gpu_alloc(scores_size);
    gpu_zero(d_prescores, scores_size);
    gpu_softmax_backward(d_prescores, model.grad_act(la.attn_scores), model.act(la.attn_scores),
                         c.n_heads * seq_len, seq_len);

    // Zero out causal mask positions and apply scale
    gpu_causal_mask_zero(d_prescores, c.n_heads, seq_len);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(dh));
    gpu_scale(d_prescores, static_cast<int>(scores_size), attn_scale);

    // Copy back to grad_act for use by d_q and d_k computation
    CUDA_CHECK(cudaMemcpy(model.grad_act(la.attn_scores), d_prescores, scores_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    gpu_free(d_prescores);

    // d_q_heads and d_k_heads from d_scores
    for (int head = 0; head < c.n_heads; ++head) {
      size_t h_off = static_cast<size_t>(head) * seq_len * dh;
      size_t s_off = static_cast<size_t>(head) * seq_len * seq_len;

      float* d_scores_h = model.grad_act(la.attn_scores) + s_off;
      float* k_h = model.act(la.k_heads) + h_off;
      float* q_h = model.act(la.q_heads) + h_off;

      // d_q_h += d_scores × K_h
      gpu_matmul(model.cublas_handle, model.grad_act(la.q_heads) + h_off, d_scores_h, k_h,
                 seq_len, seq_len, dh);

      // d_k_h += d_scores^T × Q_h  (d_scores is [S×S], Q_h is [S×dh], result is [S×dh])
      gpu_matmul_transposed_a(model.cublas_handle, model.grad_act(la.k_heads) + h_off, d_scores_h,
                              q_h, seq_len, seq_len, dh);
    }

    // Merge head gradients back to [seq_len × d_model]
    gpu_merge_heads(model.grad_act(la.q), model.grad_act(la.q_heads), seq_len, c.n_heads, dh);
    gpu_merge_heads(model.grad_act(la.k), model.grad_act(la.k_heads), seq_len, c.n_heads, dh);
    gpu_merge_heads(model.grad_act(la.v), model.grad_act(la.v_heads), seq_len, c.n_heads, dh);

    // Q/K/V bias backward
    for (int s = 0; s < seq_len; ++s) {
      gpu_add_vectors(model.grad(lp.bq), model.grad(lp.bq), model.grad_act(la.q) + s * d, d);
      gpu_add_vectors(model.grad(lp.bk), model.grad(lp.bk), model.grad_act(la.k) + s * d, d);
      gpu_add_vectors(model.grad(lp.bv), model.grad(lp.bv), model.grad_act(la.v) + s * d, d);
    }

    // Q/K/V matmul backward
    gpu_matmul_backward(h, model.grad_act(la.q), model.act(la.ln1_out), model.param(lp.Wq),
                        model.grad_act(la.ln1_out), model.grad(lp.Wq), seq_len, d, d);
    gpu_matmul_backward(h, model.grad_act(la.k), model.act(la.ln1_out), model.param(lp.Wk),
                        model.grad_act(la.ln1_out), model.grad(lp.Wk), seq_len, d, d);
    gpu_matmul_backward(h, model.grad_act(la.v), model.act(la.ln1_out), model.param(lp.Wv),
                        model.grad_act(la.ln1_out), model.grad(lp.Wv), seq_len, d, d);

    // Layernorm 1 backward
    size_t ln1_in = (l > 0) ? ao.layers[l - 1].residual2 : ao.embedded;
    gpu_layernorm_backward(model.grad_act(ln1_in), model.grad(lp.ln1_w), model.grad(lp.ln1_b),
                           model.grad_act(la.ln1_out), model.act(ln1_in), model.param(lp.ln1_w),
                           seq_len, d);
  }

  // 1. Embedding backward
  gpu_embedding_backward(model.grad(po.tok_emb), model.grad_act(ao.embedded), model.d_tokens,
                         seq_len, d);
  gpu_add_vectors(model.grad(po.pos_emb), model.grad(po.pos_emb), model.grad_act(ao.embedded),
                  seq_len * d);
}

void model_save(const Model& model, const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write " + path);

  f.write(reinterpret_cast<const char*>(&model.config), sizeof(ModelConfig));

  auto num_params = static_cast<uint64_t>(model.param_offsets.total);
  f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));

  // Copy params from GPU to host, then write
  std::vector<float> h_params(num_params);
  gpu_copy_to_host(h_params.data(), model.params, num_params);
  f.write(reinterpret_cast<const char*>(h_params.data()),
          static_cast<std::streamsize>(num_params * sizeof(float)));
}

void model_load(Model& model, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot read " + path);

  ModelConfig config{};
  f.read(reinterpret_cast<char*>(&config), sizeof(ModelConfig));

  model_alloc(model, config);

  uint64_t num_params = 0;
  f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));

  // Read to host buffer, then copy to GPU
  std::vector<float> h_params(num_params);
  f.read(reinterpret_cast<char*>(h_params.data()),
         static_cast<std::streamsize>(num_params * sizeof(float)));
  gpu_copy_to_device(model.params, h_params.data(), num_params);
}
