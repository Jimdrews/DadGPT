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
  int B = c.batch_size;

  auto alloc = [&](size_t count) {
    size_t o = offset;
    offset += count;
    return o;
  };

  // Most activations use B*S rows of D columns (flattened batch+seq)
  size_t bsd = static_cast<size_t>(B) * c.ctx_len * c.d_model;

  a.embedded = alloc(bsd);

  a.layers.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) {
    auto& ly = a.layers[l];
    ly.ln1_out = alloc(bsd);
    ly.q = alloc(bsd);
    ly.k = alloc(bsd);
    ly.v = alloc(bsd);
    ly.q_heads = alloc(bsd);  // [B*H, S, dh]
    ly.k_heads = alloc(bsd);
    ly.v_heads = alloc(bsd);
    // Attention scores: [B*H, S, S]
    ly.attn_scores = alloc(static_cast<size_t>(B) * c.n_heads * c.ctx_len * c.ctx_len);
    ly.attn_out_heads = alloc(bsd);  // [B*H, S, dh]
    ly.attn_out = alloc(bsd);
    ly.attn_proj = alloc(bsd);
    ly.residual1 = alloc(bsd);
    ly.ln2_out = alloc(bsd);
    ly.fc1_out = alloc(static_cast<size_t>(B) * c.ctx_len * c.d_ff);
    ly.gelu_out = alloc(static_cast<size_t>(B) * c.ctx_len * c.d_ff);
    ly.fc2_out = alloc(bsd);
    ly.residual2 = alloc(bsd);
  }

  a.lnf_out = alloc(bsd);
  a.logits = alloc(static_cast<size_t>(B) * c.ctx_len * c.vocab_size);

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
  model.d_tokens = gpu_alloc_u16(static_cast<size_t>(config.batch_size) * config.ctx_len);

  // Pre-allocate temp buffer for backward attention (B * n_heads * ctx_len * ctx_len)
  size_t scores_buf_size = static_cast<size_t>(config.batch_size) * config.n_heads *
                           config.ctx_len * config.ctx_len;
  model.temp_scores = gpu_alloc(scores_buf_size);

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
  gpu_free(model.temp_scores);
  gpu_free(model.d_tokens);
  model.params = nullptr;
  model.activations = nullptr;
  model.gradients = nullptr;
  model.grad_activations = nullptr;
  model.temp_scores = nullptr;
  model.d_tokens = nullptr;
  model.cublas_handle = nullptr;
}

void model_zero_grad(Model& model) {
  gpu_zero(model.gradients, model.param_offsets.total);
  gpu_zero(model.grad_activations, model.act_offsets.total);
}

void model_forward(Model& model, const uint16_t* tokens, int batch_size, int seq_len) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  const auto& ao = model.act_offsets;
  auto h = model.cublas_handle;
  int d = c.d_model;
  int BS = batch_size * seq_len;  // flattened batch*seq for row-independent ops

  // Copy tokens to device
  gpu_copy_u16_to_device(model.d_tokens, tokens, BS);

  // 1. Embedding + positional (broadcast pos_emb across batch)
  gpu_embedding_lookup(model.act(ao.embedded), model.d_tokens, model.param(po.tok_emb), BS, d);
  gpu_add_bias(model.act(ao.embedded), model.param(po.pos_emb), batch_size, seq_len * d);

  float* residual = model.act(ao.embedded);

  // 2. Transformer blocks
  for (int l = 0; l < c.n_layers; ++l) {
    const auto& lp = po.layers[l];
    const auto& la = ao.layers[l];

    // LayerNorm 1 — BS rows of d columns
    gpu_layernorm(model.act(la.ln1_out), residual, model.param(lp.ln1_w), model.param(lp.ln1_b),
                  BS, d);

    // Q, K, V projections + bias — [BS, d] × [d, d] = [BS, d]
    gpu_matmul(h, model.act(la.q), model.act(la.ln1_out), model.param(lp.Wq), BS, d, d);
    gpu_matmul(h, model.act(la.k), model.act(la.ln1_out), model.param(lp.Wk), BS, d, d);
    gpu_matmul(h, model.act(la.v), model.act(la.ln1_out), model.param(lp.Wv), BS, d, d);

    gpu_add_bias(model.act(la.q), model.param(lp.bq), BS, d);
    gpu_add_bias(model.act(la.k), model.param(lp.bk), BS, d);
    gpu_add_bias(model.act(la.v), model.param(lp.bv), BS, d);

    // Multi-head attention
    int dh = c.d_head();
    int BH = batch_size * c.n_heads;

    // Split heads: [B*S, D] → [B*H, S, dh]
    gpu_split_heads(model.act(la.q_heads), model.act(la.q), batch_size, seq_len, c.n_heads, dh);
    gpu_split_heads(model.act(la.k_heads), model.act(la.k), batch_size, seq_len, c.n_heads, dh);
    gpu_split_heads(model.act(la.v_heads), model.act(la.v), batch_size, seq_len, c.n_heads, dh);

    // scores = Q × K^T / sqrt(d_head) — all B*H heads in one call
    long long stride_qkv = static_cast<long long>(seq_len) * dh;
    long long stride_scores = static_cast<long long>(seq_len) * seq_len;
    gpu_matmul_strided_batched_bt(h, model.act(la.attn_scores), model.act(la.q_heads),
                                  model.act(la.k_heads), seq_len, dh, seq_len, stride_qkv,
                                  stride_qkv, stride_scores, BH);

    // Scale by 1/sqrt(d_head)
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    gpu_scale(model.act(la.attn_scores), BH * seq_len * seq_len, scale);

    // Causal mask all B*H blocks
    gpu_causal_mask(model.act(la.attn_scores), BH, seq_len);

    // Softmax: B*H*S rows of S cols
    gpu_softmax(model.act(la.attn_scores), model.act(la.attn_scores), BH * seq_len, seq_len);

    // attn_out = scores × V — all B*H heads in one call
    gpu_matmul_strided_batched(h, model.act(la.attn_out_heads), model.act(la.attn_scores),
                               model.act(la.v_heads), seq_len, seq_len, dh, stride_scores,
                               stride_qkv, stride_qkv, BH);

    // Merge heads: [B*H, S, dh] → [B*S, D]
    gpu_merge_heads(model.act(la.attn_out), model.act(la.attn_out_heads), batch_size, seq_len,
                    c.n_heads, dh);

    // Output projection — [BS, d] × [d, d] = [BS, d]
    gpu_matmul(h, model.act(la.attn_proj), model.act(la.attn_out), model.param(lp.Wo), BS, d, d);
    gpu_add_bias(model.act(la.attn_proj), model.param(lp.bo), BS, d);

    // Residual 1
    gpu_add_vectors(model.act(la.residual1), residual, model.act(la.attn_proj), BS * d);

    // LayerNorm 2
    gpu_layernorm(model.act(la.ln2_out), model.act(la.residual1), model.param(lp.ln2_w),
                  model.param(lp.ln2_b), BS, d);

    // MLP
    gpu_matmul(h, model.act(la.fc1_out), model.act(la.ln2_out), model.param(lp.fc1_w), BS, d,
               c.d_ff);
    gpu_add_bias(model.act(la.fc1_out), model.param(lp.fc1_b), BS, c.d_ff);

    gpu_gelu(model.act(la.gelu_out), model.act(la.fc1_out), BS * c.d_ff);

    gpu_matmul(h, model.act(la.fc2_out), model.act(la.gelu_out), model.param(lp.fc2_w), BS,
               c.d_ff, d);
    gpu_add_bias(model.act(la.fc2_out), model.param(lp.fc2_b), BS, d);

    // Residual 2
    gpu_add_vectors(model.act(la.residual2), model.act(la.residual1), model.act(la.fc2_out),
                    BS * d);

    residual = model.act(la.residual2);
  }

  // 3. Final layernorm
  gpu_layernorm(model.act(ao.lnf_out), residual, model.param(po.lnf_w), model.param(po.lnf_b),
                BS, d);

  // 4. Logits (weight-tied) — [BS, d] × [vocab, d]^T = [BS, vocab]
  gpu_matmul_transposed_b(h, model.act(ao.logits), model.act(ao.lnf_out), model.param(po.tok_emb),
                          BS, d, c.vocab_size);
}

void model_backward(Model& model, const uint16_t* tokens, int batch_size, int seq_len) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  const auto& ao = model.act_offsets;
  auto h = model.cublas_handle;
  int d = c.d_model;
  int BS = batch_size * seq_len;

  // 4. Backward logits: logits = lnf_out × tok_emb^T
  gpu_matmul(h, model.grad_act(ao.lnf_out), model.grad_act(ao.logits), model.param(po.tok_emb),
             BS, c.vocab_size, d);
  gpu_matmul_transposed_a(h, model.grad(po.tok_emb), model.grad_act(ao.logits),
                          model.act(ao.lnf_out), c.vocab_size, BS, d);

  // 3. Backward final layernorm
  size_t lnf_in = (c.n_layers > 0) ? ao.layers[c.n_layers - 1].residual2 : ao.embedded;
  gpu_layernorm_backward(model.grad_act(lnf_in), model.grad(po.lnf_w), model.grad(po.lnf_b),
                         model.grad_act(ao.lnf_out), model.act(lnf_in), model.param(po.lnf_w),
                         BS, d);

  // 2. Backward transformer blocks
  for (int l = c.n_layers - 1; l >= 0; --l) {
    const auto& lp = po.layers[l];
    const auto& la = ao.layers[l];
    int dh = c.d_head();
    int BH = batch_size * c.n_heads;

    // Residual 2 backward
    gpu_add_vectors(model.grad_act(la.residual1), model.grad_act(la.residual1),
                    model.grad_act(la.residual2), BS * d);
    gpu_add_vectors(model.grad_act(la.fc2_out), model.grad_act(la.fc2_out),
                    model.grad_act(la.residual2), BS * d);

    // fc2 bias backward
    gpu_reduce_rows(model.grad(lp.fc2_b), model.grad_act(la.fc2_out), BS, d);

    // fc2 matmul backward
    gpu_matmul_backward(h, model.grad_act(la.fc2_out), model.act(la.gelu_out),
                        model.param(lp.fc2_w), model.grad_act(la.gelu_out), model.grad(lp.fc2_w),
                        BS, c.d_ff, d);

    // GELU backward
    gpu_gelu_backward(model.grad_act(la.fc1_out), model.grad_act(la.gelu_out),
                      model.act(la.fc1_out), BS * c.d_ff);

    // fc1 bias backward
    gpu_reduce_rows(model.grad(lp.fc1_b), model.grad_act(la.fc1_out), BS, c.d_ff);

    // fc1 matmul backward
    gpu_matmul_backward(h, model.grad_act(la.fc1_out), model.act(la.ln2_out),
                        model.param(lp.fc1_w), model.grad_act(la.ln2_out), model.grad(lp.fc1_w),
                        BS, d, c.d_ff);

    // Layernorm 2 backward
    gpu_layernorm_backward(model.grad_act(la.residual1), model.grad(lp.ln2_w),
                           model.grad(lp.ln2_b), model.grad_act(la.ln2_out),
                           model.act(la.residual1), model.param(lp.ln2_w), BS, d);

    // Residual 1 backward
    size_t prev = (l > 0) ? ao.layers[l - 1].residual2 : ao.embedded;
    gpu_add_vectors(model.grad_act(prev), model.grad_act(prev), model.grad_act(la.residual1),
                    BS * d);
    gpu_add_vectors(model.grad_act(la.attn_proj), model.grad_act(la.attn_proj),
                    model.grad_act(la.residual1), BS * d);

    // Output projection bias backward
    gpu_reduce_rows(model.grad(lp.bo), model.grad_act(la.attn_proj), BS, d);

    // Output projection matmul backward
    gpu_matmul_backward(h, model.grad_act(la.attn_proj), model.act(la.attn_out),
                        model.param(lp.Wo), model.grad_act(la.attn_out), model.grad(lp.Wo),
                        BS, d, d);

    // Attention backward — split to [B*H, S, dh]
    gpu_split_heads(model.grad_act(la.attn_out_heads), model.grad_act(la.attn_out), batch_size,
                    seq_len, c.n_heads, dh);

    long long stride_qkv = static_cast<long long>(seq_len) * dh;
    long long stride_scores = static_cast<long long>(seq_len) * seq_len;

    // d_scores = d_attn_out × V^T  (all B*H heads at once)
    gpu_matmul_strided_batched_bt(h, model.grad_act(la.attn_scores),
                                  model.grad_act(la.attn_out_heads), model.act(la.v_heads),
                                  seq_len, dh, seq_len, stride_qkv, stride_qkv, stride_scores, BH);

    // d_V += scores^T × d_attn_out (all B*H heads, accumulate)
    gpu_matmul_strided_batched_at(h, model.grad_act(la.v_heads), model.act(la.attn_scores),
                                  model.grad_act(la.attn_out_heads), seq_len, seq_len, dh,
                                  stride_scores, stride_qkv, stride_qkv, BH);

    // Softmax backward — use pre-allocated temp buffer
    size_t scores_size = static_cast<size_t>(BH) * seq_len * seq_len;
    gpu_zero(model.temp_scores, scores_size);
    gpu_softmax_backward(model.temp_scores, model.grad_act(la.attn_scores),
                         model.act(la.attn_scores), BH * seq_len, seq_len);

    // Zero out causal mask positions and apply scale
    gpu_causal_mask_zero(model.temp_scores, BH, seq_len);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(dh));
    gpu_scale(model.temp_scores, static_cast<int>(scores_size), attn_scale);

    // Copy back to grad_act
    CUDA_CHECK(cudaMemcpy(model.grad_act(la.attn_scores), model.temp_scores,
                          scores_size * sizeof(float), cudaMemcpyDeviceToDevice));

    // d_Q = d_scores × K  (all B*H heads)
    gpu_matmul_strided_batched(h, model.grad_act(la.q_heads), model.grad_act(la.attn_scores),
                               model.act(la.k_heads), seq_len, seq_len, dh, stride_scores,
                               stride_qkv, stride_qkv, BH);

    // d_K += d_scores^T × Q (all B*H heads, accumulate)
    gpu_matmul_strided_batched_at(h, model.grad_act(la.k_heads), model.grad_act(la.attn_scores),
                                  model.act(la.q_heads), seq_len, seq_len, dh, stride_scores,
                                  stride_qkv, stride_qkv, BH);

    // Merge head gradients: [B*H, S, dh] → [B*S, D]
    gpu_merge_heads(model.grad_act(la.q), model.grad_act(la.q_heads), batch_size, seq_len,
                    c.n_heads, dh);
    gpu_merge_heads(model.grad_act(la.k), model.grad_act(la.k_heads), batch_size, seq_len,
                    c.n_heads, dh);
    gpu_merge_heads(model.grad_act(la.v), model.grad_act(la.v_heads), batch_size, seq_len,
                    c.n_heads, dh);

    // Q/K/V bias backward
    gpu_reduce_rows(model.grad(lp.bq), model.grad_act(la.q), BS, d);
    gpu_reduce_rows(model.grad(lp.bk), model.grad_act(la.k), BS, d);
    gpu_reduce_rows(model.grad(lp.bv), model.grad_act(la.v), BS, d);

    // Q/K/V matmul backward
    gpu_matmul_backward(h, model.grad_act(la.q), model.act(la.ln1_out), model.param(lp.Wq),
                        model.grad_act(la.ln1_out), model.grad(lp.Wq), BS, d, d);
    gpu_matmul_backward(h, model.grad_act(la.k), model.act(la.ln1_out), model.param(lp.Wk),
                        model.grad_act(la.ln1_out), model.grad(lp.Wk), BS, d, d);
    gpu_matmul_backward(h, model.grad_act(la.v), model.act(la.ln1_out), model.param(lp.Wv),
                        model.grad_act(la.ln1_out), model.grad(lp.Wv), BS, d, d);

    // Layernorm 1 backward
    size_t ln1_in = (l > 0) ? ao.layers[l - 1].residual2 : ao.embedded;
    gpu_layernorm_backward(model.grad_act(ln1_in), model.grad(lp.ln1_w), model.grad(lp.ln1_b),
                           model.grad_act(la.ln1_out), model.act(ln1_in), model.param(lp.ln1_w),
                           BS, d);
  }

  // 1. Embedding backward
  gpu_embedding_backward(model.grad(po.tok_emb), model.grad_act(ao.embedded), model.d_tokens,
                         BS, d);
  // Positional embedding gradient: accumulate across batch
  gpu_reduce_rows(model.grad(po.pos_emb), model.grad_act(ao.embedded), batch_size, seq_len * d);
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
