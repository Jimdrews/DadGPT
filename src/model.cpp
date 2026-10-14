#include "model.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_ops.cuh"
#include "optimizer.hpp"

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

  // Dropout mask offsets (uint8 buffer, separate from float activations)
  size_t doff = 0;
  auto dalloc = [&](size_t count) {
    size_t o = doff;
    doff += count;
    return o;
  };
  a.drop_emb = dalloc(bsd);  // after embedding
  a.drop_masks.resize(c.n_layers);
  for (int l = 0; l < c.n_layers; ++l) {
    auto& dm = a.drop_masks[l];
    dm.attn_scores = dalloc(static_cast<size_t>(B) * c.n_heads * c.ctx_len * c.ctx_len);
    dm.attn_proj = dalloc(bsd);
    dm.mlp = dalloc(bsd);
  }
  a.drop_total = doff;

  return a;
}

// Generate weights on CPU, then copy to GPU
void init_weights(Model& model) {
  const auto& c = model.config;
  size_t total = model.param_offsets.total;
  std::vector<float> h_params(total, 0.0f);

  std::mt19937 rng(42);

  auto normal_init = [&](size_t offset, size_t count, float std_dev) {
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (size_t i = 0; i < count; ++i) h_params[offset + i] = dist(rng);
  };

  // GPT-2 init: every weight N(0, 0.02); the two projections that write into the residual
  // stream (Wo, fc2) are scaled by 1/sqrt(2 * n_layers) so the residual variance stays
  // roughly constant with depth. Biases are zero, LayerNorm weights one.
  const float std_dev = 0.02f;
  const float residual_std = std_dev / std::sqrt(2.0f * static_cast<float>(c.n_layers));
  const size_t dd = static_cast<size_t>(c.d_model) * c.d_model;
  const size_t dff = static_cast<size_t>(c.d_model) * c.d_ff;

  const auto& po = model.param_offsets;

  normal_init(po.tok_emb, static_cast<size_t>(c.vocab_size) * c.d_model, std_dev);
  normal_init(po.pos_emb, static_cast<size_t>(c.ctx_len) * c.d_model, std_dev);

  for (int l = 0; l < c.n_layers; ++l) {
    const auto& ly = po.layers[l];

    for (int i = 0; i < c.d_model; ++i) {
      h_params[ly.ln1_w + i] = 1.0f;
      h_params[ly.ln2_w + i] = 1.0f;
    }

    normal_init(ly.Wq, dd, std_dev);
    normal_init(ly.Wk, dd, std_dev);
    normal_init(ly.Wv, dd, std_dev);
    normal_init(ly.Wo, dd, residual_std);
    normal_init(ly.fc1_w, dff, std_dev);
    normal_init(ly.fc2_w, dff, residual_std);
  }

  for (int i = 0; i < c.d_model; ++i) {
    h_params[po.lnf_w + i] = 1.0f;
  }

  gpu_copy_to_device(model.params, h_params.data(), total);
}

// Activation buffers are sized for config.batch_size x ctx_len; larger inputs would overflow.
void check_batch_shape(const ModelConfig& c, int batch_size, int seq_len) {
  if (batch_size < 1 || batch_size > c.batch_size || seq_len < 1 || seq_len > c.ctx_len) {
    throw std::out_of_range("batch " + std::to_string(batch_size) + "x" +
                            std::to_string(seq_len) + " exceeds model buffers " +
                            std::to_string(c.batch_size) + "x" + std::to_string(c.ctx_len));
  }
}

// Dropped attention probabilities, one [B*H, ctx, ctx] block per layer. Kept outside the
// activation layout so it doesn't also get a (never used) gradient mirror.
size_t attn_block_size(const ModelConfig& c) {
  return static_cast<size_t>(c.batch_size) * c.n_heads * c.ctx_len * c.ctx_len;
}

void ensure_attn_dropout_buffer(Model& model) {
  if (!model.attn_dropped) {
    model.attn_dropped = gpu_alloc(attn_block_size(model.config) * model.config.n_layers);
  }
}

float* attn_dropped_probs(Model& model, int layer) {
  return model.attn_dropped + attn_block_size(model.config) * layer;
}

}  // namespace

ModelConfig model_preset(const std::string& name) {
  ModelConfig c;
  if (name == "tiny") {
    c.d_model = 128, c.n_layers = 2, c.n_heads = 4, c.d_ff = 512, c.ctx_len = 64;
  } else if (name == "small") {
    c.d_model = 512, c.n_layers = 8, c.n_heads = 8, c.d_ff = 2048, c.ctx_len = 128;
  } else if (name == "gpt2-124m") {
    c.d_model = 768, c.n_layers = 12, c.n_heads = 12, c.d_ff = 3072, c.ctx_len = 1024;
  } else {
    throw std::invalid_argument("Unknown model preset '" + name +
                                "' (expected tiny, small or gpt2-124m)");
  }
  return c;
}

void model_alloc(Model& model, const ModelConfig& config) {
  if (config.n_heads <= 0 || config.d_model % config.n_heads != 0) {
    throw std::invalid_argument("d_model must be divisible by n_heads");
  }
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

  // Dropout masks (uint8)
  if (model.act_offsets.drop_total > 0) {
    model.dropout_masks = gpu_alloc_u8(model.act_offsets.drop_total);
  }

  // Weight decay flags: 1 for weight matrices, 0 for biases and layernorm
  {
    std::vector<uint8_t> flags(model.param_offsets.total, 0);
    const auto& po = model.param_offsets;
    auto mark_decay = [&](size_t offset, size_t count) {
      for (size_t i = 0; i < count; ++i) flags[offset + i] = 1;
    };
    // Token and positional embeddings get decay
    mark_decay(po.tok_emb, static_cast<size_t>(config.vocab_size) * config.d_model);
    mark_decay(po.pos_emb, static_cast<size_t>(config.ctx_len) * config.d_model);
    for (int l = 0; l < config.n_layers; ++l) {
      const auto& ly = po.layers[l];
      // Weight matrices get decay; biases and layernorm params don't
      mark_decay(ly.Wq, static_cast<size_t>(config.d_model) * config.d_model);
      mark_decay(ly.Wk, static_cast<size_t>(config.d_model) * config.d_model);
      mark_decay(ly.Wv, static_cast<size_t>(config.d_model) * config.d_model);
      mark_decay(ly.Wo, static_cast<size_t>(config.d_model) * config.d_model);
      mark_decay(ly.fc1_w, static_cast<size_t>(config.d_model) * config.d_ff);
      mark_decay(ly.fc2_w, static_cast<size_t>(config.d_ff) * config.d_model);
    }
    model.decay_flags = gpu_alloc_u8(model.param_offsets.total);
    CUDA_CHECK(cudaMemcpy(model.decay_flags, flags.data(), model.param_offsets.total,
                          cudaMemcpyHostToDevice));
  }

  gpu_zero(model.params, model.param_offsets.total);
  gpu_zero(model.activations, model.act_offsets.total);
  gpu_zero(model.gradients, model.param_offsets.total);
  gpu_zero(model.grad_activations, model.act_offsets.total);

  CUBLAS_CHECK(cublasCreate(&model.cublas_handle));

  // Initialize cuRAND for dropout
  gpu_dropout_init(42, static_cast<int>(
      std::min(model.act_offsets.drop_total, static_cast<size_t>(256 * 1024))));

  init_weights(model);
}

void model_free(Model& model) {
  if (model.cublas_handle) cublasDestroy(model.cublas_handle);
  gpu_free(model.params);
  gpu_free(model.activations);
  gpu_free(model.gradients);
  gpu_free(model.grad_activations);
  gpu_free(model.temp_scores);
  gpu_free(model.dropout_masks);
  gpu_free(model.decay_flags);
  gpu_free(model.d_tokens);
  gpu_free(model.attn_dropped);
  gpu_dropout_free();
  model.params = nullptr;
  model.activations = nullptr;
  model.gradients = nullptr;
  model.grad_activations = nullptr;
  model.temp_scores = nullptr;
  model.dropout_masks = nullptr;
  model.decay_flags = nullptr;
  model.d_tokens = nullptr;
  model.attn_dropped = nullptr;
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
  check_batch_shape(c, batch_size, seq_len);

  bool attn_dropout = model.training && model.dropout_rate > 0.0f;
  if (attn_dropout) ensure_attn_dropout_buffer(model);

  // Copy tokens to device
  gpu_copy_u16_to_device(model.d_tokens, tokens, BS);

  // 1. Embedding + positional (broadcast pos_emb across batch)
  gpu_embedding_lookup(model.act(ao.embedded), model.d_tokens, model.param(po.tok_emb), BS, d);
  gpu_add_bias(model.act(ao.embedded), model.param(po.pos_emb), batch_size, seq_len * d);

  // Dropout after embedding
  if (model.training && model.dropout_rate > 0.0f) {
    gpu_dropout_forward(model.act(ao.embedded), model.act(ao.embedded),
                        model.drop_mask(ao.drop_emb), BS * d, model.dropout_rate);
  }

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

    // Attention dropout. The softmax output P stays in attn_scores (softmax backward needs
    // it); the dropped probabilities go to a separate buffer that is used for P × V.
    const float* probs = model.act(la.attn_scores);
    if (attn_dropout) {
      float* dropped = attn_dropped_probs(model, l);
      gpu_dropout_forward(dropped, probs, model.drop_mask(ao.drop_masks[l].attn_scores),
                          BH * seq_len * seq_len, model.dropout_rate);
      probs = dropped;
    }

    // attn_out = probs × V — all B*H heads in one call
    gpu_matmul_strided_batched(h, model.act(la.attn_out_heads), probs, model.act(la.v_heads),
                               seq_len, seq_len, dh, stride_scores, stride_qkv, stride_qkv, BH);

    // Merge heads: [B*H, S, dh] → [B*S, D]
    gpu_merge_heads(model.act(la.attn_out), model.act(la.attn_out_heads), batch_size, seq_len,
                    c.n_heads, dh);

    // Output projection — [BS, d] × [d, d] = [BS, d]
    gpu_matmul(h, model.act(la.attn_proj), model.act(la.attn_out), model.param(lp.Wo), BS, d, d);
    gpu_add_bias(model.act(la.attn_proj), model.param(lp.bo), BS, d);

    // Dropout after attention projection
    if (model.training && model.dropout_rate > 0.0f) {
      gpu_dropout_forward(model.act(la.attn_proj), model.act(la.attn_proj),
                          model.drop_mask(ao.drop_masks[l].attn_proj), BS * d, model.dropout_rate);
    }

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

    // Dropout after MLP
    if (model.training && model.dropout_rate > 0.0f) {
      gpu_dropout_forward(model.act(la.fc2_out), model.act(la.fc2_out),
                          model.drop_mask(ao.drop_masks[l].mlp), BS * d, model.dropout_rate);
    }

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
  check_batch_shape(c, batch_size, seq_len);
  bool attn_dropout = model.training && model.dropout_rate > 0.0f;

  // 4. Backward logits: logits = lnf_out × tok_emb^T
  gpu_matmul(h, model.grad_act(ao.lnf_out), model.grad_act(ao.logits), model.param(po.tok_emb),
             BS, c.vocab_size, d);
  // Accumulate: tok_emb is tied, and grads must also sum across accumulation micro-batches
  gpu_matmul_transposed_a(h, model.grad(po.tok_emb), model.grad_act(ao.logits),
                          model.act(ao.lnf_out), c.vocab_size, BS, d, /*accumulate=*/true);

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

    // Dropout backward (MLP)
    if (model.training && model.dropout_rate > 0.0f) {
      gpu_dropout_backward(model.grad_act(la.fc2_out),
                           model.drop_mask(ao.drop_masks[l].mlp), BS * d, model.dropout_rate);
    }

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

    // Dropout backward (attention projection)
    if (model.training && model.dropout_rate > 0.0f) {
      gpu_dropout_backward(model.grad_act(la.attn_proj),
                           model.drop_mask(ao.drop_masks[l].attn_proj), BS * d, model.dropout_rate);
    }

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

    // Forward was attn_out = Pd × V with Pd = dropout(P), or P itself without dropout
    const float* probs =
        attn_dropout ? attn_dropped_probs(model, l) : model.act(la.attn_scores);

    // d_Pd = d_attn_out × V^T  (all B*H heads at once)
    gpu_matmul_strided_batched_bt(h, model.grad_act(la.attn_scores),
                                  model.grad_act(la.attn_out_heads), model.act(la.v_heads),
                                  seq_len, dh, seq_len, stride_qkv, stride_qkv, stride_scores, BH);

    // d_V += Pd^T × d_attn_out (all B*H heads, accumulate)
    gpu_matmul_strided_batched_at(h, model.grad_act(la.v_heads), probs,
                                  model.grad_act(la.attn_out_heads), seq_len, seq_len, dh,
                                  stride_scores, stride_qkv, stride_qkv, BH);

    // d_P = mask * scale * d_Pd, then back through the softmax using the undropped P
    size_t scores_size = static_cast<size_t>(BH) * seq_len * seq_len;
    if (attn_dropout) {
      gpu_dropout_backward(model.grad_act(la.attn_scores),
                           model.drop_mask(ao.drop_masks[l].attn_scores),
                           static_cast<int>(scores_size), model.dropout_rate);
    }
    float* d_scores = model.temp_scores;
    gpu_zero(d_scores, scores_size);
    gpu_softmax_backward(d_scores, model.grad_act(la.attn_scores), model.act(la.attn_scores),
                         BH * seq_len, seq_len);

    // Zero out causal mask positions and apply scale
    gpu_causal_mask_zero(d_scores, BH, seq_len);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(dh));
    gpu_scale(d_scores, static_cast<int>(scores_size), attn_scale);

    // d_Q = d_scores × K  (all B*H heads)
    gpu_matmul_strided_batched(h, model.grad_act(la.q_heads), d_scores, model.act(la.k_heads),
                               seq_len, seq_len, dh, stride_scores, stride_qkv, stride_qkv, BH);

    // d_K += d_scores^T × Q (all B*H heads, accumulate)
    gpu_matmul_strided_batched_at(h, model.grad_act(la.k_heads), d_scores, model.act(la.q_heads),
                                  seq_len, seq_len, dh, stride_scores, stride_qkv, stride_qkv,
                                  BH);

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

  // Dropout backward (embedding)
  if (model.training && model.dropout_rate > 0.0f) {
    gpu_dropout_backward(model.grad_act(ao.embedded), model.drop_mask(ao.drop_emb), BS * d,
                         model.dropout_rate);
  }

  // 1. Embedding backward
  gpu_embedding_backward(model.grad(po.tok_emb), model.grad_act(ao.embedded), model.d_tokens,
                         BS, d);
  // Positional embedding gradient: accumulate across batch
  gpu_reduce_rows(model.grad(po.pos_emb), model.grad_act(ao.embedded), batch_size, seq_len * d);
}

void kv_cache_alloc(KVCache& cache, const ModelConfig& config) {
  cache.n_layers = config.n_layers;
  cache.n_heads = config.n_heads;
  cache.d_head = config.d_head();
  cache.max_len = config.ctx_len;
  cache.length = 0;

  size_t per_layer = static_cast<size_t>(config.n_heads) * config.ctx_len * cache.d_head;
  cache.k_cache.resize(config.n_layers);
  cache.v_cache.resize(config.n_layers);
  for (int l = 0; l < config.n_layers; ++l) {
    cache.k_cache[l] = gpu_alloc(per_layer);
    cache.v_cache[l] = gpu_alloc(per_layer);
    gpu_zero(cache.k_cache[l], per_layer);
    gpu_zero(cache.v_cache[l], per_layer);
  }
}

void kv_cache_free(KVCache& cache) {
  for (int l = 0; l < cache.n_layers; ++l) {
    gpu_free(cache.k_cache[l]);
    gpu_free(cache.v_cache[l]);
  }
  cache.k_cache.clear();
  cache.v_cache.clear();
  cache.length = 0;
}

void model_forward_cached(Model& model, KVCache& cache, uint16_t token, int pos) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  auto h = model.cublas_handle;
  int d = c.d_model;
  int dh = c.d_head();
  if (pos < 0 || pos >= c.ctx_len || pos >= cache.max_len) {
    throw std::out_of_range("model_forward_cached: pos " + std::to_string(pos) +
                            " outside context length " + std::to_string(c.ctx_len));
  }

  // We use small scratch buffers from model.activations for single-position computation.
  // The activation buffer is big enough (sized for batch*ctx_len), so we use the first row.
  // q_buf, k_buf, v_buf: [1, d]
  // ln_buf: [1, d]
  // fc1_buf: [1, d_ff]
  // fc2_buf: [1, d]

  // Allocate temp buffers on stack for host-side single token
  gpu_copy_u16_to_device(model.d_tokens, &token, 1);

  // Embedding for single token at position pos
  float* emb = model.activations;  // reuse start of activation buffer
  gpu_embedding_lookup(emb, model.d_tokens, model.param(po.tok_emb), 1, d);
  // Add positional embedding for this position
  gpu_add_vectors(emb, emb, model.param(po.pos_emb) + pos * d, d);

  float* residual = emb;
  // We need scratch space after emb. Use offsets into activation buffer.
  float* ln_out = emb + d;
  float* q_buf = ln_out + d;
  float* k_buf = q_buf + d;
  float* v_buf = k_buf + d;
  float* attn_out_buf = v_buf + d;
  float* proj_buf = attn_out_buf + d;
  float* res1_buf = proj_buf + d;
  float* ln2_buf = res1_buf + d;
  float* fc1_buf = ln2_buf + d;
  float* gelu_buf = fc1_buf + c.d_ff;
  float* fc2_buf = gelu_buf + c.d_ff;
  float* res2_buf = fc2_buf + d;
  float* lnf_buf = res2_buf + d;
  float* logits_buf = lnf_buf + d;

  for (int l = 0; l < c.n_layers; ++l) {
    const auto& lp = po.layers[l];

    // LayerNorm 1
    gpu_layernorm(ln_out, residual, model.param(lp.ln1_w), model.param(lp.ln1_b), 1, d);

    // Q, K, V projections for single position
    gpu_matmul(h, q_buf, ln_out, model.param(lp.Wq), 1, d, d);
    gpu_matmul(h, k_buf, ln_out, model.param(lp.Wk), 1, d, d);
    gpu_matmul(h, v_buf, ln_out, model.param(lp.Wv), 1, d, d);
    gpu_add_bias(q_buf, model.param(lp.bq), 1, d);
    gpu_add_bias(k_buf, model.param(lp.bk), 1, d);
    gpu_add_bias(v_buf, model.param(lp.bv), 1, d);

    // Store K, V into cache at position pos
    // K cache layout: [H, max_len, dh] — copy each head's dh values into position pos
    // k_buf is [d] = [H*dh], cache is [H, max_len, dh]
    for (int head = 0; head < c.n_heads; ++head) {
      // k_buf[head*dh ... head*dh + dh-1] → k_cache[head*max_len*dh + pos*dh]
      CUDA_CHECK(cudaMemcpy(
          cache.k_cache[l] + static_cast<size_t>(head) * cache.max_len * dh + pos * dh,
          k_buf + head * dh, dh * sizeof(float), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(
          cache.v_cache[l] + static_cast<size_t>(head) * cache.max_len * dh + pos * dh,
          v_buf + head * dh, dh * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Attention: Q[1, dh] × K_cache[pos+1, dh]^T = scores[1, pos+1] per head
    // Then softmax, then scores × V_cache = out[1, dh]
    int cache_len = pos + 1;

    // Use scratch for scores: [H, 1, cache_len] — need H*cache_len floats
    float* scores_buf = logits_buf + c.vocab_size;  // borrow space after logits area

    for (int head = 0; head < c.n_heads; ++head) {
      float* q_h = q_buf + head * dh;
      float* k_cached = cache.k_cache[l] + static_cast<size_t>(head) * cache.max_len * dh;
      float* v_cached = cache.v_cache[l] + static_cast<size_t>(head) * cache.max_len * dh;
      float* s_h = scores_buf + head * cache_len;

      // scores = Q[1, dh] × K[cache_len, dh]^T = [1, cache_len]
      gpu_matmul_transposed_b(h, s_h, q_h, k_cached, 1, dh, cache_len);

      // Scale
      float scale = 1.0f / std::sqrt(static_cast<float>(dh));
      gpu_scale(s_h, cache_len, scale);
    }

    // Softmax each head's row: H rows of cache_len cols
    gpu_softmax(scores_buf, scores_buf, c.n_heads, cache_len);

    // attn_out per head: scores[1, cache_len] × V[cache_len, dh] = [1, dh]
    // Merge directly into attn_out_buf [d]
    for (int head = 0; head < c.n_heads; ++head) {
      float* s_h = scores_buf + head * cache_len;
      float* v_cached = cache.v_cache[l] + static_cast<size_t>(head) * cache.max_len * dh;
      float* out_h = attn_out_buf + head * dh;

      gpu_matmul(h, out_h, s_h, v_cached, 1, cache_len, dh);
    }

    // Output projection
    gpu_matmul(h, proj_buf, attn_out_buf, model.param(lp.Wo), 1, d, d);
    gpu_add_bias(proj_buf, model.param(lp.bo), 1, d);

    // Residual 1
    gpu_add_vectors(res1_buf, residual, proj_buf, d);

    // LayerNorm 2
    gpu_layernorm(ln2_buf, res1_buf, model.param(lp.ln2_w), model.param(lp.ln2_b), 1, d);

    // MLP
    gpu_matmul(h, fc1_buf, ln2_buf, model.param(lp.fc1_w), 1, d, c.d_ff);
    gpu_add_bias(fc1_buf, model.param(lp.fc1_b), 1, c.d_ff);
    gpu_gelu(gelu_buf, fc1_buf, c.d_ff);
    gpu_matmul(h, fc2_buf, gelu_buf, model.param(lp.fc2_w), 1, c.d_ff, d);
    gpu_add_bias(fc2_buf, model.param(lp.fc2_b), 1, d);

    // Residual 2
    gpu_add_vectors(res2_buf, res1_buf, fc2_buf, d);
    residual = res2_buf;
  }

  // Final layernorm
  gpu_layernorm(lnf_buf, residual, model.param(po.lnf_w), model.param(po.lnf_b), 1, d);

  // Logits (weight-tied)
  gpu_matmul_transposed_b(h, logits_buf, lnf_buf, model.param(po.tok_emb), 1, d, c.vocab_size);

  // Copy logits to the standard location so generate_tokens can read them
  CUDA_CHECK(cudaMemcpy(model.act(model.act_offsets.logits), logits_buf,
                        c.vocab_size * sizeof(float), cudaMemcpyDeviceToDevice));

  cache.length = pos + 1;
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

namespace {

// Reads config + params from an open checkpoint stream and allocates the model.
// batch_size > 0 overrides the saved batch size (activation buffers are sized from it).
void read_checkpoint_params(Model& model, std::ifstream& f, const std::string& path,
                            int batch_size) {
  ModelConfig config{};
  f.read(reinterpret_cast<char*>(&config), sizeof(ModelConfig));
  if (!f) throw std::runtime_error("Truncated checkpoint header in " + path);
  if (batch_size > 0) config.batch_size = batch_size;

  model_alloc(model, config);

  uint64_t num_params = 0;
  f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
  if (!f || num_params != model.param_offsets.total) {
    throw std::runtime_error("Checkpoint " + path + " param count does not match its config");
  }

  // Read to host buffer, then copy to GPU
  std::vector<float> h_params(num_params);
  f.read(reinterpret_cast<char*>(h_params.data()),
         static_cast<std::streamsize>(num_params * sizeof(float)));
  if (!f) throw std::runtime_error("Truncated checkpoint params in " + path);
  gpu_copy_to_device(model.params, h_params.data(), num_params);
}

}  // namespace

void model_load(Model& model, const std::string& path, int batch_size) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot read " + path);
  read_checkpoint_params(model, f, path, batch_size);
}

void model_save_with_optimizer(const Model& model, const Adam& opt, const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write " + path);

  // Model config + params (same as model_save)
  f.write(reinterpret_cast<const char*>(&model.config), sizeof(ModelConfig));
  auto num_params = static_cast<uint64_t>(model.param_offsets.total);
  f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
  std::vector<float> h_params(num_params);
  gpu_copy_to_host(h_params.data(), model.params, num_params);
  f.write(reinterpret_cast<const char*>(h_params.data()),
          static_cast<std::streamsize>(num_params * sizeof(float)));

  // Optimizer state
  uint8_t has_opt = 1;
  f.write(reinterpret_cast<const char*>(&has_opt), 1);
  f.write(reinterpret_cast<const char*>(&opt.step), sizeof(opt.step));
  std::vector<float> h_m(num_params), h_v(num_params);
  gpu_copy_to_host(h_m.data(), opt.m, num_params);
  gpu_copy_to_host(h_v.data(), opt.v, num_params);
  f.write(reinterpret_cast<const char*>(h_m.data()),
          static_cast<std::streamsize>(num_params * sizeof(float)));
  f.write(reinterpret_cast<const char*>(h_v.data()),
          static_cast<std::streamsize>(num_params * sizeof(float)));
}

void model_load_with_optimizer(Model& model, Adam& opt, const std::string& path,
                               int batch_size) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot read " + path);
  read_checkpoint_params(model, f, path, batch_size);

  size_t num_params = model.param_offsets.total;
  opt.num_params = num_params;
  if (!opt.m) opt.m = gpu_alloc(num_params);
  if (!opt.v) opt.v = gpu_alloc(num_params);
  gpu_zero(opt.m, num_params);
  gpu_zero(opt.v, num_params);
  opt.step = 0;

  // Optimizer state is optional (weights-only checkpoints end here)
  uint8_t has_opt = 0;
  f.read(reinterpret_cast<char*>(&has_opt), 1);
  if (f && has_opt == 1) {
    int step = 0;
    f.read(reinterpret_cast<char*>(&step), sizeof(step));
    std::vector<float> h_m(num_params), h_v(num_params);
    f.read(reinterpret_cast<char*>(h_m.data()),
           static_cast<std::streamsize>(num_params * sizeof(float)));
    f.read(reinterpret_cast<char*>(h_v.data()),
           static_cast<std::streamsize>(num_params * sizeof(float)));
    if (!f) throw std::runtime_error("Truncated optimizer state in " + path);
    opt.step = step;
    gpu_copy_to_device(opt.m, h_m.data(), num_params);
    gpu_copy_to_device(opt.v, h_v.data(), num_params);
  }
}
