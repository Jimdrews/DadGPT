#pragma once

#include <cublas_v2.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ModelConfig {
  int d_model = 512;
  int n_layers = 8;
  int n_heads = 8;
  int d_ff = 2048;
  int ctx_len = 128;
  int vocab_size = 512;
  int batch_size = 1;  // activation buffers sized for this batch size

  int d_head() const { return d_model / n_heads; }
};

// Named architectures. vocab_size and batch_size are left at defaults; the caller sets them
// from the tokenizer and the training run. Throws on an unknown name.
//   tiny       d=128  L=2  H=4  ff=512   ctx=64    (smoke tests)
//   small      d=512  L=8  H=8  ff=2048  ctx=128   (the original dad-gpt model)
//   gpt2-124m  d=768  L=12 H=12 ff=3072  ctx=1024  (GPT-2 small)
ModelConfig model_preset(const std::string& name);

struct ParameterOffsets {
  size_t tok_emb;
  size_t pos_emb;

  // Per-layer offsets (indexed by layer)
  struct LayerOffsets {
    size_t ln1_w, ln1_b;
    size_t Wq, bq, Wk, bk, Wv, bv, Wo, bo;
    size_t ln2_w, ln2_b;
    size_t fc1_w, fc1_b, fc2_w, fc2_b;
  };
  std::vector<LayerOffsets> layers;

  size_t lnf_w, lnf_b;
  size_t total;
};

struct ActivationOffsets {
  size_t embedded;  // [seq_len × d_model]

  struct LayerActivations {
    size_t ln1_out;
    size_t q, k, v;                    // [seq_len × d_model] (after projection)
    size_t q_heads, k_heads, v_heads;  // [n_heads × seq_len × d_head] (reshaped for cuBLAS)
    size_t attn_lse;                   // [B*H × seq_len] softmax row logsumexps (fused path)
    size_t attn_out_heads;             // [n_heads × seq_len × d_head]
    size_t attn_out;                   // [seq_len × d_model] (reshaped back)
    size_t attn_proj;                  // [seq_len × d_model]
    size_t residual1;                  // [seq_len × d_model]
    size_t ln2_out;
    size_t fc1_out;    // [seq_len × d_ff]
    size_t gelu_out;   // [seq_len × d_ff]
    size_t fc2_out;    // [seq_len × d_model]
    size_t residual2;  // [seq_len × d_model]
  };

  // Dropout mask offsets (uint8, separate buffer from float activations). Attention
  // probability masks live in Model::attn_masks with the rest of the materialized path.
  struct DropoutMasks {
    size_t attn_proj;  // after attention output projection
    size_t mlp;        // after MLP fc2
  };
  std::vector<DropoutMasks> drop_masks;
  size_t drop_emb;    // embedding dropout mask
  size_t drop_total;  // total uint8 mask elements
  std::vector<LayerActivations> layers;

  size_t lnf_out;
  size_t logits;  // [seq_len × vocab_size]
  size_t total;
};

enum class AttentionImpl { Auto, Fused, Materialized };

struct Model {
  ModelConfig config;
  bool training = true;
  float dropout_rate = 0.1f;

  // All pointers are GPU device memory
  float* params = nullptr;
  float* activations = nullptr;
  float* gradients = nullptr;
  float* grad_activations = nullptr;
  uint8_t* dropout_masks = nullptr;
  uint8_t* decay_flags = nullptr;  // per-param weight decay flags for AdamW
  uint16_t* d_tokens = nullptr;    // device buffer for input tokens
  float* attn_delta = nullptr;     // [B*H × ctx] fused-attention backward scratch

  // Fused attention (cuda/attention.cu) never stores the [S, S] matrix: at GPT-2's ctx 1024
  // and batch 4 it needs 8.7 GB instead of 11.9 GB, but its fp32 backward is ~1.5x slower
  // than the cuBLAS materialized path. Auto uses fused for ctx_len >= 512. Attention-
  // probability dropout always takes the materialized path, whose [B*H, ctx, ctx] buffers are
  // allocated the first time it runs.
  AttentionImpl attention = AttentionImpl::Auto;
  float* attn_probs = nullptr;    // per layer: softmax P
  float* attn_dropped = nullptr;  // per layer: dropout(P)
  uint8_t* attn_masks = nullptr;  // per layer: dropout mask
  float* attn_grad = nullptr;     // dP for the layer being backpropagated
  float* attn_temp = nullptr;     // d(scores) for the layer being backpropagated
  ParameterOffsets param_offsets;
  ActivationOffsets act_offsets;
  cublasHandle_t cublas_handle = nullptr;

  float* param(size_t offset) { return params + offset; }
  const float* param(size_t offset) const { return params + offset; }
  float* act(size_t offset) { return activations + offset; }
  float* grad(size_t offset) { return gradients + offset; }
  float* grad_act(size_t offset) { return grad_activations + offset; }
  uint8_t* drop_mask(size_t offset) { return dropout_masks + offset; }
};

// Allocate and initialize model
void model_alloc(Model& model, const ModelConfig& config);
void model_free(Model& model);

// Forward pass: tokens[batch_size * seq_len] → logits in activations at act_offsets.logits
void model_forward(Model& model, const uint16_t* tokens, int batch_size, int seq_len);

// Backward pass: compute gradients for all parameters (call after forward + loss)
void model_backward(Model& model, const uint16_t* tokens, int batch_size, int seq_len);

// Zero all gradients
void model_zero_grad(Model& model);

// Save/load
void model_save(const Model& model, const std::string& path);
// batch_size > 0 sizes activation buffers for that batch instead of the saved one
void model_load(Model& model, const std::string& path, int batch_size = 0);

// KV cache for autoregressive generation
struct KVCache {
  // Per-layer K and V: [n_heads, max_seq_len, d_head] each (GPU memory)
  std::vector<float*> k_cache;  // one per layer
  std::vector<float*> v_cache;  // one per layer
  int length = 0;               // current number of cached positions
  int max_len = 0;
  int n_layers = 0;
  int n_heads = 0;
  int d_head = 0;
};

void kv_cache_alloc(KVCache& cache, const ModelConfig& config);
void kv_cache_free(KVCache& cache);

// Cached forward: processes one new position using the KV cache
// tokens points to the single new token (on host), pos is the position index
void model_forward_cached(Model& model, KVCache& cache, uint16_t token, int pos);

// Prefill: runs the whole prompt through one batched forward pass, fills cache positions
// [0, n) from its K/V activations, and leaves the last position's logits where
// model_forward_cached puts them. Equivalent to n model_forward_cached calls, much faster.
void model_prefill(Model& model, KVCache& cache, const uint16_t* tokens, int n);

// Save/load with optimizer state
struct Adam;
void model_save_with_optimizer(const Model& model, const Adam& opt, const std::string& path);
// Allocates opt.m / opt.v if null. Weights-only checkpoints leave the optimizer zeroed.
void model_load_with_optimizer(Model& model, Adam& opt, const std::string& path,
                               int batch_size = 0);
