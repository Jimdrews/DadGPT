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

  int d_head() const { return d_model / n_heads; }
};

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
    size_t q, k, v;              // [seq_len × d_model] (after projection)
    size_t q_heads, k_heads, v_heads;  // [n_heads × seq_len × d_head] (reshaped for cuBLAS)
    size_t attn_scores;          // [n_heads × seq_len × seq_len]
    size_t attn_out_heads;       // [n_heads × seq_len × d_head]
    size_t attn_out;             // [seq_len × d_model] (reshaped back)
    size_t attn_proj;            // [seq_len × d_model]
    size_t residual1;            // [seq_len × d_model]
    size_t ln2_out;
    size_t fc1_out;              // [seq_len × d_ff]
    size_t gelu_out;             // [seq_len × d_ff]
    size_t fc2_out;              // [seq_len × d_model]
    size_t residual2;            // [seq_len × d_model]
  };
  std::vector<LayerActivations> layers;

  size_t lnf_out;
  size_t logits;  // [seq_len × vocab_size]
  size_t total;
};

struct Model {
  ModelConfig config;
  // All pointers are GPU device memory
  float* params = nullptr;
  float* activations = nullptr;
  float* gradients = nullptr;
  float* grad_activations = nullptr;
  float* temp_scores = nullptr;  // pre-allocated temp buffer for backward attention
  uint16_t* d_tokens = nullptr;  // device buffer for input tokens
  ParameterOffsets param_offsets;
  ActivationOffsets act_offsets;
  cublasHandle_t cublas_handle = nullptr;

  float* param(size_t offset) { return params + offset; }
  const float* param(size_t offset) const { return params + offset; }
  float* act(size_t offset) { return activations + offset; }
  float* grad(size_t offset) { return gradients + offset; }
  float* grad_act(size_t offset) { return grad_activations + offset; }
};

// Allocate and initialize model
void model_alloc(Model& model, const ModelConfig& config);
void model_free(Model& model);

// Forward pass: tokens[seq_len] → logits stored in model.activations at act_offsets.logits
void model_forward(Model& model, const uint16_t* tokens, int seq_len);

// Backward pass: compute gradients for all parameters (call after forward + loss)
void model_backward(Model& model, const uint16_t* tokens, int seq_len);

// Zero all gradients
void model_zero_grad(Model& model);

// Save/load
void model_save(const Model& model, const std::string& path);
void model_load(Model& model, const std::string& path);
