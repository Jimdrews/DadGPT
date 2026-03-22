#include "model.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "tensor_ops.hpp"

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
    ly.attn_scores = alloc(static_cast<size_t>(c.n_heads) * c.ctx_len * c.ctx_len);
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

void init_weights(Model& model) {
  const auto& c = model.config;
  std::mt19937 rng(42);

  // Xavier uniform for weight matrices
  auto xavier_init = [&](float* w, int fan_in, int fan_out, int count) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (int i = 0; i < count; ++i) {
      w[i] = dist(rng);
    }
  };

  // Normal distribution for embeddings
  auto normal_init = [&](float* w, int count, float std_dev) {
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (int i = 0; i < count; ++i) {
      w[i] = dist(rng);
    }
  };

  // Zero init
  auto zero_init = [](float* w, int count) { std::memset(w, 0, count * sizeof(float)); };

  // Embeddings
  normal_init(model.param(model.param_offsets.tok_emb), c.vocab_size * c.d_model, 0.02f);
  normal_init(model.param(model.param_offsets.pos_emb), c.ctx_len * c.d_model, 0.02f);

  // Per-layer weights
  for (int l = 0; l < c.n_layers; ++l) {
    const auto& ly = model.param_offsets.layers[l];

    // LayerNorm: weight=1, bias=0
    for (int i = 0; i < c.d_model; ++i) {
      model.param(ly.ln1_w)[i] = 1.0f;
      model.param(ly.ln2_w)[i] = 1.0f;
    }
    zero_init(model.param(ly.ln1_b), c.d_model);
    zero_init(model.param(ly.ln2_b), c.d_model);

    // Attention weights
    xavier_init(model.param(ly.Wq), c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(model.param(ly.Wk), c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(model.param(ly.Wv), c.d_model, c.d_model, c.d_model * c.d_model);
    xavier_init(model.param(ly.Wo), c.d_model, c.d_model, c.d_model * c.d_model);
    zero_init(model.param(ly.bq), c.d_model);
    zero_init(model.param(ly.bk), c.d_model);
    zero_init(model.param(ly.bv), c.d_model);
    zero_init(model.param(ly.bo), c.d_model);

    // MLP weights
    xavier_init(model.param(ly.fc1_w), c.d_model, c.d_ff, c.d_model * c.d_ff);
    zero_init(model.param(ly.fc1_b), c.d_ff);
    xavier_init(model.param(ly.fc2_w), c.d_ff, c.d_model, c.d_ff * c.d_model);
    zero_init(model.param(ly.fc2_b), c.d_model);
  }

  // Final layernorm
  for (int i = 0; i < c.d_model; ++i) {
    model.param(model.param_offsets.lnf_w)[i] = 1.0f;
  }
  zero_init(model.param(model.param_offsets.lnf_b), c.d_model);
}

}  // namespace

void model_alloc(Model& model, const ModelConfig& config) {
  model.config = config;
  model.param_offsets = compute_param_offsets(config);
  model.act_offsets = compute_act_offsets(config);

  model.params = new float[model.param_offsets.total]();
  model.activations = new float[model.act_offsets.total]();

  init_weights(model);
}

void model_free(Model& model) {
  delete[] model.params;
  delete[] model.activations;
  model.params = nullptr;
  model.activations = nullptr;
}

void model_forward(Model& model, const uint16_t* tokens, int seq_len) {
  const auto& c = model.config;
  const auto& po = model.param_offsets;
  const auto& ao = model.act_offsets;
  int d = c.d_model;
  int dh = c.d_head();

  // 1. Token embedding + positional embedding
  float* embedded = model.act(ao.embedded);
  embedding_lookup(embedded, tokens, model.param(po.tok_emb), seq_len, d);

  // Add positional embeddings
  const float* pos_emb = model.param(po.pos_emb);
  for (int i = 0; i < seq_len * d; ++i) {
    embedded[i] += pos_emb[i];
  }

  // Current residual stream
  float* residual = embedded;

  // 2. Transformer blocks
  for (int l = 0; l < c.n_layers; ++l) {
    const auto& lp = po.layers[l];
    const auto& la = ao.layers[l];

    // LayerNorm 1
    layernorm(model.act(la.ln1_out), residual, model.param(lp.ln1_w), model.param(lp.ln1_b),
              seq_len, d);

    // Q, K, V projections: [seq_len × d_model] × [d_model × d_model] + bias
    float* q = model.act(la.q);
    float* k = model.act(la.k);
    float* v = model.act(la.v);
    matmul(q, model.act(la.ln1_out), model.param(lp.Wq), seq_len, d, d);
    matmul(k, model.act(la.ln1_out), model.param(lp.Wk), seq_len, d, d);
    matmul(v, model.act(la.ln1_out), model.param(lp.Wv), seq_len, d, d);

    // Add biases
    for (int s = 0; s < seq_len; ++s) {
      add_vectors(q + s * d, q + s * d, model.param(lp.bq), d);
      add_vectors(k + s * d, k + s * d, model.param(lp.bk), d);
      add_vectors(v + s * d, v + s * d, model.param(lp.bv), d);
    }

    // Multi-head attention
    float* attn_scores = model.act(la.attn_scores);
    float* attn_out = model.act(la.attn_out);
    float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    for (int h = 0; h < c.n_heads; ++h) {
      // Extract head slices: Q_h[seq_len × dh], K_h[seq_len × dh], V_h[seq_len × dh]
      // They're interleaved in the full Q/K/V: position s, head h → offset s*d + h*dh

      // Compute attention scores: Q_h × K_h^T → [seq_len × seq_len]
      float* scores = attn_scores + static_cast<size_t>(h) * seq_len * seq_len;

      for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
          float dot = 0.0f;
          for (int x = 0; x < dh; ++x) {
            dot += q[i * d + h * dh + x] * k[j * d + h * dh + x];
          }
          scores[i * seq_len + j] = dot * scale;

          // Causal mask: set future positions to large negative
          if (j > i) {
            scores[i * seq_len + j] = -1e9f;
          }
        }
      }

      // Softmax over each row
      softmax(scores, scores, seq_len, seq_len);

      // Weighted sum of values: attn × V_h → [seq_len × dh]
      for (int i = 0; i < seq_len; ++i) {
        for (int x = 0; x < dh; ++x) {
          float sum = 0.0f;
          for (int j = 0; j < seq_len; ++j) {
            sum += scores[i * seq_len + j] * v[j * d + h * dh + x];
          }
          attn_out[i * d + h * dh + x] = sum;
        }
      }
    }

    // Output projection: attn_out × Wo + bo
    matmul(model.act(la.attn_proj), attn_out, model.param(lp.Wo), seq_len, d, d);
    for (int s = 0; s < seq_len; ++s) {
      add_vectors(model.act(la.attn_proj) + s * d, model.act(la.attn_proj) + s * d,
                  model.param(lp.bo), d);
    }

    // Residual connection 1
    add_vectors(model.act(la.residual1), residual, model.act(la.attn_proj), seq_len * d);

    // LayerNorm 2
    layernorm(model.act(la.ln2_out), model.act(la.residual1), model.param(lp.ln2_w),
              model.param(lp.ln2_b), seq_len, d);

    // MLP: fc1 → gelu → fc2
    matmul(model.act(la.fc1_out), model.act(la.ln2_out), model.param(lp.fc1_w), seq_len, d,
           c.d_ff);
    for (int s = 0; s < seq_len; ++s) {
      add_vectors(model.act(la.fc1_out) + s * c.d_ff, model.act(la.fc1_out) + s * c.d_ff,
                  model.param(lp.fc1_b), c.d_ff);
    }

    gelu(model.act(la.gelu_out), model.act(la.fc1_out), seq_len * c.d_ff);

    matmul(model.act(la.fc2_out), model.act(la.gelu_out), model.param(lp.fc2_w), seq_len, c.d_ff,
           d);
    for (int s = 0; s < seq_len; ++s) {
      add_vectors(model.act(la.fc2_out) + s * d, model.act(la.fc2_out) + s * d,
                  model.param(lp.fc2_b), d);
    }

    // Residual connection 2
    add_vectors(model.act(la.residual2), model.act(la.residual1), model.act(la.fc2_out),
                seq_len * d);

    // Update residual pointer for next layer
    residual = model.act(la.residual2);
  }

  // 3. Final layer norm
  layernorm(model.act(ao.lnf_out), residual, model.param(po.lnf_w), model.param(po.lnf_b),
            seq_len, d);

  // 4. Logits: lnf_out × tok_emb^T (weight tying)
  // lnf_out is [seq_len × d], tok_emb is [vocab × d]
  // out[s][v] = sum_k lnf_out[s][k] * tok_emb[v][k] → matmul with B transposed
  matmul_transposed_b(model.act(ao.logits), model.act(ao.lnf_out), model.param(po.tok_emb),
                      seq_len, d, c.vocab_size);
}

void model_save(const Model& model, const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot write " + path);

  // Write config
  f.write(reinterpret_cast<const char*>(&model.config), sizeof(ModelConfig));

  // Write params
  auto num_params = static_cast<uint64_t>(model.param_offsets.total);
  f.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
  f.write(reinterpret_cast<const char*>(model.params),
          static_cast<std::streamsize>(num_params * sizeof(float)));
}

void model_load(Model& model, const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot read " + path);

  // Read config
  ModelConfig config{};
  f.read(reinterpret_cast<char*>(&config), sizeof(ModelConfig));

  // Allocate model (this also initializes weights, but we'll overwrite them)
  model_alloc(model, config);

  // Read params
  uint64_t num_params = 0;
  f.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
  f.read(reinterpret_cast<char*>(model.params),
         static_cast<std::streamsize>(num_params * sizeof(float)));
}
