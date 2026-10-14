#include "hf_gpt2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_ops.cuh"

namespace {

// Minimal safetensors reader: u64 header length, JSON header, then raw tensor bytes.
class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path) : path_(path), f_(path, std::ios::binary) {
    if (!f_) throw std::runtime_error("Cannot read " + path);
    uint64_t header_len = 0;
    f_.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!f_ || header_len > (100u << 20)) throw std::runtime_error("Bad safetensors header");
    std::string header(header_len, '\0');
    f_.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!f_) throw std::runtime_error("Truncated safetensors header in " + path);
    header_ = nlohmann::json::parse(header);
    data_start_ = sizeof(header_len) + header_len;
  }

  bool has(const std::string& name) const { return header_.contains(name); }

  std::vector<int64_t> shape(const std::string& name) const {
    return entry(name).at("shape").get<std::vector<int64_t>>();
  }

  // Reads an F32 tensor and checks its shape
  std::vector<float> read(const std::string& name, const std::vector<int64_t>& expected) {
    const auto& e = entry(name);
    if (e.at("dtype") != "F32") {
      throw std::runtime_error(name + " is " + e.at("dtype").get<std::string>() +
                               "; only F32 checkpoints are supported");
    }
    if (shape(name) != expected) throw std::runtime_error(name + " has an unexpected shape");
    auto offsets = e.at("data_offsets").get<std::vector<uint64_t>>();
    size_t count = 1;
    for (int64_t dim : expected) count *= static_cast<size_t>(dim);
    if (offsets[1] - offsets[0] != count * sizeof(float)) {
      throw std::runtime_error(name + " byte size does not match its shape");
    }
    std::vector<float> data(count);
    f_.seekg(static_cast<std::streamoff>(data_start_ + offsets[0]));
    f_.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * 4));
    if (!f_) throw std::runtime_error("Truncated tensor " + name + " in " + path_);
    return data;
  }

 private:
  const nlohmann::json& entry(const std::string& name) const {
    if (!header_.contains(name)) throw std::runtime_error("Missing tensor " + name);
    return header_.at(name);
  }

  std::string path_;
  std::ifstream f_;
  nlohmann::json header_;
  uint64_t data_start_ = 0;
};

}  // namespace

void model_load_hf_gpt2(Model& model, const std::string& dir, int batch_size, int ctx_len) {
  std::ifstream cf(dir + "/config.json");
  if (!cf) throw std::runtime_error("Cannot read " + dir + "/config.json");
  nlohmann::json hf = nlohmann::json::parse(cf);
  if (hf.value("activation_function", "gelu_new") != "gelu_new") {
    throw std::runtime_error("Only gelu_new (tanh GELU) GPT-2 checkpoints are supported");
  }

  SafeTensors st(dir + "/model.safetensors");
  auto wte_shape = st.shape("wte.weight");
  auto wpe_shape = st.shape("wpe.weight");

  ModelConfig c;
  c.vocab_size = static_cast<int>(wte_shape[0]);
  c.d_model = static_cast<int>(wte_shape[1]);
  c.n_layers = hf.at("n_layer").get<int>();
  c.n_heads = hf.at("n_head").get<int>();
  c.d_ff = static_cast<int>(st.shape("h.0.mlp.c_fc.weight")[1]);
  int n_positions = static_cast<int>(wpe_shape[0]);
  c.ctx_len = ctx_len > 0 ? ctx_len : n_positions;
  c.batch_size = batch_size;
  if (c.ctx_len > n_positions) {
    throw std::runtime_error("ctx_len exceeds the checkpoint's n_positions");
  }
  if (std::abs(hf.value("layer_norm_epsilon", 1e-5) - 1e-5) > 1e-12) {
    throw std::runtime_error("Only layer_norm_epsilon = 1e-5 is supported");
  }

  model_alloc(model, c);
  const auto& po = model.param_offsets;
  std::vector<float> host(po.total);
  const int64_t d = c.d_model, ff = c.d_ff, V = c.vocab_size;

  auto put = [&](size_t offset, const std::vector<float>& src) {
    std::copy(src.begin(), src.end(), host.begin() + static_cast<std::ptrdiff_t>(offset));
  };

  put(po.tok_emb, st.read("wte.weight", {V, d}));
  auto wpe = st.read("wpe.weight", {n_positions, d});
  wpe.resize(static_cast<size_t>(c.ctx_len) * d);
  put(po.pos_emb, wpe);

  for (int l = 0; l < c.n_layers; ++l) {
    const auto& ly = po.layers[l];
    std::string p = "h." + std::to_string(l) + ".";
    put(ly.ln1_w, st.read(p + "ln_1.weight", {d}));
    put(ly.ln1_b, st.read(p + "ln_1.bias", {d}));

    // c_attn: [d, 3d] = [Wq | Wk | Wv] column blocks; bias [3d] = [bq | bk | bv]
    auto w = st.read(p + "attn.c_attn.weight", {d, 3 * d});
    auto b = st.read(p + "attn.c_attn.bias", {3 * d});
    size_t dst[3] = {ly.Wq, ly.Wk, ly.Wv};
    size_t dst_b[3] = {ly.bq, ly.bk, ly.bv};
    for (int part = 0; part < 3; ++part) {
      for (int64_t row = 0; row < d; ++row) {
        for (int64_t col = 0; col < d; ++col) {
          host[dst[part] + row * d + col] = w[row * 3 * d + part * d + col];
        }
      }
      for (int64_t col = 0; col < d; ++col) host[dst_b[part] + col] = b[part * d + col];
    }

    put(ly.Wo, st.read(p + "attn.c_proj.weight", {d, d}));
    put(ly.bo, st.read(p + "attn.c_proj.bias", {d}));
    put(ly.ln2_w, st.read(p + "ln_2.weight", {d}));
    put(ly.ln2_b, st.read(p + "ln_2.bias", {d}));
    put(ly.fc1_w, st.read(p + "mlp.c_fc.weight", {d, ff}));
    put(ly.fc1_b, st.read(p + "mlp.c_fc.bias", {ff}));
    put(ly.fc2_w, st.read(p + "mlp.c_proj.weight", {ff, d}));
    put(ly.fc2_b, st.read(p + "mlp.c_proj.bias", {d}));
  }
  put(po.lnf_w, st.read("ln_f.weight", {d}));
  put(po.lnf_b, st.read("ln_f.bias", {d}));

  // A separate lm_head would mean the checkpoint is not weight-tied, which this model assumes
  if (st.has("lm_head.weight")) {
    auto lm = st.read("lm_head.weight", {V, d});
    for (size_t i = 0; i < lm.size(); ++i) {
      if (lm[i] != host[po.tok_emb + i]) throw std::runtime_error("lm_head is not tied to wte");
    }
  }

  gpu_copy_to_device(model.params, host.data(), host.size());
}
