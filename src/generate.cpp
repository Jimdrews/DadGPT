#include "generate.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "gpt2_tokenizer.hpp"
#include "gpu_ops.cuh"
#include "hf_gpt2.hpp"

uint16_t sample_token(std::vector<float> logits, const std::vector<uint16_t>& history,
                      float temperature, int top_k, float rep_penalty, std::mt19937& rng) {
  int vocab_size = static_cast<int>(logits.size());

  // Repetition penalty: each distinct token already in the sequence is penalized once
  if (rep_penalty != 1.0f) {
    std::vector<bool> seen(vocab_size, false);
    for (uint16_t tok : history) {
      if (tok >= vocab_size || seen[tok]) continue;
      seen[tok] = true;
      if (logits[tok] > 0.0f) {
        logits[tok] /= rep_penalty;
      } else {
        logits[tok] *= rep_penalty;
      }
    }
  }

  // Greedy decoding
  if (temperature <= 0.0f || top_k == 1) {
    return static_cast<uint16_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }

  // Temperature scaling
  if (temperature != 1.0f) {
    for (float& logit : logits) {
      logit /= temperature;
    }
  }

  // Softmax
  float max_val = *std::max_element(logits.begin(), logits.end());
  float sum = 0.0f;
  for (float& v : logits) {
    v = std::exp(v - max_val);
    sum += v;
  }
  for (float& v : logits) {
    v /= sum;
  }

  // Top-k filtering: keep probabilities >= the k-th largest (ties at the boundary are kept)
  if (top_k > 0 && top_k < vocab_size) {
    std::vector<float> sorted_probs = logits;
    std::nth_element(sorted_probs.begin(), sorted_probs.begin() + (top_k - 1), sorted_probs.end(),
                     std::greater<>());
    float threshold = sorted_probs[top_k - 1];

    float filtered_sum = 0.0f;
    for (float& v : logits) {
      if (v < threshold) {
        v = 0.0f;
      } else {
        filtered_sum += v;
      }
    }
    if (filtered_sum > 0.0f) {
      for (float& v : logits) {
        v /= filtered_sum;
      }
    }
  }

  // Sample from distribution. If rounding leaves the cumulative sum just short of r,
  // fall back to the last token that had nonzero probability.
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);
  float cumulative = 0.0f;
  int last_nonzero = 0;
  for (int v = 0; v < vocab_size; ++v) {
    if (logits[v] <= 0.0f) continue;
    last_nonzero = v;
    cumulative += logits[v];
    if (cumulative >= r) return static_cast<uint16_t>(v);
  }
  return static_cast<uint16_t>(last_nonzero);
}

std::vector<uint16_t> generate_tokens(Model& model, const std::vector<uint16_t>& seed,
                                      float temperature, int top_k, int max_len, float rep_penalty,
                                      int stop_token) {
  int ctx_len = model.config.ctx_len;
  max_len = std::min(max_len, ctx_len);

  // Leave room for at least one generated token: keep START plus the most recent tokens
  std::vector<uint16_t> tokens = seed;
  if (static_cast<int>(tokens.size()) >= max_len && tokens.size() > 1) {
    size_t keep = static_cast<size_t>(std::max(max_len - 2, 0));
    std::cerr << std::format("Prompt has {} tokens; truncating to the last {} (context {})\n",
                             tokens.size(), keep, ctx_len);
    std::vector<uint16_t> truncated{tokens.front()};
    truncated.insert(truncated.end(), tokens.end() - static_cast<std::ptrdiff_t>(keep),
                     tokens.end());
    tokens = std::move(truncated);
  }

  std::mt19937 rng(std::random_device{}());
  int vocab_size = model.config.vocab_size;
  std::vector<float> h_logits(vocab_size);

  // Initialize KV cache
  KVCache cache;
  kv_cache_alloc(cache, model.config);

  // One batched forward pass over the seed populates the cache
  model_prefill(model, cache, tokens.data(), static_cast<int>(tokens.size()));

  while (static_cast<int>(tokens.size()) < max_len) {
    // Copy logits from standard location (set by model_forward_cached)
    gpu_copy_to_host(h_logits.data(), model.act(model.act_offsets.logits), vocab_size);

    uint16_t sampled = sample_token(h_logits, tokens, temperature, top_k, rep_penalty, rng);
    tokens.push_back(sampled);

    // Stop on end token or when the context is full
    if (sampled == stop_token || static_cast<int>(tokens.size()) >= max_len) break;

    // Process the new token through KV cache
    model_forward_cached(model, cache, sampled, static_cast<int>(tokens.size()) - 1);
  }

  kv_cache_free(cache);
  return tokens;
}

namespace {

void trim(std::string& text) {
  auto start = text.find_first_not_of(" \n\r\t");
  auto end = text.find_last_not_of(" \n\r\t");
  text = start == std::string::npos ? std::string() : text.substr(start, end - start + 1);
}

// Dad-jokes BPE: START question SEP response END
class DadBpeCodec : public TextCodec {
 public:
  explicit DadBpeCodec(const std::string& data_dir) {
    bpe_.load(data_dir + "/vocab.bin", data_dir + "/merges.bin");
  }
  std::vector<uint16_t> seed(const std::string& prompt) const override {
    std::vector<uint16_t> seed{BPE::START_TOKEN};
    if (!prompt.empty()) {
      auto prompt_tokens = bpe_.encode(prompt);
      seed.insert(seed.end(), prompt_tokens.begin(), prompt_tokens.end());
      seed.push_back(BPE::SEP_TOKEN);
    }
    return seed;
  }
  std::string render(const std::vector<uint16_t>& tokens) const override {
    std::string text;
    for (uint16_t t : tokens) {
      if (t == BPE::START_TOKEN || t == BPE::END_TOKEN || t == BPE::PAD_TOKEN) continue;
      text += t == BPE::SEP_TOKEN ? std::string("\n") : bpe_.decode({t});
    }
    trim(text);
    return text;
  }
  int stop_token() const override { return BPE::END_TOKEN; }
  std::string name() const override { return std::format("dadbpe ({} tokens)", bpe_.vocab_size()); }

 private:
  BPE bpe_;
};

// GPT-2 BPE: documents separated by <|endoftext|>; prepare writes jokes as "question\nresponse"
class Gpt2Codec : public TextCodec {
 public:
  explicit Gpt2Codec(const std::string& dir) {
    tok_.load(dir + "/vocab.json", dir + "/merges.txt");
  }
  std::vector<uint16_t> seed(const std::string& prompt) const override {
    std::vector<uint16_t> seed{Gpt2Tokenizer::EOT};
    if (!prompt.empty()) {
      auto ids = tok_.encode(prompt + "\n");
      seed.insert(seed.end(), ids.begin(), ids.end());
    }
    return seed;
  }
  std::string render(const std::vector<uint16_t>& tokens) const override {
    std::vector<uint16_t> body;
    for (uint16_t t : tokens) {
      if (t != Gpt2Tokenizer::EOT) body.push_back(t);
    }
    std::string text = tok_.decode(body);
    trim(text);
    return text;
  }
  int stop_token() const override { return Gpt2Tokenizer::EOT; }
  std::string name() const override { return "gpt2 (50257 tokens)"; }

 private:
  Gpt2Tokenizer tok_;
};

}  // namespace

std::unique_ptr<TextCodec> make_codec(const DatasetMeta& meta, const std::string& data_dir) {
  if (meta.tokenizer == "gpt2") return std::make_unique<Gpt2Codec>(meta.tokenizer_dir);
  if (meta.tokenizer == "dadbpe") return std::make_unique<DadBpeCodec>(data_dir);
  throw std::runtime_error("Unknown tokenizer '" + meta.tokenizer + "' in meta.txt");
}

std::string generate_text(Model& model, const TextCodec& codec, const GenerateConfig& config) {
  int max_len = config.max_len > 0 ? config.max_len : model.config.ctx_len;
  auto tokens = generate_tokens(model, codec.seed(config.prompt), config.temperature, config.top_k,
                                max_len, config.rep_penalty, codec.stop_token());
  return codec.render(tokens);
}

void run_generate(const GenerateConfig& config) {
  std::cout << "=== dad-gpt generate ===\n\n";

  Model model{};
  std::unique_ptr<TextCodec> codec;
  if (!config.hf_dir.empty()) {
    // Pretrained Hugging Face GPT-2 straight from config.json + model.safetensors
    model_load_hf_gpt2(model, config.hf_dir, /*batch_size=*/1);
    codec = std::make_unique<Gpt2Codec>(config.hf_dir);
    std::cout << std::format("Loaded Hugging Face GPT-2 from {}\n", config.hf_dir);
  } else {
    DatasetMeta meta = read_dataset_meta(config.data_dir);
    codec = make_codec(meta, config.data_dir);
    model_load(model, config.checkpoint_path, /*batch_size=*/1);  // inference needs one row
    if (model.config.vocab_size != meta.vocab_size) {
      throw std::runtime_error(std::format("Checkpoint vocab {} != tokenizer vocab {}",
                                           model.config.vocab_size, meta.vocab_size));
    }
    std::cout << std::format("Loaded checkpoint: {}\n", config.checkpoint_path);
  }
  model.training = false;
  std::cout << std::format("Tokenizer: {}\n", codec->name());
  std::cout << std::format("Temperature: {}, Top-k: {}\n\n", config.temperature, config.top_k);

  for (int i = 0; i < config.count; ++i) {
    std::string text = generate_text(model, *codec, config);
    if (config.count > 1) {
      std::cout << std::format("[{}/{}] ", i + 1, config.count);
    }
    std::cout << text << "\n\n";
  }

  model_free(model);
}
