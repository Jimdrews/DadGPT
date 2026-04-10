#include "generate.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "gpu_ops.cuh"

std::vector<uint16_t> generate_tokens(Model& model, const std::vector<uint16_t>& seed,
                                      float temperature, int top_k, int max_len,
                                      float rep_penalty) {
  std::vector<uint16_t> tokens = seed;
  std::mt19937 rng(std::random_device{}());
  int vocab_size = model.config.vocab_size;
  std::vector<float> h_logits(vocab_size);

  // Initialize KV cache
  KVCache cache;
  kv_cache_alloc(cache, model.config);

  // Process seed with full forward pass to populate cache
  for (int i = 0; i < static_cast<int>(seed.size()); ++i) {
    model_forward_cached(model, cache, seed[i], i);
  }

  while (static_cast<int>(tokens.size()) < max_len) {
    // Copy logits from standard location (set by model_forward_cached)
    gpu_copy_to_host(h_logits.data(), model.act(model.act_offsets.logits), vocab_size);

    // Repetition penalty: penalize tokens already in sequence
    if (rep_penalty != 1.0f) {
      for (uint16_t tok : tokens) {
        if (h_logits[tok] > 0.0f) {
          h_logits[tok] /= rep_penalty;
        } else {
          h_logits[tok] *= rep_penalty;
        }
      }
    }

    // Temperature scaling
    if (temperature != 1.0f) {
      for (float& logit : h_logits) {
        logit /= temperature;
      }
    }

    // Softmax
    float max_val = *std::max_element(h_logits.begin(), h_logits.end());
    float sum = 0.0f;
    for (float& v : h_logits) {
      v = std::exp(v - max_val);
      sum += v;
    }
    for (float& v : h_logits) {
      v /= sum;
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
      std::vector<float> sorted_probs = h_logits;
      std::nth_element(sorted_probs.begin(), sorted_probs.begin() + top_k, sorted_probs.end(),
                       std::greater<>());
      float threshold = sorted_probs[top_k];

      float filtered_sum = 0.0f;
      for (float& v : h_logits) {
        if (v < threshold) {
          v = 0.0f;
        } else {
          filtered_sum += v;
        }
      }
      if (filtered_sum > 0.0f) {
        for (float& v : h_logits) {
          v /= filtered_sum;
        }
      }
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumulative = 0.0f;
    uint16_t sampled = 0;
    for (int v = 0; v < vocab_size; ++v) {
      cumulative += h_logits[v];
      if (cumulative >= r) {
        sampled = static_cast<uint16_t>(v);
        break;
      }
    }

    tokens.push_back(sampled);

    // Stop on end token
    if (sampled == BPE::END_TOKEN) break;

    // Process the new token through KV cache
    int pos = static_cast<int>(tokens.size()) - 1;
    if (pos < max_len) {
      model_forward_cached(model, cache, sampled, pos);
    }
  }

  kv_cache_free(cache);
  return tokens;
}

std::string generate_text(Model& model, const BPE& bpe, const GenerateConfig& config) {
  // Build seed sequence
  std::vector<uint16_t> seed;
  seed.push_back(BPE::START_TOKEN);

  if (!config.prompt.empty()) {
    auto prompt_tokens = bpe.encode(config.prompt);
    seed.insert(seed.end(), prompt_tokens.begin(), prompt_tokens.end());
    seed.push_back(BPE::SEP_TOKEN);
  }

  // Generate
  auto tokens = generate_tokens(model, seed, config.temperature, config.top_k, config.max_len,
                                config.rep_penalty);

  // Decode
  std::string text = bpe.decode(tokens);

  // Strip special tokens and format
  auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.length(), to);
      pos += to.length();
    }
  };

  replace_all(text, "<|start|>", "");
  replace_all(text, "<|end|>", "");
  replace_all(text, "<|sep|>", "\n");

  // Trim leading/trailing whitespace
  auto start = text.find_first_not_of(" \n\r\t");
  auto end = text.find_last_not_of(" \n\r\t");
  if (start != std::string::npos) {
    text = text.substr(start, end - start + 1);
  }

  return text;
}

void run_generate(const GenerateConfig& config) {
  std::cout << "=== dad-gpt generate ===\n\n";

  // Load tokenizer
  BPE bpe;
  bpe.load(config.vocab_path, config.merges_path);
  std::cout << std::format("Loaded tokenizer (vocab_size={})\n", bpe.vocab_size());

  // Load model
  Model model{};
  model_load(model, config.checkpoint_path);
  model.training = false;
  std::cout << std::format("Loaded checkpoint: {}\n", config.checkpoint_path);
  std::cout << std::format("Temperature: {}, Top-k: {}\n\n", config.temperature, config.top_k);

  for (int i = 0; i < config.count; ++i) {
    std::string joke = generate_text(model, bpe, config);
    if (config.count > 1) {
      std::cout << std::format("[{}/{}] ", i + 1, config.count);
    }
    std::cout << joke << "\n\n";
  }

  model_free(model);
}
