#include "generate.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "gpu_ops.cuh"

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
    std::nth_element(sorted_probs.begin(), sorted_probs.begin() + (top_k - 1),
                     sorted_probs.end(), std::greater<>());
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
                                      float temperature, int top_k, int max_len,
                                      float rep_penalty) {
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

  // Process seed one position at a time to populate cache
  for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
    model_forward_cached(model, cache, tokens[i], i);
  }

  while (static_cast<int>(tokens.size()) < max_len) {
    // Copy logits from standard location (set by model_forward_cached)
    gpu_copy_to_host(h_logits.data(), model.act(model.act_offsets.logits), vocab_size);

    uint16_t sampled = sample_token(h_logits, tokens, temperature, top_k, rep_penalty, rng);
    tokens.push_back(sampled);

    // Stop on end token or when the context is full
    if (sampled == BPE::END_TOKEN || static_cast<int>(tokens.size()) >= max_len) break;

    // Process the new token through KV cache
    model_forward_cached(model, cache, sampled, static_cast<int>(tokens.size()) - 1);
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
  model_load(model, config.checkpoint_path, /*batch_size=*/1);  // inference needs one row
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
