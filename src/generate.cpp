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
                                      float temperature, int top_k, int max_len) {
  std::vector<uint16_t> tokens = seed;
  std::mt19937 rng(std::random_device{}());
  int vocab_size = model.config.vocab_size;
  std::vector<float> h_logits(vocab_size);

  while (static_cast<int>(tokens.size()) < max_len) {
    int seq_len = static_cast<int>(tokens.size());

    // Forward pass on current sequence
    gpu_zero(model.activations, model.act_offsets.total);
    model_forward(model, tokens.data(), seq_len);

    // Copy last position's logits from GPU to host
    gpu_copy_to_host(h_logits.data(),
                     model.act(model.act_offsets.logits) + (seq_len - 1) * vocab_size, vocab_size);

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
      // Find the k-th largest probability
      std::vector<float> sorted_probs = h_logits;
      std::nth_element(sorted_probs.begin(), sorted_probs.begin() + top_k, sorted_probs.end(),
                       std::greater<>());
      float threshold = sorted_probs[top_k];

      // Zero out everything below threshold
      float filtered_sum = 0.0f;
      for (float& v : h_logits) {
        if (v < threshold) {
          v = 0.0f;
        } else {
          filtered_sum += v;
        }
      }

      // Renormalize
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
  }

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
  auto tokens = generate_tokens(model, seed, config.temperature, config.top_k, config.max_len);

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
