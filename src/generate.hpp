#pragma once

#include <random>
#include <string>
#include <vector>

#include "bpe.hpp"
#include "model.hpp"

struct GenerateConfig {
  std::string checkpoint_path = "data/checkpoint_epoch_20.bin";
  std::string vocab_path = "data/vocab.bin";
  std::string merges_path = "data/merges.bin";
  float temperature = 0.8f;
  float rep_penalty = 1.0f;
  int top_k = 40;
  int max_len = 128;
  int count = 1;
  std::string prompt;
};

// Pick the next token from raw logits. Applies the repetition penalty once per distinct token
// in history (CTRL-style), then temperature (<= 0 means greedy argmax), then top-k (0 = off).
uint16_t sample_token(std::vector<float> logits, const std::vector<uint16_t>& history,
                      float temperature, int top_k, float rep_penalty, std::mt19937& rng);

// Generate a sequence of tokens from the model (uses KV cache). Seeds longer than the model's
// context are truncated to START + their most recent tokens; max_len is capped at ctx_len.
std::vector<uint16_t> generate_tokens(Model& model, const std::vector<uint16_t>& seed,
                                      float temperature, int top_k, int max_len,
                                      float rep_penalty = 1.0f);

// Generate text: run generate_tokens, decode with BPE, format output
std::string generate_text(Model& model, const BPE& bpe, const GenerateConfig& config);

// Run the full generate CLI command
void run_generate(const GenerateConfig& config);
