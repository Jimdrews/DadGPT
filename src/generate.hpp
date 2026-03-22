#pragma once

#include <string>
#include <vector>

#include "bpe.hpp"
#include "model.hpp"

struct GenerateConfig {
  std::string checkpoint_path = "data/checkpoint_epoch_20.bin";
  std::string vocab_path = "data/vocab.bin";
  std::string merges_path = "data/merges.bin";
  float temperature = 0.8f;
  int top_k = 40;
  int max_len = 128;
  int count = 1;
  std::string prompt;
};

// Generate a sequence of tokens from the model
std::vector<uint16_t> generate_tokens(Model& model, const std::vector<uint16_t>& seed,
                                      float temperature, int top_k, int max_len);

// Generate text: run generate_tokens, decode with BPE, format output
std::string generate_text(Model& model, const BPE& bpe, const GenerateConfig& config);

// Run the full generate CLI command
void run_generate(const GenerateConfig& config);
