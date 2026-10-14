#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bpe.hpp"
#include "dataloader.hpp"
#include "model.hpp"

struct GenerateConfig {
  std::string data_dir = "data";  // meta.txt and tokenizer files live here
  std::string checkpoint_path;
  std::string hf_dir;  // if set, sample from pretrained HF GPT-2 in this directory instead
  float temperature = 0.8f;
  float rep_penalty = 1.0f;
  int top_k = 40;
  int max_len = 0;  // total tokens including the prompt; 0 = the model's context length
  int count = 1;
  std::string prompt;
};

// How a dataset's tokenizer frames a prompt and turns generated tokens back into text
class TextCodec {
 public:
  virtual ~TextCodec() = default;
  virtual std::vector<uint16_t> seed(const std::string& prompt) const = 0;
  virtual std::string render(const std::vector<uint16_t>& tokens) const = 0;
  virtual int stop_token() const = 0;
  virtual std::string name() const = 0;
};

// The codec for the tokenizer recorded in a data directory's meta.txt
std::unique_ptr<TextCodec> make_codec(const DatasetMeta& meta, const std::string& data_dir);

// Pick the next token from raw logits. Applies the repetition penalty once per distinct token
// in history (CTRL-style), then temperature (<= 0 means greedy argmax), then top-k (0 = off).
uint16_t sample_token(std::vector<float> logits, const std::vector<uint16_t>& history,
                      float temperature, int top_k, float rep_penalty, std::mt19937& rng);

// Generate a sequence of tokens from the model (uses KV cache), stopping after stop_token.
// Seeds longer than the model's context keep their first token plus their most recent
// tokens; max_len is capped at ctx_len.
std::vector<uint16_t> generate_tokens(Model& model, const std::vector<uint16_t>& seed,
                                      float temperature, int top_k, int max_len,
                                      float rep_penalty = 1.0f, int stop_token = BPE::END_TOKEN);

// Seed from the prompt, generate, and render the result as text
std::string generate_text(Model& model, const TextCodec& codec, const GenerateConfig& config);

// Run the full generate CLI command
void run_generate(const GenerateConfig& config);
