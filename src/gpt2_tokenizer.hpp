#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// OpenAI GPT-2 byte-level BPE (50257 tokens), loaded from the standard vocab.json + merges.txt
// (e.g. huggingface.co/openai-community/gpt2). Encoding matches OpenAI's encoder.py / HF
// GPT2Tokenizer: regex pre-tokenization, byte -> unicode mapping, then rank-ordered merges
// within each pre-token.
class Gpt2Tokenizer {
 public:
  static constexpr uint16_t EOT = 50256;  // <|endoftext|>

  Gpt2Tokenizer();
  ~Gpt2Tokenizer();
  Gpt2Tokenizer(const Gpt2Tokenizer&) = delete;
  Gpt2Tokenizer& operator=(const Gpt2Tokenizer&) = delete;

  void load(const std::string& vocab_json_path, const std::string& merges_path);

  // Plain text only: a literal "<|endoftext|>" in the input is encoded as ordinary text.
  // Insert EOT ids yourself to separate documents.
  std::vector<uint16_t> encode(const std::string& text) const;
  std::string decode(const std::vector<uint16_t>& ids) const;
  int vocab_size() const { return static_cast<int>(decoder_.size()); }

 private:
  // Applies the BPE merges to one pre-token given as mapped unicode symbols
  void bpe(const std::vector<std::string>& symbols, std::vector<uint16_t>& out) const;

  std::array<std::string, 256> byte_encoder_;         // byte -> UTF-8 of its symbol
  std::unordered_map<std::string, int> encoder_;      // symbol string -> id
  std::vector<std::string> decoder_;                  // id -> raw bytes
  std::unordered_map<std::string, int> merge_ranks_;  // "left right" -> rank
  // Pre-token -> ids. Mutable cache: encode() is logically const but not thread-safe.
  mutable std::unordered_map<std::string, std::vector<uint16_t>> cache_;
  struct Regex;
  std::unique_ptr<Regex> regex_;
};
