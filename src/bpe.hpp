#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class BPE {
 public:
  static constexpr uint16_t START_TOKEN = 0;
  static constexpr uint16_t SEP_TOKEN = 1;
  static constexpr uint16_t END_TOKEN = 2;
  static constexpr uint16_t PAD_TOKEN = 3;
  static constexpr uint16_t BYTE_OFFSET = 4;

  void train(const std::string& text, int vocab_size);
  std::vector<uint16_t> encode(const std::string& text) const;
  std::string decode(const std::vector<uint16_t>& tokens) const;

  void save(const std::string& vocab_path, const std::string& merges_path) const;
  void load(const std::string& vocab_path, const std::string& merges_path);

  int vocab_size() const { return static_cast<int>(id_to_bytes_.size()); }

 private:
  struct PairHash {
    size_t operator()(const std::pair<uint16_t, uint16_t>& p) const {
      return std::hash<uint32_t>{}(
          (static_cast<uint32_t>(p.first) << 16) | p.second);
    }
  };

  // token_id → byte string
  std::unordered_map<uint16_t, std::vector<uint8_t>> id_to_bytes_;

  // Ordered merge rules: (token_a, token_b) → merged_token_id
  struct MergeRule {
    uint16_t a;
    uint16_t b;
    uint16_t merged;
  };
  std::vector<MergeRule> merges_;

  void init_byte_vocab();
};
