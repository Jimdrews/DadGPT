#include "bpe.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

void BPE::init_byte_vocab() {
  id_to_bytes_.clear();
  merges_.clear();

  // Special tokens
  id_to_bytes_[START_TOKEN] = {'<', '|', 's', 't', 'a', 'r', 't', '|', '>'};
  id_to_bytes_[SEP_TOKEN] = {'<', '|', 's', 'e', 'p', '|', '>'};
  id_to_bytes_[END_TOKEN] = {'<', '|', 'e', 'n', 'd', '|', '>'};
  id_to_bytes_[PAD_TOKEN] = {'<', '|', 'p', 'a', 'd', '|', '>'};

  // Single-byte tokens (0-255 mapped to IDs 4-259)
  for (int b = 0; b < 256; ++b) {
    id_to_bytes_[static_cast<uint16_t>(BYTE_OFFSET + b)] = {static_cast<uint8_t>(b)};
  }
}

void BPE::train(const std::string& text, int vocab_size) {
  init_byte_vocab();

  // Use a doubly-linked list for O(1) node removal during merges.
  // Each node holds a token ID; prev/next are indices (-1 = end).
  struct Node {
    uint16_t token;
    int prev;
    int next;
  };

  std::vector<Node> nodes;
  nodes.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    auto token = static_cast<uint16_t>(BYTE_OFFSET + static_cast<uint8_t>(text[i]));
    nodes.push_back({token, static_cast<int>(i) - 1, static_cast<int>(i) + 1});
  }
  if (!nodes.empty()) {
    nodes.back().next = -1;
  }

  // Build initial pair counts
  using Pair = std::pair<uint16_t, uint16_t>;
  std::unordered_map<Pair, int64_t, PairHash> pair_counts;
  for (size_t i = 0; i + 1 < nodes.size(); ++i) {
    if (nodes[i].next != -1) {
      pair_counts[{nodes[i].token, nodes[nodes[i].next].token}]++;
    }
  }

  int current_vocab_size = 260;
  auto next_id = static_cast<uint16_t>(current_vocab_size);

  while (current_vocab_size < vocab_size) {
    if (pair_counts.empty()) break;

    // Find most frequent pair
    Pair best_pair;
    int64_t best_count = 0;
    for (const auto& [pair, cnt] : pair_counts) {
      if (cnt > best_count) {
        best_count = cnt;
        best_pair = pair;
      }
    }

    if (best_count < 2) break;

    // Create merged token
    auto& bytes_a = id_to_bytes_[best_pair.first];
    auto& bytes_b = id_to_bytes_[best_pair.second];
    std::vector<uint8_t> merged_bytes;
    merged_bytes.reserve(bytes_a.size() + bytes_b.size());
    merged_bytes.insert(merged_bytes.end(), bytes_a.begin(), bytes_a.end());
    merged_bytes.insert(merged_bytes.end(), bytes_b.begin(), bytes_b.end());
    id_to_bytes_[next_id] = std::move(merged_bytes);
    merges_.push_back({best_pair.first, best_pair.second, next_id});

    // Walk the linked list, find and merge all occurrences
    // For each merge: update neighbor pair counts incrementally
    int idx = 0;
    while (idx != -1 && static_cast<size_t>(idx) < nodes.size()) {
      int next_idx = nodes[idx].next;
      if (next_idx == -1) break;

      if (nodes[idx].token == best_pair.first && nodes[next_idx].token == best_pair.second) {
        // Decrement old pairs involving these two nodes
        int prev_idx = nodes[idx].prev;
        int after_idx = nodes[next_idx].next;

        // Left neighbor pair: (prev, idx) is disappearing
        if (prev_idx != -1) {
          auto& c = pair_counts[{nodes[prev_idx].token, nodes[idx].token}];
          if (--c <= 0) pair_counts.erase({nodes[prev_idx].token, nodes[idx].token});
        }
        // The pair being merged: (idx, next_idx)
        {
          auto& c = pair_counts[{nodes[idx].token, nodes[next_idx].token}];
          if (--c <= 0) pair_counts.erase({nodes[idx].token, nodes[next_idx].token});
        }
        // Right neighbor pair: (next_idx, after) is disappearing
        if (after_idx != -1) {
          auto& c = pair_counts[{nodes[next_idx].token, nodes[after_idx].token}];
          if (--c <= 0) pair_counts.erase({nodes[next_idx].token, nodes[after_idx].token});
        }

        // Replace idx's token with the merged token
        nodes[idx].token = next_id;

        // Remove next_idx from the list
        nodes[idx].next = after_idx;
        if (after_idx != -1) {
          nodes[after_idx].prev = idx;
        }

        // Add new pairs involving the merged token
        if (prev_idx != -1) {
          pair_counts[{nodes[prev_idx].token, next_id}]++;
        }
        if (after_idx != -1) {
          pair_counts[{next_id, nodes[after_idx].token}]++;
        }

        // Move to the node after the merge (skip the removed node)
        idx = after_idx;
      } else {
        idx = next_idx;
      }
    }

    ++next_id;
    ++current_vocab_size;

    if ((current_vocab_size - 260) % 500 == 0) {
      std::cout << "  BPE merges: " << (current_vocab_size - 260) << " / " << (vocab_size - 260)
                << "\n";
    }
  }
}

std::vector<uint16_t> BPE::encode(const std::string& text) const {
  // Convert text to byte-level tokens
  std::vector<uint16_t> tokens;
  tokens.reserve(text.size());
  for (uint8_t byte : text) {
    tokens.push_back(static_cast<uint16_t>(BYTE_OFFSET + byte));
  }

  // Apply merges in priority order
  for (const auto& merge : merges_) {
    std::vector<uint16_t> new_tokens;
    new_tokens.reserve(tokens.size());
    size_t i = 0;
    while (i < tokens.size()) {
      if (i + 1 < tokens.size() && tokens[i] == merge.a && tokens[i + 1] == merge.b) {
        new_tokens.push_back(merge.merged);
        i += 2;
      } else {
        new_tokens.push_back(tokens[i]);
        ++i;
      }
    }
    tokens = std::move(new_tokens);
  }

  return tokens;
}

std::string BPE::decode(const std::vector<uint16_t>& tokens) const {
  std::string result;
  for (uint16_t token : tokens) {
    auto it = id_to_bytes_.find(token);
    if (it != id_to_bytes_.end()) {
      result.append(it->second.begin(), it->second.end());
    }
  }
  return result;
}

void BPE::save(const std::string& vocab_path, const std::string& merges_path) const {
  // Write vocab.bin
  {
    std::ofstream f(vocab_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write " + vocab_path);

    auto vs = static_cast<uint32_t>(id_to_bytes_.size());
    f.write(reinterpret_cast<const char*>(&vs), sizeof(vs));

    for (const auto& [id, bytes] : id_to_bytes_) {
      f.write(reinterpret_cast<const char*>(&id), sizeof(id));
      auto len = static_cast<uint16_t>(bytes.size());
      f.write(reinterpret_cast<const char*>(&len), sizeof(len));
      f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
  }

  // Write merges.bin
  {
    std::ofstream f(merges_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write " + merges_path);

    auto num_merges = static_cast<uint32_t>(merges_.size());
    f.write(reinterpret_cast<const char*>(&num_merges), sizeof(num_merges));

    for (const auto& merge : merges_) {
      f.write(reinterpret_cast<const char*>(&merge.a), sizeof(merge.a));
      f.write(reinterpret_cast<const char*>(&merge.b), sizeof(merge.b));
      f.write(reinterpret_cast<const char*>(&merge.merged), sizeof(merge.merged));
    }
  }
}

void BPE::load(const std::string& vocab_path, const std::string& merges_path) {
  id_to_bytes_.clear();
  merges_.clear();

  // Read vocab.bin
  {
    std::ifstream f(vocab_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read " + vocab_path);

    uint32_t vs = 0;
    f.read(reinterpret_cast<char*>(&vs), sizeof(vs));

    for (uint32_t i = 0; i < vs; ++i) {
      uint16_t id = 0;
      uint16_t len = 0;
      f.read(reinterpret_cast<char*>(&id), sizeof(id));
      f.read(reinterpret_cast<char*>(&len), sizeof(len));
      std::vector<uint8_t> bytes(len);
      f.read(reinterpret_cast<char*>(bytes.data()), len);
      id_to_bytes_[id] = std::move(bytes);
    }
  }

  // Read merges.bin
  {
    std::ifstream f(merges_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read " + merges_path);

    uint32_t num_merges = 0;
    f.read(reinterpret_cast<char*>(&num_merges), sizeof(num_merges));

    merges_.resize(num_merges);
    for (uint32_t i = 0; i < num_merges; ++i) {
      f.read(reinterpret_cast<char*>(&merges_[i].a), sizeof(uint16_t));
      f.read(reinterpret_cast<char*>(&merges_[i].b), sizeof(uint16_t));
      f.read(reinterpret_cast<char*>(&merges_[i].merged), sizeof(uint16_t));
    }
  }
}
