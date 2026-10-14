#include "gpt2_tokenizer.hpp"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <climits>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace {

// GPT-2's pre-tokenizer: contractions, optional-space + letters / numbers / other, whitespace
constexpr const char* kPattern =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

std::string utf8_encode(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += static_cast<char>(cp);
  } else if (cp < 0x800) {
    s += static_cast<char>(0xC0 | (cp >> 6));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    s += static_cast<char>(0xE0 | (cp >> 12));
    s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return s;
}

// Decodes the next code point of a valid UTF-8 string starting at i, advancing i
uint32_t utf8_next(const std::string& s, size_t& i) {
  auto c = static_cast<unsigned char>(s[i]);
  int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
  uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
  for (int k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
  i += len;
  return cp;
}

// OpenAI's bytes_to_unicode: printable bytes map to themselves, the rest to 256 + n, so every
// byte has a visible, non-whitespace symbol and BPE never sees raw control bytes or spaces.
std::array<uint32_t, 256> bytes_to_unicode() {
  std::array<uint32_t, 256> table{};
  std::array<bool, 256> printable{};
  for (int b = '!'; b <= '~'; ++b) printable[b] = true;
  for (int b = 0xA1; b <= 0xAC; ++b) printable[b] = true;
  for (int b = 0xAE; b <= 0xFF; ++b) printable[b] = true;
  int n = 0;
  for (int b = 0; b < 256; ++b) table[b] = printable[b] ? b : 256 + n++;
  return table;
}

std::string merge_key(const std::string& a, const std::string& b) { return a + " " + b; }

}  // namespace

struct Gpt2Tokenizer::Regex {
  pcre2_code* code = nullptr;
  pcre2_match_data* match = nullptr;
  ~Regex() {
    if (match) pcre2_match_data_free(match);
    if (code) pcre2_code_free(code);
  }
};

Gpt2Tokenizer::Gpt2Tokenizer() : regex_(std::make_unique<Regex>()) {
  int err = 0;
  PCRE2_SIZE err_offset = 0;
  regex_->code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(kPattern), PCRE2_ZERO_TERMINATED,
                               PCRE2_UTF | PCRE2_UCP | PCRE2_MATCH_INVALID_UTF, &err,
                               &err_offset, nullptr);
  if (!regex_->code) {
    PCRE2_UCHAR msg[256];
    pcre2_get_error_message(err, msg, sizeof(msg));
    throw std::runtime_error(std::string("GPT-2 regex: ") + reinterpret_cast<char*>(msg));
  }
  pcre2_jit_compile(regex_->code, PCRE2_JIT_COMPLETE);  // optional speedup; falls back if absent
  regex_->match = pcre2_match_data_create_from_pattern(regex_->code, nullptr);

  auto table = bytes_to_unicode();
  for (int b = 0; b < 256; ++b) byte_encoder_[b] = utf8_encode(table[b]);
}

Gpt2Tokenizer::~Gpt2Tokenizer() = default;

void Gpt2Tokenizer::load(const std::string& vocab_json_path, const std::string& merges_path) {
  std::ifstream vf(vocab_json_path, std::ios::binary);
  if (!vf) throw std::runtime_error("Cannot read " + vocab_json_path);
  nlohmann::json vocab = nlohmann::json::parse(vf);

  // Reverse the byte -> symbol map to turn each vocab entry back into raw bytes
  std::unordered_map<uint32_t, char> byte_decoder;
  auto table = bytes_to_unicode();
  for (int b = 0; b < 256; ++b) byte_decoder[table[b]] = static_cast<char>(b);

  encoder_.clear();
  decoder_.assign(vocab.size(), std::string());
  for (const auto& [symbols, id_json] : vocab.items()) {
    int id = id_json.get<int>();
    if (id < 0 || id >= static_cast<int>(vocab.size()) || id > UINT16_MAX) {
      throw std::runtime_error("vocab.json id out of range: " + std::to_string(id));
    }
    encoder_[symbols] = id;
    std::string bytes;
    for (size_t i = 0; i < symbols.size();) {
      auto it = byte_decoder.find(utf8_next(symbols, i));
      if (it == byte_decoder.end()) throw std::runtime_error("vocab.json symbol not byte-mapped");
      bytes += it->second;
    }
    decoder_[id] = std::move(bytes);
  }

  std::ifstream mf(merges_path, std::ios::binary);
  if (!mf) throw std::runtime_error("Cannot read " + merges_path);
  merge_ranks_.clear();
  std::string line;
  int rank = 0;
  while (std::getline(mf, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.starts_with("#version")) continue;
    auto space = line.find(' ');
    if (space == std::string::npos) throw std::runtime_error("Bad merges line: " + line);
    merge_ranks_[line] = rank++;
  }
  cache_.clear();
}

void Gpt2Tokenizer::bpe(const std::vector<std::string>& symbols,
                        std::vector<uint16_t>& out) const {
  std::vector<std::string> word = symbols;
  // Repeatedly merge the lowest-ranked adjacent pair (all its occurrences, left to right),
  // exactly as OpenAI's encoder.py does
  while (word.size() > 1) {
    int best_rank = INT_MAX;
    size_t best = 0;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
      auto it = merge_ranks_.find(merge_key(word[i], word[i + 1]));
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best_rank == INT_MAX) break;

    std::string first = word[best], second = word[best + 1];
    std::vector<std::string> merged;
    merged.reserve(word.size());
    for (size_t i = 0; i < word.size();) {
      if (i + 1 < word.size() && word[i] == first && word[i + 1] == second) {
        merged.push_back(first + second);
        i += 2;
      } else {
        merged.push_back(word[i]);
        ++i;
      }
    }
    word = std::move(merged);
  }
  for (const auto& sym : word) {
    auto it = encoder_.find(sym);
    if (it == encoder_.end()) throw std::runtime_error("BPE produced a symbol not in vocab.json");
    out.push_back(static_cast<uint16_t>(it->second));
  }
}

std::vector<uint16_t> Gpt2Tokenizer::encode(const std::string& text) const {
  if (encoder_.empty()) throw std::runtime_error("Gpt2Tokenizer::encode before load");
  std::vector<uint16_t> out;
  auto subject = reinterpret_cast<PCRE2_SPTR>(text.data());
  PCRE2_SIZE offset = 0;
  while (offset < text.size()) {
    int rc = pcre2_match(regex_->code, subject, text.size(), offset, 0, regex_->match, nullptr);
    if (rc < 0) break;  // no further match (only possible in invalid UTF-8 tails)
    PCRE2_SIZE* ov = pcre2_get_ovector_pointer(regex_->match);
    if (ov[1] == ov[0]) {
      offset = ov[1] + 1;
      continue;
    }
    std::string piece = text.substr(ov[0], ov[1] - ov[0]);
    offset = ov[1];

    auto cached = cache_.find(piece);
    if (cached != cache_.end()) {
      out.insert(out.end(), cached->second.begin(), cached->second.end());
      continue;
    }
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());
    for (unsigned char b : piece) symbols.push_back(byte_encoder_[b]);
    std::vector<uint16_t> ids;
    bpe(symbols, ids);
    out.insert(out.end(), ids.begin(), ids.end());
    if (cache_.size() < 100000) cache_.emplace(std::move(piece), std::move(ids));
  }
  return out;
}

std::string Gpt2Tokenizer::decode(const std::vector<uint16_t>& ids) const {
  std::string text;
  for (uint16_t id : ids) {
    if (id >= decoder_.size()) throw std::runtime_error("decode: token id out of range");
    text += decoder_[id];
  }
  return text;
}
