#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "generate.hpp"
#include "gpt2_tokenizer.hpp"
#include "gpu_ops.cuh"
#include "hf_gpt2.hpp"
#include "model.hpp"

// Parity with Hugging Face's GPT-2 (openai-community/gpt2). The fixture is produced by
// scripts/gpt2_reference.py; the model files (~550 MB) live in data/gpt2 or $DADGPT_GPT2_DIR,
// and these tests skip when they are absent.

namespace {

std::filesystem::path gpt2_dir() {
  if (const char* env = std::getenv("DADGPT_GPT2_DIR")) return env;
  return std::filesystem::path(DADGPT_SOURCE_DIR) / "data" / "gpt2";
}

nlohmann::json load_fixture() {
  std::ifstream f(std::filesystem::path(DADGPT_SOURCE_DIR) / "tests" / "fixtures" /
                  "gpt2_reference.json");
  return nlohmann::json::parse(f);
}

std::vector<uint16_t> to_ids(const nlohmann::json& j) { return j.get<std::vector<uint16_t>>(); }

#define REQUIRE_GPT2_FILE(name)                                                         \
  if (!std::filesystem::exists(gpt2_dir() / (name))) {                                  \
    GTEST_SKIP() << (gpt2_dir() / (name)).string() << " not found (see test_gpt2.cpp)"; \
  }

}  // namespace

TEST(Gpt2Tokenizer, MatchesHuggingFace) {
  REQUIRE_GPT2_FILE("vocab.json");
  Gpt2Tokenizer tok;
  tok.load((gpt2_dir() / "vocab.json").string(), (gpt2_dir() / "merges.txt").string());
  EXPECT_EQ(tok.vocab_size(), 50257);

  nlohmann::json fixture = load_fixture();  // keep alive: the loop iterates into it
  for (const auto& c : fixture["tokenize"]) {
    std::string text = c["text"].get<std::string>();
    auto ids = tok.encode(text);
    EXPECT_EQ(ids, to_ids(c["ids"])) << "text: " << text;
    EXPECT_EQ(tok.decode(ids), text) << "round trip";
  }
}

TEST(Gpt2Weights, LogitsMatchHuggingFace) {
  REQUIRE_GPT2_FILE("model.safetensors");
  auto ref = load_fixture()["logits"];
  auto ids = to_ids(ref["ids"]);
  auto probe_ids = ref["probe_ids"].get<std::vector<int>>();

  Model model{};
  model_load_hf_gpt2(model, gpt2_dir().string(), /*batch_size=*/1, /*ctx_len=*/64);
  model.training = false;
  // GPT-2 small is 124,439,808 parameters; ctx_len 64 keeps 64 of the 1024 position rows
  EXPECT_EQ(model.param_offsets.total, 124439808u - (1024u - 64u) * 768u);

  int T = static_cast<int>(ids.size());
  int V = model.config.vocab_size;
  model_forward(model, ids.data(), 1, T);
  std::vector<float> logits(static_cast<size_t>(T) * V);
  gpu_copy_to_host(logits.data(), model.act(model.act_offsets.logits), logits.size());

  for (int t = 0; t < T; ++t) {
    const auto& pos = ref["positions"][t];
    const float* row = logits.data() + static_cast<size_t>(t) * V;

    int argmax = static_cast<int>(std::max_element(row, row + V) - row);
    EXPECT_EQ(argmax, pos["argmax"].get<int>()) << "position " << t;

    double m = *std::max_element(row, row + V), s = 0.0;
    for (int v = 0; v < V; ++v) s += std::exp(row[v] - m);
    EXPECT_NEAR(m + std::log(s), pos["logsumexp"].get<double>(), 2e-3) << "position " << t;

    for (size_t k = 0; k < probe_ids.size(); ++k) {
      EXPECT_NEAR(row[probe_ids[k]], pos["probes"][k].get<double>(), 2e-3)
          << "position " << t << " token " << probe_ids[k];
    }
    for (int k = 0; k < 5; ++k) {
      int id = pos["top5_ids"][k].get<int>();
      EXPECT_NEAR(row[id], pos["top5_logits"][k].get<double>(), 2e-3)
          << "position " << t << " top" << k;
    }
  }
  model_free(model);
}

TEST(Gpt2Weights, GreedyGenerationMatchesHuggingFace) {
  // Exercises the KV-cache path end to end with real weights
  REQUIRE_GPT2_FILE("model.safetensors");
  auto ref = load_fixture()["greedy"];
  auto prompt = to_ids(ref["prompt_ids"]);
  auto expected = to_ids(ref["continuation"]);

  Model model{};
  model_load_hf_gpt2(model, gpt2_dir().string(), /*batch_size=*/1, /*ctx_len=*/64);
  model.training = false;

  int max_len = static_cast<int>(prompt.size() + expected.size());
  auto out = generate_tokens(model, prompt, /*temperature=*/0.0f, /*top_k=*/0, max_len, 1.0f,
                             Gpt2Tokenizer::EOT);
  std::vector<uint16_t> continuation(out.begin() + static_cast<std::ptrdiff_t>(prompt.size()),
                                     out.end());
  EXPECT_EQ(continuation, expected);
  model_free(model);
}
