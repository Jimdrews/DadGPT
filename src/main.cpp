#include <format>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "bpe.hpp"
#include "csv_parser.hpp"
#include "dataloader.hpp"
#include "generate.hpp"
#include "gpt2_tokenizer.hpp"
#include "train.hpp"

namespace {

// GPT-2 tokenization: each joke becomes "question\nresponse", documents separated by
// <|endoftext|> (nanoGPT style), so a fine-tuned GPT-2 sees text in its native format.
int prepare_gpt2(const std::string& data_dir, const std::string& gpt2_dir,
                 const std::vector<Joke>& train_jokes, const std::vector<Joke>& test_jokes) {
  Gpt2Tokenizer tok;
  tok.load(gpt2_dir + "/vocab.json", gpt2_dir + "/merges.txt");
  std::cout << std::format("\nEncoding with GPT-2 BPE from {}...\n", gpt2_dir);

  auto encode_jokes = [&tok](const std::vector<Joke>& jokes) {
    std::vector<uint16_t> tokens;
    for (const auto& joke : jokes) {
      tokens.push_back(Gpt2Tokenizer::EOT);
      auto ids = tok.encode(joke.question + "\n" + joke.response);
      tokens.insert(tokens.end(), ids.begin(), ids.end());
    }
    tokens.push_back(Gpt2Tokenizer::EOT);
    return tokens;
  };
  auto train_tokens = encode_jokes(train_jokes);
  auto val_tokens = encode_jokes(test_jokes);
  std::cout << std::format("  Train tokens: {}\n  Val tokens:   {}\n", train_tokens.size(),
                           val_tokens.size());

  write_token_file(data_dir + "/train.bin", train_tokens);
  write_token_file(data_dir + "/val.bin", val_tokens);
  write_dataset_meta(data_dir, {"gpt2", tok.vocab_size(), gpt2_dir});
  std::cout << "  Saved train.bin, val.bin and meta.txt\n";

  std::cout << "\n--- Sample decode (first 2 jokes from train) ---\n";
  size_t end = 1;
  for (int shown = 0; shown < 2 && end < train_tokens.size(); ++shown) {
    size_t start = end;
    while (end < train_tokens.size() && train_tokens[end] != Gpt2Tokenizer::EOT) ++end;
    std::cout << tok.decode({train_tokens.begin() + static_cast<std::ptrdiff_t>(start),
                             train_tokens.begin() + static_cast<std::ptrdiff_t>(end)})
              << "\n<|endoftext|>\n";
    ++end;
  }
  std::cout << "\nDone!\n";
  return 0;
}

int prepare(const std::string& data_dir, const std::string& tokenizer,
            const std::string& gpt2_dir) {
  constexpr int VOCAB_SIZE = 512;

  std::cout << "=== dad-gpt prepare ===\n\n";

  // Parse train and test CSVs
  std::cout << "Parsing dataset...\n";
  auto train_jokes = parse_jokes_csv(data_dir + "/train.csv");
  auto test_jokes = parse_jokes_csv(data_dir + "/test.csv");
  std::cout << std::format("  Train: {} jokes\n", train_jokes.size());
  std::cout << std::format("  Test:  {} jokes\n", test_jokes.size());

  if (tokenizer == "gpt2") return prepare_gpt2(data_dir, gpt2_dir, train_jokes, test_jokes);
  if (tokenizer != "dadbpe") throw std::invalid_argument("--tokenizer must be dadbpe or gpt2");

  // Build corpus for BPE training (train split only)
  std::string corpus;
  for (const auto& joke : train_jokes) {
    corpus += joke.question;
    corpus += " ";
    corpus += joke.response;
    corpus += " ";
  }

  // Train BPE
  std::cout << std::format("\nTraining BPE tokenizer (vocab_size={})...\n", VOCAB_SIZE);
  BPE bpe;
  bpe.train(corpus, VOCAB_SIZE);
  std::cout << std::format("  Vocab size: {}\n", bpe.vocab_size());

  // Save vocab and merges
  bpe.save(data_dir + "/vocab.bin", data_dir + "/merges.bin");
  std::cout << "  Saved vocab.bin and merges.bin\n";

  // Encode jokes to token streams
  auto encode_jokes = [&bpe](const std::vector<Joke>& jokes) {
    std::vector<uint16_t> tokens;
    for (const auto& joke : jokes) {
      tokens.push_back(BPE::START_TOKEN);
      auto q_tokens = bpe.encode(joke.question);
      tokens.insert(tokens.end(), q_tokens.begin(), q_tokens.end());
      tokens.push_back(BPE::SEP_TOKEN);
      auto r_tokens = bpe.encode(joke.response);
      tokens.insert(tokens.end(), r_tokens.begin(), r_tokens.end());
      tokens.push_back(BPE::END_TOKEN);
    }
    return tokens;
  };

  std::cout << "\nEncoding jokes...\n";
  auto train_tokens = encode_jokes(train_jokes);
  auto val_tokens = encode_jokes(test_jokes);

  std::cout << std::format("  Train tokens: {}\n", train_tokens.size());
  std::cout << std::format("  Val tokens:   {}\n", val_tokens.size());

  // Write binary token files
  write_token_file(data_dir + "/train.bin", train_tokens);
  write_token_file(data_dir + "/val.bin", val_tokens);
  write_dataset_meta(data_dir, {"dadbpe", bpe.vocab_size()});
  std::cout << "  Saved train.bin, val.bin and meta.txt\n";

  // Verify: decode a sample
  std::cout << "\n--- Sample decode (first 3 jokes from train) ---\n";
  size_t pos = 0;
  int shown = 0;
  while (pos < train_tokens.size() && shown < 3) {
    // Find next END token
    size_t end = pos;
    while (end < train_tokens.size() && train_tokens[end] != BPE::END_TOKEN) {
      ++end;
    }
    if (end < train_tokens.size()) ++end;  // include END token

    std::vector<uint16_t> joke_tokens(train_tokens.begin() + static_cast<long>(pos),
                                      train_tokens.begin() + static_cast<long>(end));
    std::cout << bpe.decode(joke_tokens) << "\n";
    pos = end;
    ++shown;
  }

  std::cout << "\nDone!\n";
  return 0;
}

using FlagTable = std::map<std::string, std::function<void(const std::string&)>>;

// Applies "--flag value" pairs from argv[first..]. Unknown flags and missing values are errors
// rather than being silently ignored.
void parse_flags(int argc, char* argv[], int first, const FlagTable& flags) {
  for (int i = first; i < argc; ++i) {
    std::string arg = argv[i];
    auto it = flags.find(arg);
    if (it == flags.end()) throw std::invalid_argument("Unknown option " + arg);
    if (i + 1 >= argc) throw std::invalid_argument("Missing value for " + arg);
    it->second(argv[++i]);
  }
}

void print_usage() {
  std::cout << std::format("dad-gpt v0.2.0 | C++{}\n", __cplusplus);
  std::cout << R"(
Usage: dad-gpt <command> [options]

  prepare [data_dir]          Tokenize data_dir/{train,test}.csv into train.bin / val.bin
    --tokenizer NAME          dadbpe (train a 512-token BPE) | gpt2 (OpenAI's 50257 BPE)
    --gpt2-dir DIR            vocab.json + merges.txt for gpt2 (default data/gpt2)

  train                       Train (resumes from the latest data/checkpoint_step_N.bin)
    --data DIR                data directory (default data)
    --model NAME              tiny | small | gpt2-124m (default small)
    --init-from DIR           fine-tune pretrained HF GPT-2 (config.json + model.safetensors)
    --d-model N --layers N --heads N --seq-len N     override the preset
    --batch N --accum N       micro-batch size and gradient accumulation steps
    --steps N | --epochs F    optimizer steps, or passes over the training tokens
    --lr F --warmup N --wd F --dropout F --seed N
    --log-every N --eval-every N --eval-batches N --ckpt-every N --keep N

  generate                    Sample from a checkpoint
    --data DIR                data directory with meta.txt and the tokenizer
    --checkpoint PATH         default: latest checkpoint in --data
    --hf DIR                  sample from pretrained HF GPT-2 instead of a checkpoint
    --prompt TEXT --count N --max-len N
    --temp F (0 = greedy) --top-k N --rep-penalty F
)";
}

int run(int argc, char* argv[]) {
  std::string cmd = argc >= 2 ? argv[1] : "";

  if (cmd == "prepare") {
    std::string data_dir = "data", tokenizer = "dadbpe", gpt2_dir = "data/gpt2";
    int first = 2;
    if (argc >= 3 && std::string(argv[2]).rfind("--", 0) != 0) {
      data_dir = argv[2];
      first = 3;
    }
    parse_flags(argc, argv, first,
                {
                    {"--tokenizer", [&](const std::string& v) { tokenizer = v; }},
                    {"--gpt2-dir", [&](const std::string& v) { gpt2_dir = v; }},
                });
    return prepare(data_dir, tokenizer, gpt2_dir);
  }

  if (cmd == "train") {
    TrainConfig tc;
    auto i = [](int& dst) { return [&dst](const std::string& v) { dst = std::stoi(v); }; };
    auto f = [](float& dst) { return [&dst](const std::string& v) { dst = std::stof(v); }; };
    parse_flags(argc, argv, 2,
                {
                    {"--data", [&](const std::string& v) { tc.data_dir = v; }},
                    {"--model", [&](const std::string& v) { tc.model = v; }},
                    {"--init-from", [&](const std::string& v) { tc.init_from = v; }},
                    {"--d-model", i(tc.d_model)},
                    {"--layers", i(tc.n_layers)},
                    {"--heads", i(tc.n_heads)},
                    {"--seq-len", i(tc.seq_len)},
                    {"--batch", i(tc.batch_size)},
                    {"--accum", i(tc.accumulation_steps)},
                    {"--steps", i(tc.max_steps)},
                    {"--epochs", f(tc.epochs)},
                    {"--lr", f(tc.max_lr)},
                    {"--warmup", i(tc.warmup_steps)},
                    {"--wd", f(tc.weight_decay)},
                    {"--dropout", f(tc.dropout_rate)},
                    {"--seed", [&](const std::string& v) { tc.seed = std::stoull(v); }},
                    {"--log-every", i(tc.log_interval)},
                    {"--eval-every", i(tc.eval_interval)},
                    {"--eval-batches", i(tc.eval_batches)},
                    {"--ckpt-every", i(tc.checkpoint_interval)},
                    {"--keep", i(tc.keep_checkpoints)},
                });
    if (tc.dropout_rate < 0.0f || tc.dropout_rate >= 1.0f) {
      throw std::invalid_argument("--dropout must be in [0, 1)");
    }
    if (tc.accumulation_steps < 1 || tc.batch_size < 1 || tc.log_interval < 1 ||
        tc.eval_interval < 1) {
      throw std::invalid_argument("--accum, --batch, --log-every and --eval-every must be >= 1");
    }
    run_training(tc);
    return 0;
  }

  if (cmd == "generate") {
    GenerateConfig gc;
    std::string data_dir = "data";
    parse_flags(argc, argv, 2,
                {
                    {"--data", [&](const std::string& v) { data_dir = v; }},
                    {"--checkpoint", [&](const std::string& v) { gc.checkpoint_path = v; }},
                    {"--hf", [&](const std::string& v) { gc.hf_dir = v; }},
                    {"--temp", [&](const std::string& v) { gc.temperature = std::stof(v); }},
                    {"--top-k", [&](const std::string& v) { gc.top_k = std::stoi(v); }},
                    {"--count", [&](const std::string& v) { gc.count = std::stoi(v); }},
                    {"--max-len", [&](const std::string& v) { gc.max_len = std::stoi(v); }},
                    {"--rep-penalty", [&](const std::string& v) { gc.rep_penalty = std::stof(v); }},
                    {"--prompt", [&](const std::string& v) { gc.prompt = v; }},
                });
    gc.data_dir = data_dir;
    if (gc.checkpoint_path.empty()) gc.checkpoint_path = find_latest_checkpoint(data_dir);
    if (gc.checkpoint_path.empty() && gc.hf_dir.empty()) {
      throw std::invalid_argument("No checkpoint_step_*.bin in " + data_dir +
                                  "; pass --checkpoint");
    }
    run_generate(gc);
    return 0;
  }

  print_usage();
  return cmd.empty() || cmd == "help" || cmd == "--help" ? 0 : 1;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    return run(argc, argv);
  } catch (const std::invalid_argument& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
