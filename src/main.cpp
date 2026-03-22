#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "bpe.hpp"
#include "csv_parser.hpp"
#include "dataloader.hpp"
#include "train.hpp"

namespace {

int prepare(const std::string& data_dir) {
  constexpr int VOCAB_SIZE = 4096;

  std::cout << "=== dad-gpt prepare ===\n\n";

  // Parse train and test CSVs
  std::cout << "Parsing dataset...\n";
  auto train_jokes = parse_jokes_csv(data_dir + "/train.csv");
  auto test_jokes = parse_jokes_csv(data_dir + "/test.csv");
  std::cout << std::format("  Train: {} jokes\n", train_jokes.size());
  std::cout << std::format("  Test:  {} jokes\n", test_jokes.size());

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
  std::cout << "  Saved train.bin and val.bin\n";

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

}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 2 && std::string(argv[1]) == "prepare") {
    std::string data_dir = "data";
    if (argc >= 3) {
      data_dir = argv[2];
    }
    return prepare(data_dir);
  }

  if (argc >= 2 && std::string(argv[1]) == "train") {
    TrainConfig tc;
    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--epochs" && i + 1 < argc) {
        tc.epochs = std::stoi(argv[++i]);
      } else if (arg == "--data" && i + 1 < argc) {
        tc.data_dir = argv[++i];
      } else if (arg == "--batch" && i + 1 < argc) {
        tc.batch_size = std::stoi(argv[++i]);
      }
    }
    run_training(tc);
    return 0;
  }

  std::cout << std::format("dad-gpt v0.1.0 | C++{}\n", __cplusplus);
  std::cout << "\nUsage: dad-gpt <command>\n";
  std::cout << "  prepare [data_dir]              - Tokenize dataset\n";
  std::cout << "  train [--epochs N] [--batch N]  - Train the model\n";
  std::cout << "        [--data DIR]\n";

  return 0;
}
