#include <gtest/gtest.h>

#include <cstdio>

#include "bpe.hpp"
#include "test_util.hpp"

TEST(BPE, TrainProducesTargetVocabSize) {
  BPE bpe;
  bpe.train("hello world hello world hello world", 270);
  EXPECT_EQ(bpe.vocab_size(), 270);
}

TEST(BPE, SpecialTokenIds) {
  EXPECT_EQ(BPE::START_TOKEN, 0);
  EXPECT_EQ(BPE::SEP_TOKEN, 1);
  EXPECT_EQ(BPE::END_TOKEN, 2);
  EXPECT_EQ(BPE::PAD_TOKEN, 3);
  EXPECT_EQ(BPE::BYTE_OFFSET, 4);
}

TEST(BPE, EncodeDecodeRoundTrip) {
  BPE bpe;
  bpe.train("the quick brown fox jumps over the lazy dog. the quick brown fox.", 300);

  std::string text = "the quick brown fox";
  auto tokens = bpe.encode(text);
  std::string decoded = bpe.decode(tokens);
  EXPECT_EQ(decoded, text);
}

TEST(BPE, EncodeDecodeRoundTripVariousTexts) {
  BPE bpe;
  std::string corpus =
      "Why did the chicken cross the road? To get to the other side. "
      "What do you call a fake noodle? An impasta. "
      "I used to hate facial hair, but then it grew on me.";
  bpe.train(corpus, 300);

  for (const auto& text :
       {"chicken", "the road", "impasta", "facial hair", "hello"}) {
    auto tokens = bpe.encode(text);
    EXPECT_EQ(bpe.decode(tokens), text) << "Failed round-trip for: " << text;
  }
}

TEST(BPE, EncodingIsDeterministic) {
  BPE bpe;
  bpe.train("abcabc abcabc abcabc", 270);

  auto tokens1 = bpe.encode("abcabc");
  auto tokens2 = bpe.encode("abcabc");
  EXPECT_EQ(tokens1, tokens2);
}

TEST(BPE, SpecialTokensDecode) {
  BPE bpe;
  bpe.train("test", 262);

  std::vector<uint16_t> tokens = {BPE::START_TOKEN, BPE::SEP_TOKEN, BPE::END_TOKEN};
  std::string decoded = bpe.decode(tokens);
  EXPECT_EQ(decoded, "<|start|><|sep|><|end|>");
}

TEST(BPE, SaveLoadRoundTrip) {
  BPE bpe;
  bpe.train("hello world hello world hello world foo bar foo bar", 280);

  std::string text = "hello world foo bar";
  auto original_tokens = bpe.encode(text);

  std::string vocab_path = test_util::temp_path("vocab.bin");
  std::string merges_path = test_util::temp_path("merges.bin");
  bpe.save(vocab_path, merges_path);

  BPE loaded;
  loaded.load(vocab_path, merges_path);

  auto loaded_tokens = loaded.encode(text);
  EXPECT_EQ(original_tokens, loaded_tokens);
  EXPECT_EQ(loaded.decode(loaded_tokens), text);
  EXPECT_EQ(loaded.vocab_size(), bpe.vocab_size());

  std::remove(vocab_path.c_str());
  std::remove(merges_path.c_str());
}
