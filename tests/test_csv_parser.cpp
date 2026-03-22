#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>

#include "csv_parser.hpp"

namespace {

std::string write_temp_csv(const std::string& content) {
  std::string path = "/tmp/test_jokes.csv";
  std::ofstream f(path);
  f << content;
  return path;
}

}  // namespace

TEST(CsvParser, BasicParsing) {
  auto path = write_temp_csv(
      "\"question\",\"response\"\n"
      "\"Why?\",\"Because.\"\n"
      "\"How?\",\"Like this.\"\n");

  auto jokes = parse_jokes_csv(path);
  ASSERT_EQ(jokes.size(), 2);
  EXPECT_EQ(jokes[0].question, "Why?");
  EXPECT_EQ(jokes[0].response, "Because.");
  EXPECT_EQ(jokes[1].question, "How?");
  EXPECT_EQ(jokes[1].response, "Like this.");
}

TEST(CsvParser, QuotedFieldsWithCommas) {
  auto path = write_temp_csv(
      "\"question\",\"response\"\n"
      "\"Hello, world\",\"Goodbye, world\"\n");

  auto jokes = parse_jokes_csv(path);
  ASSERT_EQ(jokes.size(), 1);
  EXPECT_EQ(jokes[0].question, "Hello, world");
  EXPECT_EQ(jokes[0].response, "Goodbye, world");
}

TEST(CsvParser, EscapedQuotes) {
  auto path = write_temp_csv(
      "\"question\",\"response\"\n"
      "\"He said \"\"hello\"\"\",\"She said \"\"goodbye\"\"\"\n");

  auto jokes = parse_jokes_csv(path);
  ASSERT_EQ(jokes.size(), 1);
  EXPECT_EQ(jokes[0].question, "He said \"hello\"");
  EXPECT_EQ(jokes[0].response, "She said \"goodbye\"");
}

TEST(CsvParser, NewlinesInFields) {
  auto path = write_temp_csv(
      "\"question\",\"response\"\n"
      "\"Line one\nLine two\",\"Answer\"\n");

  auto jokes = parse_jokes_csv(path);
  ASSERT_EQ(jokes.size(), 1);
  EXPECT_EQ(jokes[0].question, "Line one\nLine two");
  EXPECT_EQ(jokes[0].response, "Answer");
}

TEST(CsvParser, MissingFileThrows) {
  EXPECT_THROW(parse_jokes_csv("/nonexistent/path.csv"), std::runtime_error);
}
