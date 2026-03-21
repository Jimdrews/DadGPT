#include <gtest/gtest.h>

TEST(SampleTest, BasicAssertion) { EXPECT_EQ(1 + 1, 2); }

TEST(SampleTest, StringComparison) {
  std::string greeting = "dad-gpt";
  EXPECT_EQ(greeting, "dad-gpt");
}
