#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "tensor_ops.hpp"

TEST(TensorOps, MatmulBasic) {
  // [2×3] × [3×2] = [2×2]
  float a[] = {1, 2, 3, 4, 5, 6};
  float b[] = {7, 8, 9, 10, 11, 12};
  float c[4] = {};

  matmul(c, a, b, 2, 3, 2);

  EXPECT_NEAR(c[0], 58.0f, 1e-5);   // 1*7 + 2*9 + 3*11
  EXPECT_NEAR(c[1], 64.0f, 1e-5);   // 1*8 + 2*10 + 3*12
  EXPECT_NEAR(c[2], 139.0f, 1e-5);  // 4*7 + 5*9 + 6*11
  EXPECT_NEAR(c[3], 154.0f, 1e-5);  // 4*8 + 5*10 + 6*12
}

TEST(TensorOps, MatmulIdentity) {
  float a[] = {1, 2, 3, 4};
  float eye[] = {1, 0, 0, 1};
  float c[4] = {};

  matmul(c, a, eye, 2, 2, 2);

  EXPECT_NEAR(c[0], 1.0f, 1e-5);
  EXPECT_NEAR(c[1], 2.0f, 1e-5);
  EXPECT_NEAR(c[2], 3.0f, 1e-5);
  EXPECT_NEAR(c[3], 4.0f, 1e-5);
}

TEST(TensorOps, SoftmaxSumsToOne) {
  float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float out[4] = {};

  softmax(out, in, 1, 4);

  float sum = 0.0f;
  for (float v : out) sum += v;
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(TensorOps, SoftmaxNumericalStability) {
  float in[] = {1000.0f, 1001.0f, 1002.0f};
  float out[3] = {};

  softmax(out, in, 1, 3);

  for (float v : out) {
    EXPECT_TRUE(std::isfinite(v)) << "Got non-finite value: " << v;
  }
  float sum = out[0] + out[1] + out[2];
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

TEST(TensorOps, SoftmaxPreservesOrder) {
  float in[] = {1.0f, 3.0f, 2.0f};
  float out[3] = {};

  softmax(out, in, 1, 3);

  EXPECT_GT(out[1], out[2]);
  EXPECT_GT(out[2], out[0]);
}

TEST(TensorOps, LayernormStats) {
  float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float bias[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float out[4] = {};

  layernorm(out, in, weight, bias, 1, 4);

  // Check mean ≈ 0
  float mean = 0.0f;
  for (float v : out) mean += v;
  mean /= 4.0f;
  EXPECT_NEAR(mean, 0.0f, 1e-5);

  // Check variance ≈ 1
  float var = 0.0f;
  for (float v : out) var += (v - mean) * (v - mean);
  var /= 4.0f;
  EXPECT_NEAR(var, 1.0f, 1e-2);
}

TEST(TensorOps, LayernormScaleBias) {
  float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float weight[] = {2.0f, 2.0f, 2.0f, 2.0f};
  float bias[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float out[4] = {};

  layernorm(out, in, weight, bias, 1, 4);

  // Mean after scale+bias should be bias (1.0) since normalized mean is 0
  float mean = 0.0f;
  for (float v : out) mean += v;
  mean /= 4.0f;
  EXPECT_NEAR(mean, 1.0f, 1e-5);
}

TEST(TensorOps, GeluKnownValues) {
  float in[] = {0.0f, 3.0f, -3.0f};
  float out[3] = {};

  gelu(out, in, 3);

  EXPECT_NEAR(out[0], 0.0f, 1e-5);
  EXPECT_NEAR(out[1], 3.0f, 0.01f);   // large positive ≈ input
  EXPECT_NEAR(out[2], 0.0f, 0.01f);   // large negative ≈ 0
}

TEST(TensorOps, EmbeddingLookup) {
  // 4 tokens, d_model=3, embedding table has 10 entries
  float table[30] = {};
  for (int i = 0; i < 30; ++i) table[i] = static_cast<float>(i);

  uint16_t tokens[] = {0, 2, 5, 1};
  float out[12] = {};

  embedding_lookup(out, tokens, table, 4, 3);

  // Token 0 → table[0..2] = {0, 1, 2}
  EXPECT_NEAR(out[0], 0.0f, 1e-5);
  EXPECT_NEAR(out[1], 1.0f, 1e-5);
  EXPECT_NEAR(out[2], 2.0f, 1e-5);

  // Token 2 → table[6..8] = {6, 7, 8}
  EXPECT_NEAR(out[3], 6.0f, 1e-5);
  EXPECT_NEAR(out[4], 7.0f, 1e-5);
  EXPECT_NEAR(out[5], 8.0f, 1e-5);
}

TEST(TensorOps, AddVectors) {
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  float out[3] = {};

  add_vectors(out, a, b, 3);

  EXPECT_NEAR(out[0], 5.0f, 1e-5);
  EXPECT_NEAR(out[1], 7.0f, 1e-5);
  EXPECT_NEAR(out[2], 9.0f, 1e-5);
}

TEST(TensorOps, MatmulTransposedA) {
  // a is (3×2) row-major, a^T is (2×3)
  // a^T × b where b is (3×2) → out is (2×2)
  float a[] = {1, 4, 2, 5, 3, 6};  // stored as 3×2
  float b[] = {7, 8, 9, 10, 11, 12};
  float c[4] = {};

  matmul_transposed_a(c, a, b, 2, 3, 2);

  // a^T = [1 2 3; 4 5 6], same as the matmul test
  EXPECT_NEAR(c[0], 58.0f, 1e-5);
  EXPECT_NEAR(c[1], 64.0f, 1e-5);
  EXPECT_NEAR(c[2], 139.0f, 1e-5);
  EXPECT_NEAR(c[3], 154.0f, 1e-5);
}
