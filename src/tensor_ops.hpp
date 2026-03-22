#pragma once

#include <cstdint>

// CPU reference implementations of tensor operations.
// Used by unit tests. The model uses GPU versions from cuda/gpu_ops.cuh.

// C = A × B where A is (M×K), B is (K×N), C is (M×N)
void matmul(float* out, const float* a, const float* b, int M, int K, int N);

// Row-wise softmax with max subtraction for numerical stability
void softmax(float* out, const float* in, int rows, int cols);

// Layer normalization: normalize each row, then apply scale and bias
void layernorm(float* out, const float* in, const float* weight, const float* bias, int rows,
               int cols, float eps = 1e-5f);

// GELU activation (tanh approximation from GPT-2)
void gelu(float* out, const float* in, int size);

// Look up embeddings by token ID: out[b][s] = table[tokens[b][s]]
void embedding_lookup(float* out, const uint16_t* tokens, const float* table, int num_tokens,
                      int d_model);

// Element-wise addition: out = a + b
void add_vectors(float* out, const float* a, const float* b, int size);

// out = a^T × b where a is (K×M) stored row-major, result is (M×N)
void matmul_transposed_a(float* out, const float* a, const float* b, int M, int K, int N);

// out = a × b^T where b is (N×K) stored row-major, result is (M×N)
// Used for weight-tied output projection: logits = hidden · embedding^T
void matmul_transposed_b(float* out, const float* a, const float* b, int M, int K, int N);
