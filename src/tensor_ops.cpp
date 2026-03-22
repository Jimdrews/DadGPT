#include "tensor_ops.hpp"

#include <cmath>
#include <cstring>

void matmul(float* out, const float* a, const float* b, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
      }
      out[i * N + j] = sum;
    }
  }
}

void matmul_transposed_a(float* out, const float* a, const float* b, int M, int K, int N) {
  // a is (K×M) row-major, treat as (M×K) transposed
  // out = a^T × b = (M×K) × (K×N) = (M×N)
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a[k * M + i] * b[k * N + j];  // a transposed: row k, col i → a[k*M + i]
      }
      out[i * N + j] = sum;
    }
  }
}

void softmax(float* out, const float* in, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row_in = in + r * cols;
    float* row_out = out + r * cols;

    // Find max for numerical stability
    float max_val = row_in[0];
    for (int c = 1; c < cols; ++c) {
      if (row_in[c] > max_val) max_val = row_in[c];
    }

    // Exponentiate and sum
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      row_out[c] = std::exp(row_in[c] - max_val);
      sum += row_out[c];
    }

    // Normalize
    for (int c = 0; c < cols; ++c) {
      row_out[c] /= sum;
    }
  }
}

void layernorm(float* out, const float* in, const float* weight, const float* bias, int rows,
               int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* row_in = in + r * cols;
    float* row_out = out + r * cols;

    // Compute mean
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
      mean += row_in[c];
    }
    mean /= static_cast<float>(cols);

    // Compute variance
    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float diff = row_in[c] - mean;
      var += diff * diff;
    }
    var /= static_cast<float>(cols);

    // Normalize, scale, and shift
    float inv_std = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < cols; ++c) {
      row_out[c] = (row_in[c] - mean) * inv_std * weight[c] + bias[c];
    }
  }
}

void gelu(float* out, const float* in, int size) {
  constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/π)
  constexpr float coeff = 0.044715f;

  for (int i = 0; i < size; ++i) {
    float x = in[i];
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    out[i] = 0.5f * x * (1.0f + std::tanh(inner));
  }
}

void embedding_lookup(float* out, const uint16_t* tokens, const float* table, int num_tokens,
                      int d_model) {
  for (int i = 0; i < num_tokens; ++i) {
    const float* row = table + tokens[i] * d_model;
    std::memcpy(out + i * d_model, row, static_cast<size_t>(d_model) * sizeof(float));
  }
}

void add_vectors(float* out, const float* a, const float* b, int size) {
  for (int i = 0; i < size; ++i) {
    out[i] = a[i] + b[i];
  }
}

void matmul_transposed_b(float* out, const float* a, const float* b, int M, int K, int N) {
  // out = a × b^T where a is (M×K), b is (N×K) row-major
  // out[i][j] = sum_k a[i*K+k] * b[j*K+k]
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[j * K + k];
      }
      out[i * N + j] = sum;
    }
  }
}
