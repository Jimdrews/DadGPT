#include "backward.hpp"

#include <cmath>
#include <cstring>

#include "tensor_ops.hpp"

void matmul_backward(const float* dout, const float* a, const float* b, float* da, float* db,
                     int M, int K, int N) {
  // da += dout × b^T: da[i][k] += sum_j dout[i][j] * b[k][j]
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float sum = 0.0f;
      for (int j = 0; j < N; ++j) {
        sum += dout[i * N + j] * b[k * N + j];
      }
      da[i * K + k] += sum;
    }
  }

  // db += a^T × dout: db[k][j] += sum_i a[i][k] * dout[i][j]
  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int i = 0; i < M; ++i) {
        sum += a[i * K + k] * dout[i * N + j];
      }
      db[k * N + j] += sum;
    }
  }
}

void softmax_backward(const float* dout, const float* out, float* din, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* dout_row = dout + r * cols;
    const float* out_row = out + r * cols;
    float* din_row = din + r * cols;

    // dot = sum(dout * out)
    float dot = 0.0f;
    for (int c = 0; c < cols; ++c) {
      dot += dout_row[c] * out_row[c];
    }

    // din_i = out_i * (dout_i - dot)
    for (int c = 0; c < cols; ++c) {
      din_row[c] += out_row[c] * (dout_row[c] - dot);
    }
  }
}

void layernorm_backward(const float* dout, const float* in, const float* weight, float* din,
                        float* dweight, float* dbias, int rows, int cols, float eps) {
  for (int r = 0; r < rows; ++r) {
    const float* dout_row = dout + r * cols;
    const float* in_row = in + r * cols;
    float* din_row = din + r * cols;
    float n = static_cast<float>(cols);

    // Recompute forward stats
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) {
      mean += in_row[c];
    }
    mean /= n;

    float var = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float diff = in_row[c] - mean;
      var += diff * diff;
    }
    var /= n;
    float inv_std = 1.0f / std::sqrt(var + eps);

    // Normalized values (before scale/bias)
    // norm_c = (in_c - mean) * inv_std

    // Gradient for weight and bias
    for (int c = 0; c < cols; ++c) {
      float norm_c = (in_row[c] - mean) * inv_std;
      dweight[c] += dout_row[c] * norm_c;
      dbias[c] += dout_row[c];
    }

    // Gradient for input
    // dnorm_c = dout_c * weight_c
    // din = inv_std * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))
    float sum_dnorm = 0.0f;
    float sum_dnorm_norm = 0.0f;
    for (int c = 0; c < cols; ++c) {
      float dnorm_c = dout_row[c] * weight[c];
      float norm_c = (in_row[c] - mean) * inv_std;
      sum_dnorm += dnorm_c;
      sum_dnorm_norm += dnorm_c * norm_c;
    }

    for (int c = 0; c < cols; ++c) {
      float dnorm_c = dout_row[c] * weight[c];
      float norm_c = (in_row[c] - mean) * inv_std;
      din_row[c] += inv_std * (dnorm_c - sum_dnorm / n - norm_c * sum_dnorm_norm / n);
    }
  }
}

void gelu_backward(const float* dout, const float* in, float* din, int size) {
  constexpr float sqrt_2_over_pi = 0.7978845608f;
  constexpr float coeff = 0.044715f;

  for (int i = 0; i < size; ++i) {
    float x = in[i];
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = std::tanh(inner);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
    // gelu(x) = 0.5 * x * (1 + tanh(inner))
    // gelu'(x) = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech²(inner) * d_inner
    float gelu_grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
    din[i] += dout[i] * gelu_grad;
  }
}

void embedding_backward(const float* dout, const uint16_t* tokens, float* dtable, int num_tokens,
                        int d_model) {
  for (int i = 0; i < num_tokens; ++i) {
    float* row = dtable + tokens[i] * d_model;
    const float* grad = dout + i * d_model;
    for (int j = 0; j < d_model; ++j) {
      row[j] += grad[j];
    }
  }
}

float cross_entropy_loss(const float* logits, const uint16_t* targets, float* dlogits, int seq_len,
                         int vocab_size) {
  float total_loss = 0.0f;

  for (int s = 0; s < seq_len; ++s) {
    const float* row = logits + s * vocab_size;
    float* drow = dlogits + s * vocab_size;

    // Softmax
    float max_val = row[0];
    for (int v = 1; v < vocab_size; ++v) {
      if (row[v] > max_val) max_val = row[v];
    }

    float sum_exp = 0.0f;
    for (int v = 0; v < vocab_size; ++v) {
      drow[v] = std::exp(row[v] - max_val);
      sum_exp += drow[v];
    }

    for (int v = 0; v < vocab_size; ++v) {
      drow[v] /= sum_exp;
    }

    // Loss: -log(prob[target])
    total_loss += -std::log(drow[targets[s]] + 1e-10f);

    // Gradient: softmax - one_hot
    drow[targets[s]] -= 1.0f;

    // Average over sequence length
    for (int v = 0; v < vocab_size; ++v) {
      drow[v] /= static_cast<float>(seq_len);
    }
  }

  return total_loss / static_cast<float>(seq_len);
}
