#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error: ") +               \
                               cudaGetErrorString(err) + " at " __FILE__ + \
                               ":" + std::to_string(__LINE__));             \
    }                                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = (call);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error(std::string("cuBLAS error at " __FILE__ ":") + \
                               std::to_string(__LINE__));                      \
    }                                                                          \
  } while (0)

// --- Memory helpers ---
float* gpu_alloc(size_t num_floats);
uint16_t* gpu_alloc_u16(size_t count);
void gpu_free(void* ptr);
void gpu_copy_to_device(float* dst, const float* src, size_t count);
void gpu_copy_to_host(float* dst, const float* src, size_t count);
void gpu_copy_u16_to_device(uint16_t* dst, const uint16_t* src, size_t count);
void gpu_zero(float* ptr, size_t count);

// --- cuBLAS matmul (row-major) ---
// C = A × B where A is (M×K), B is (K×N), C is (M×N)
void gpu_matmul(cublasHandle_t handle, float* out, const float* a, const float* b, int M, int K,
                int N);
// C = A^T × B
void gpu_matmul_transposed_a(cublasHandle_t handle, float* out, const float* a, const float* b,
                             int M, int K, int N);
// C = A × B^T
void gpu_matmul_transposed_b(cublasHandle_t handle, float* out, const float* a, const float* b,
                             int M, int K, int N);

// --- Forward kernels ---
void gpu_softmax(float* out, const float* in, int rows, int cols);
void gpu_layernorm(float* out, const float* in, const float* weight, const float* bias, int rows,
                   int cols, float eps = 1e-5f);
void gpu_gelu(float* out, const float* in, int size);
void gpu_embedding_lookup(float* out, const uint16_t* tokens, const float* table, int num_tokens,
                          int d_model);
void gpu_add_vectors(float* out, const float* a, const float* b, int size);
// Add bias[cols] to each row of out[rows × cols]: out[r][c] += bias[c]
void gpu_add_bias(float* out, const float* bias, int rows, int cols);
// Accumulate rows into a bias gradient: dbias[c] += sum_r dout[r][c]
void gpu_reduce_rows(float* dst, const float* src, int rows, int cols);

// Returns scalar loss (on host). Writes dlogits on device.
float gpu_cross_entropy_loss(float* dlogits, const float* logits, const uint16_t* targets,
                             int seq_len, int vocab_size);

// --- Backward kernels ---
void gpu_matmul_backward(cublasHandle_t handle, const float* dout, const float* a, const float* b,
                         float* da, float* db, int M, int K, int N);
void gpu_softmax_backward(float* din, const float* dout, const float* out, int rows, int cols);
void gpu_layernorm_backward(float* din, float* dweight, float* dbias, const float* dout,
                            const float* in, const float* weight, int rows, int cols,
                            float eps = 1e-5f);
void gpu_gelu_backward(float* din, const float* dout, const float* in, int size);
void gpu_embedding_backward(float* dtable, const float* dout, const uint16_t* tokens,
                            int num_tokens, int d_model);

// --- Attention helpers ---
// Reshape [seq_len × d_model] → [n_heads × seq_len × d_head] (split heads)
void gpu_split_heads(float* out, const float* in, int seq_len, int n_heads, int d_head);
// Reshape [n_heads × seq_len × d_head] → [seq_len × d_model] (merge heads)
void gpu_merge_heads(float* out, const float* in, int seq_len, int n_heads, int d_head);
// Apply causal mask: set scores[i][j] = -1e9 where j > i (forward pass)
void gpu_causal_mask(float* scores, int n_heads, int seq_len);
// Zero out causal mask positions: set grads[i][j] = 0 where j > i (backward pass)
void gpu_causal_mask_zero(float* grads, int n_heads, int seq_len);
// Scale all elements: data[i] *= scale
void gpu_scale(float* data, int size, float scale);

// --- Optimizer kernels ---
void gpu_adam_step(float* params, const float* grads, float* m, float* v, size_t num_params,
                  float lr, float beta1, float beta2, float eps, int step);
float gpu_gradient_norm(const float* grads, size_t num_params);
void gpu_gradient_scale(float* grads, size_t num_params, float scale);
