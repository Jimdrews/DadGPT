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
uint8_t* gpu_alloc_u8(size_t count);
void gpu_free(void* ptr);
void gpu_copy_to_device(float* dst, const float* src, size_t count);
void gpu_copy_to_host(float* dst, const float* src, size_t count);
void gpu_copy_u16_to_device(uint16_t* dst, const uint16_t* src, size_t count);
void gpu_zero(float* ptr, size_t count);
// Free and total device memory in bytes (cudaMemGetInfo)
void gpu_memory_info(size_t& free_bytes, size_t& total_bytes);

// --- cuBLAS matmul (row-major) ---
// Process-wide multiply precision for every matmul below. Storage and accumulation stay fp32.
// FP32 (default) is exact fp32; TF32/BF16 round inputs for tensor cores (training speed).
enum class MatmulPrecision { FP32, TF32, BF16 };
void gpu_set_matmul_precision(MatmulPrecision p);

// C = A × B where A is (M×K), B is (K×N), C is (M×N)
void gpu_matmul(cublasHandle_t handle, float* out, const float* a, const float* b, int M, int K,
                int N);
// C = A^T × B, or C += A^T × B when accumulate is true
void gpu_matmul_transposed_a(cublasHandle_t handle, float* out, const float* a, const float* b,
                             int M, int K, int N, bool accumulate = false);
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

// --- Batched strided cuBLAS ---
// C[i] = A[i] × B[i]^T for i in [0, batch_count), each A is (M×K), B is (N×K), C is (M×N)
void gpu_matmul_strided_batched_bt(cublasHandle_t handle, float* out, const float* a,
                                   const float* b, int M, int K, int N,
                                   long long strideA, long long strideB, long long strideC,
                                   int batch_count);
// C[i] = A[i] × B[i] for i in [0, batch_count), each A is (M×K), B is (K×N), C is (M×N)
void gpu_matmul_strided_batched(cublasHandle_t handle, float* out, const float* a,
                                const float* b, int M, int K, int N,
                                long long strideA, long long strideB, long long strideC,
                                int batch_count);
// C[i] += A[i]^T × B[i] (accumulate), each A is (K×M), B is (K×N), C is (M×N)
void gpu_matmul_strided_batched_at(cublasHandle_t handle, float* out, const float* a,
                                   const float* b, int M, int K, int N,
                                   long long strideA, long long strideB, long long strideC,
                                   int batch_count);

// --- Attention helpers ---
// Reshape [B*S, D] → [B*H, S, dh] (split heads with batch)
void gpu_split_heads(float* out, const float* in, int batch_size, int seq_len, int n_heads,
                     int d_head);
// Reshape [B*H, S, dh] → [B*S, D] (merge heads with batch)
void gpu_merge_heads(float* out, const float* in, int batch_size, int seq_len, int n_heads,
                     int d_head);
// Apply causal mask: set scores[i][j] = -1e9 where j > i (per S×S block in B*H blocks)
void gpu_causal_mask(float* scores, int num_blocks, int seq_len);
// Zero out causal mask positions: set grads[i][j] = 0 where j > i (per S×S block)
void gpu_causal_mask_zero(float* grads, int num_blocks, int seq_len);
// Scale all elements: data[i] *= scale
void gpu_scale(float* data, int size, float scale);

// --- Fused causal attention (cuda/attention.cu) ---
// q, k, v, out: [BH, S, d_head]; lse: [BH, S] row logsumexps saved for backward.
// Computes out = softmax(mask(q k^T * scale)) v without materializing the [S, S] scores.
bool gpu_flash_attention_supported(int d_head);
void gpu_flash_attention_forward(float* out, float* lse, const float* q, const float* k,
                                 const float* v, int BH, int S, int d_head, float scale);
// Writes (does not accumulate) dq, dk, dv. delta: [BH, S] scratch.
void gpu_flash_attention_backward(float* dq, float* dk, float* dv, float* delta,
                                  const float* dout, const float* out, const float* lse,
                                  const float* q, const float* k, const float* v, int BH, int S,
                                  int d_head, float scale);

// --- Dropout kernels ---
// Initialize cuRAND states (call once)
void gpu_dropout_init(unsigned long long seed, int max_size);
void gpu_dropout_free();
// Forward: out = in * mask * scale, mask stored as uint8 (0 or 1)
void gpu_dropout_forward(float* out, const float* in, uint8_t* mask, int size, float rate);
// In-place backward: dout[i] *= mask[i] * scale
void gpu_dropout_backward(float* dout, const uint8_t* mask, int size, float rate);

// --- Optimizer kernels ---
// AdamW step with decoupled weight decay. decay_flags[i]=1 means apply weight decay.
void gpu_adamw_step(float* params, const float* grads, float* m, float* v,
                    const uint8_t* decay_flags, size_t num_params, float lr, float beta1,
                    float beta2, float eps, float weight_decay, int step);
void gpu_adam_step(float* params, const float* grads, float* m, float* v, size_t num_params,
                  float lr, float beta1, float beta2, float eps, int step);
float gpu_gradient_norm(const float* grads, size_t num_params);
void gpu_gradient_scale(float* grads, size_t num_params, float scale);
