#include "gpu_ops.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <curand_kernel.h>
#include <vector>

// ============================================================================
// Memory helpers
// ============================================================================

float* gpu_alloc(size_t num_floats) {
  float* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, num_floats * sizeof(float)));
  return ptr;
}

uint16_t* gpu_alloc_u16(size_t count) {
  uint16_t* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(uint16_t)));
  return ptr;
}

uint8_t* gpu_alloc_u8(size_t count) {
  uint8_t* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(uint8_t)));
  return ptr;
}

void gpu_free(void* ptr) {
  if (ptr) CUDA_CHECK(cudaFree(ptr));
}

void gpu_copy_to_device(float* dst, const float* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_copy_to_host(float* dst, const float* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_copy_u16_to_device(uint16_t* dst, const uint16_t* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(uint16_t), cudaMemcpyHostToDevice));
}

void gpu_zero(float* ptr, size_t count) {
  CUDA_CHECK(cudaMemset(ptr, 0, count * sizeof(float)));
}

// ============================================================================
// cuBLAS matmul (row-major via column-major swap)
// ============================================================================

// All matmuls keep fp32 storage and fp32 accumulation; the compute type only chooses how
// cuBLAS multiplies. BF16 rounds the inputs to bfloat16 on the fly for tensor cores, which
// is the numerics of PyTorch's bf16 autocast for matmuls (~2x FP32 on GeForce Ampere).
static cublasComputeType_t g_compute_type = CUBLAS_COMPUTE_32F;

void gpu_set_matmul_precision(MatmulPrecision p) {
  g_compute_type = p == MatmulPrecision::BF16   ? CUBLAS_COMPUTE_32F_FAST_16BF
                   : p == MatmulPrecision::TF32 ? CUBLAS_COMPUTE_32F_FAST_TF32
                                                : CUBLAS_COMPUTE_32F;
}

namespace {

// Same argument order as cublasSgemm, dispatched through cublasGemmEx
cublasStatus_t sgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                     int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
                     const float* beta, float* C, int ldc) {
  return cublasGemmEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, beta, C,
                      CUDA_R_32F, ldc, g_compute_type, CUBLAS_GEMM_DEFAULT);
}

// Same argument order as cublasSgemmStridedBatched
cublasStatus_t sgemm_strided_batched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                     int m, int n, int k, const float* alpha, const float* A,
                                     int lda, long long sA, const float* B, int ldb, long long sB,
                                     const float* beta, float* C, int ldc, long long sC,
                                     int count) {
  return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_32F, lda, sA, B,
                                    CUDA_R_32F, ldb, sB, beta, C, CUDA_R_32F, ldc, sC, count,
                                    g_compute_type, CUBLAS_GEMM_DEFAULT);
}

}  // namespace

// Row-major C = A × B  →  cuBLAS: C^T = B^T × A^T (column-major)
// Call as: cublasSgemm(handle, N_op, T_op, N, M, K, α, B, ldb, A, lda, β, C, N)
void gpu_matmul(cublasHandle_t handle, float* out, const float* a, const float* b, int M, int K,
                int N) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N, a, K, &beta,
                           out, N));
}

// Row-major C = A^T × B  where A is (K×M), result is (M×N)
void gpu_matmul_transposed_a(cublasHandle_t handle, float* out, const float* a, const float* b,
                             int M, int K, int N, bool accumulate) {
  float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
  CUBLAS_CHECK(sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, b, N, a, M, &beta,
                           out, N));
}

// Row-major C = A × B^T  where B is (N×K), result is (M×N)
void gpu_matmul_transposed_b(cublasHandle_t handle, float* out, const float* a, const float* b,
                             int M, int K, int N) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(sgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, b, K, a, K, &beta,
                           out, N));
}

// --- Batched strided cuBLAS ---

// Row-major C[i] = A[i] × B[i]^T, batch_count matrices
void gpu_matmul_strided_batched_bt(cublasHandle_t handle, float* out, const float* a,
                                   const float* b, int M, int K, int N,
                                   long long strideA, long long strideB, long long strideC,
                                   int batch_count) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(sgemm_strided_batched(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, b, K,
                                         strideB, a, K, strideA, &beta, out, N, strideC,
                                         batch_count));
}

// Row-major C[i] = A[i] × B[i], batch_count matrices
void gpu_matmul_strided_batched(cublasHandle_t handle, float* out, const float* a,
                                const float* b, int M, int K, int N,
                                long long strideA, long long strideB, long long strideC,
                                int batch_count) {
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(sgemm_strided_batched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N,
                                         strideB, a, K, strideA, &beta, out, N, strideC,
                                         batch_count));
}

// Row-major C[i] += A[i]^T × B[i] (accumulate), A is (K×M), B is (K×N), C is (M×N)
void gpu_matmul_strided_batched_at(cublasHandle_t handle, float* out, const float* a,
                                   const float* b, int M, int K, int N,
                                   long long strideA, long long strideB, long long strideC,
                                   int batch_count) {
  float alpha = 1.0f, beta = 1.0f;  // beta=1 for accumulate
  CUBLAS_CHECK(sgemm_strided_batched(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, b, N,
                                         strideB, a, M, strideA, &beta, out, N, strideC,
                                         batch_count));
}

// ============================================================================
// Forward kernels
// ============================================================================

__global__ void softmax_kernel(float* out, const float* in, int cols) {
  int row = blockIdx.x;
  const float* row_in = in + row * cols;
  float* row_out = out + row * cols;

  extern __shared__ float sdata[];

  // Find max
  float max_val = -1e30f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    max_val = fmaxf(max_val, row_in[c]);
  }
  sdata[threadIdx.x] = max_val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    __syncthreads();
  }
  max_val = sdata[0];
  __syncthreads();  // all threads must read the max before sdata is reused for the sum

  // Exp and sum
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float val = expf(row_in[c] - max_val);
    row_out[c] = val;
    sum += val;
  }
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  sum = sdata[0];

  // Normalize
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row_out[c] /= sum;
  }
}

static int next_power_of_2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

void gpu_softmax(float* out, const float* in, int rows, int cols) {
  int threads = min(256, next_power_of_2(cols));
  softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(out, in, cols);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void layernorm_kernel(float* out, const float* in, const float* weight,
                                 const float* bias, int cols, float eps) {
  int row = blockIdx.x;
  const float* row_in = in + row * cols;
  float* row_out = out + row * cols;
  float n = (float)cols;

  extern __shared__ float sdata[];

  // Mean
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += row_in[c];
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float mean = sdata[0] / n;

  __syncthreads();

  // Variance
  float var_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float d = row_in[c] - mean;
    var_sum += d * d;
  }
  sdata[threadIdx.x] = var_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(sdata[0] / n + eps);

  // Normalize, scale, shift
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    row_out[c] = (row_in[c] - mean) * inv_std * weight[c] + bias[c];
  }
}

void gpu_layernorm(float* out, const float* in, const float* weight, const float* bias, int rows,
                   int cols, float eps) {
  int threads = min(256, next_power_of_2(cols));
  layernorm_kernel<<<rows, threads, threads * sizeof(float)>>>(out, in, weight, bias, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void gelu_kernel(float* out, const float* in, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  float x = in[i];
  float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
  out[i] = 0.5f * x * (1.0f + tanhf(inner));
}

void gpu_gelu(float* out, const float* in, int size) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  gelu_kernel<<<blocks, threads>>>(out, in, size);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void embedding_kernel(float* out, const uint16_t* tokens, const float* table,
                                 int num_tokens, int d_model) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * d_model;
  if (idx >= total) return;
  int tok = idx / d_model;
  int dim = idx % d_model;
  out[idx] = table[tokens[tok] * d_model + dim];
}

void gpu_embedding_lookup(float* out, const uint16_t* tokens, const float* table, int num_tokens,
                          int d_model) {
  int total = num_tokens * d_model;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  embedding_kernel<<<blocks, threads>>>(out, tokens, table, num_tokens, d_model);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void add_vectors_kernel(float* out, const float* a, const float* b, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  out[i] = a[i] + b[i];
}

void gpu_add_vectors(float* out, const float* a, const float* b, int size) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  add_vectors_kernel<<<blocks, threads>>>(out, a, b, size);
  CUDA_CHECK(cudaGetLastError());
}

// Fused bias-add: out[r * cols + c] += bias[c]
__global__ void add_bias_kernel(float* out, const float* bias, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;
  int c = idx % cols;
  out[idx] += bias[c];
}

void gpu_add_bias(float* out, const float* bias, int rows, int cols) {
  int total = rows * cols;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  add_bias_kernel<<<blocks, threads>>>(out, bias, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

// Reduce rows: dst[c] += sum_r src[r * cols + c]  (for bias gradient accumulation).
// Blocks are 32 columns x 8 row-lanes; gridDim.y splits the rows so the whole GPU works on
// tall inputs (B*S rows), and each block adds its partial column sums with one atomic.
constexpr int kReduceCols = 32, kReduceLanes = 8;

__global__ void reduce_rows_kernel(float* dst, const float* src, int rows, int cols) {
  __shared__ float partial[kReduceLanes][kReduceCols];
  int c = blockIdx.x * kReduceCols + threadIdx.x;
  float sum = 0.0f;
  if (c < cols) {
    for (int r = blockIdx.y * kReduceLanes + threadIdx.y; r < rows;
         r += gridDim.y * kReduceLanes) {
      sum += src[static_cast<size_t>(r) * cols + c];
    }
  }
  partial[threadIdx.y][threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.y == 0 && c < cols) {
    for (int lane = 1; lane < kReduceLanes; ++lane) sum += partial[lane][threadIdx.x];
    atomicAdd(&dst[c], sum);
  }
}

void gpu_reduce_rows(float* dst, const float* src, int rows, int cols) {
  dim3 threads(kReduceCols, kReduceLanes);
  int col_blocks = (cols + kReduceCols - 1) / kReduceCols;
  // ~64 rows per lane keeps atomics negligible while filling the GPU
  int row_blocks = std::max(1, std::min(256, rows / (kReduceLanes * 64)));
  reduce_rows_kernel<<<dim3(col_blocks, row_blocks), threads>>>(dst, src, rows, cols);
  CUDA_CHECK(cudaGetLastError());
}

// Cross-entropy: compute softmax, loss, and dlogits with parallel reduction
__global__ void cross_entropy_kernel(float* dlogits, float* losses, const float* logits,
                                     const uint16_t* targets, int vocab_size, int total_rows) {
  int row = blockIdx.x;
  const float* row_in = logits + row * vocab_size;
  float* row_out = dlogits + row * vocab_size;

  extern __shared__ float sdata[];

  // Find max (parallel reduction)
  float max_val = -1e30f;
  for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
    max_val = fmaxf(max_val, row_in[v]);
  }
  sdata[threadIdx.x] = max_val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
    __syncthreads();
  }
  max_val = sdata[0];
  __syncthreads();  // all threads must read the max before sdata is reused for the sum

  // Exp and sum (parallel reduction)
  float sum = 0.0f;
  for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
    float val = expf(row_in[v] - max_val);
    row_out[v] = val;
    sum += val;
  }
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  sum = sdata[0];

  // Normalize
  for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
    row_out[v] /= sum;
  }
  __syncthreads();

  // Loss = logsumexp(z) - z[t] = log(sum) - (z[t] - max). Exact even when p[t] underflows.
  if (threadIdx.x == 0) {
    losses[row] = logf(sum) - (row_in[targets[row]] - max_val);
  }

  // Gradient: softmax - one_hot, averaged over total rows
  float inv_seq = 1.0f / (float)total_rows;
  if (threadIdx.x == 0) {
    row_out[targets[row]] -= 1.0f;
  }
  __syncthreads();
  for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
    row_out[v] *= inv_seq;
  }
}

float gpu_cross_entropy_loss(float* dlogits, const float* logits, const uint16_t* targets,
                             int seq_len, int vocab_size) {
  int threads = min(256, next_power_of_2(vocab_size));
  float* d_losses = gpu_alloc(seq_len);
  cross_entropy_kernel<<<seq_len, threads, threads * sizeof(float)>>>(dlogits, d_losses, logits,
                                                                      targets, vocab_size, seq_len);
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> h_losses(seq_len);
  gpu_copy_to_host(h_losses.data(), d_losses, seq_len);
  gpu_free(d_losses);

  float total = 0.0f;
  for (float l : h_losses) total += l;
  return total / (float)seq_len;
}

// ============================================================================
// Attention helpers
// ============================================================================

// [B*S, D] → [B*H, S, dh]
// in[(b*S + s)*D + h*dh + x] → out[(b*H + h)*S*dh + s*dh + x]
__global__ void split_heads_kernel(float* out, const float* in, int batch_size, int seq_len,
                                   int n_heads, int d_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * n_heads * seq_len * d_head;
  if (idx >= total) return;
  int x = idx % d_head;
  int s = (idx / d_head) % seq_len;
  int h = (idx / (d_head * seq_len)) % n_heads;
  int b = idx / (d_head * seq_len * n_heads);
  int d_model = n_heads * d_head;
  out[(b * n_heads + h) * seq_len * d_head + s * d_head + x] =
      in[(b * seq_len + s) * d_model + h * d_head + x];
}

void gpu_split_heads(float* out, const float* in, int batch_size, int seq_len, int n_heads,
                     int d_head) {
  int total = batch_size * n_heads * seq_len * d_head;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  split_heads_kernel<<<blocks, threads>>>(out, in, batch_size, seq_len, n_heads, d_head);
  CUDA_CHECK(cudaGetLastError());
}

// [B*H, S, dh] → [B*S, D]
// in[(b*H + h)*S*dh + s*dh + x] → out[(b*S + s)*D + h*dh + x]
__global__ void merge_heads_kernel(float* out, const float* in, int batch_size, int seq_len,
                                   int n_heads, int d_head) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch_size * n_heads * seq_len * d_head;
  if (idx >= total) return;
  int x = idx % d_head;
  int s = (idx / d_head) % seq_len;
  int h = (idx / (d_head * seq_len)) % n_heads;
  int b = idx / (d_head * seq_len * n_heads);
  int d_model = n_heads * d_head;
  out[(b * seq_len + s) * d_model + h * d_head + x] =
      in[(b * n_heads + h) * seq_len * d_head + s * d_head + x];
}

void gpu_merge_heads(float* out, const float* in, int batch_size, int seq_len, int n_heads,
                     int d_head) {
  int total = batch_size * n_heads * seq_len * d_head;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  merge_heads_kernel<<<blocks, threads>>>(out, in, batch_size, seq_len, n_heads, d_head);
  CUDA_CHECK(cudaGetLastError());
}

// Apply causal mask: scores[h][i][j] = -1e9 where j > i
__global__ void causal_mask_kernel(float* scores, int seq_len) {
  int h = blockIdx.z;
  int i = blockIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= seq_len) return;
  if (j > i) {
    scores[h * seq_len * seq_len + i * seq_len + j] = -1e9f;
  }
}

void gpu_causal_mask(float* scores, int n_heads, int seq_len) {
  int threads = min(256, seq_len);
  int j_blocks = (seq_len + threads - 1) / threads;
  dim3 grid(j_blocks, seq_len, n_heads);
  causal_mask_kernel<<<grid, threads>>>(scores, seq_len);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void causal_mask_zero_kernel(float* grads, int seq_len) {
  int h = blockIdx.z;
  int i = blockIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= seq_len) return;
  if (j > i) {
    grads[h * seq_len * seq_len + i * seq_len + j] = 0.0f;
  }
}

void gpu_causal_mask_zero(float* grads, int n_heads, int seq_len) {
  int threads = min(256, seq_len);
  int j_blocks = (seq_len + threads - 1) / threads;
  dim3 grid(j_blocks, seq_len, n_heads);
  causal_mask_zero_kernel<<<grid, threads>>>(grads, seq_len);
  CUDA_CHECK(cudaGetLastError());
}

// Scale a matrix: out[i] *= scale
__global__ void scale_matrix_kernel(float* data, int size, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  data[i] *= scale;
}

void gpu_scale(float* data, int size, float scale) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  scale_matrix_kernel<<<blocks, threads>>>(data, size, scale);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Backward kernels
// ============================================================================

void gpu_matmul_backward(cublasHandle_t handle, const float* dout, const float* a, const float* b,
                         float* da, float* db, int M, int K, int N) {
  // da += dout × b^T
  float alpha = 1.0f, beta = 1.0f;  // beta=1 for accumulate
  CUBLAS_CHECK(
      sgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N, &alpha, b, N, dout, N, &beta, da, K));
  // db += a^T × dout
  CUBLAS_CHECK(sgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K, M, &alpha, dout, N, a, K, &beta,
                           db, N));
}

__global__ void softmax_backward_kernel(float* din, const float* dout, const float* out, int cols) {
  int row = blockIdx.x;
  const float* dout_row = dout + row * cols;
  const float* out_row = out + row * cols;
  float* din_row = din + row * cols;

  // dot = sum(dout * out)
  extern __shared__ float sdata[];
  float dot = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) dot += dout_row[c] * out_row[c];
  sdata[threadIdx.x] = dot;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  dot = sdata[0];

  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    din_row[c] += out_row[c] * (dout_row[c] - dot);
  }
}

void gpu_softmax_backward(float* din, const float* dout, const float* out, int rows, int cols) {
  int threads = min(256, next_power_of_2(cols));
  softmax_backward_kernel<<<rows, threads, threads * sizeof(float)>>>(din, dout, out, cols);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void layernorm_backward_kernel(float* din, float* dweight, float* dbias,
                                          const float* dout, const float* in, const float* weight,
                                          int cols, float eps) {
  int row = blockIdx.x;
  const float* dout_row = dout + row * cols;
  const float* in_row = in + row * cols;
  float* din_row = din + row * cols;
  float n = (float)cols;

  extern __shared__ float sdata[];

  // Recompute mean
  float local_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) local_sum += in_row[c];
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float mean = sdata[0] / n;
  __syncthreads();

  // Recompute variance
  float local_var = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float d = in_row[c] - mean;
    local_var += d * d;
  }
  sdata[threadIdx.x] = local_var;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(sdata[0] / n + eps);
  __syncthreads();

  // dweight, dbias (atomic since multiple rows contribute)
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float norm_c = (in_row[c] - mean) * inv_std;
    atomicAdd(&dweight[c], dout_row[c] * norm_c);
    atomicAdd(&dbias[c], dout_row[c]);
  }

  // Reduce sum_dnorm and sum_dnorm_norm
  float local_dnorm = 0.0f, local_dnorm_norm = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float dnorm_c = dout_row[c] * weight[c];
    float norm_c = (in_row[c] - mean) * inv_std;
    local_dnorm += dnorm_c;
    local_dnorm_norm += dnorm_c * norm_c;
  }
  sdata[threadIdx.x] = local_dnorm;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float sum_dnorm = sdata[0];
  __syncthreads();

  sdata[threadIdx.x] = local_dnorm_norm;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  float sum_dnorm_norm = sdata[0];
  __syncthreads();

  // Compute din
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float dnorm_c = dout_row[c] * weight[c];
    float norm_c = (in_row[c] - mean) * inv_std;
    din_row[c] += inv_std * (dnorm_c - sum_dnorm / n - norm_c * sum_dnorm_norm / n);
  }
}

void gpu_layernorm_backward(float* din, float* dweight, float* dbias, const float* dout,
                            const float* in, const float* weight, int rows, int cols, float eps) {
  int threads = min(256, next_power_of_2(cols));
  layernorm_backward_kernel<<<rows, threads, threads * sizeof(float)>>>(din, dweight, dbias, dout,
                                                                        in, weight, cols, eps);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void gelu_backward_kernel(float* din, const float* dout, const float* in, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  float x = in[i];
  float x3 = x * x * x;
  float inner = 0.7978845608f * (x + 0.044715f * x3);
  float tanh_val = tanhf(inner);
  float sech2 = 1.0f - tanh_val * tanh_val;
  float d_inner = 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
  float gelu_grad = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
  din[i] += dout[i] * gelu_grad;
}

void gpu_gelu_backward(float* din, const float* dout, const float* in, int size) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  gelu_backward_kernel<<<blocks, threads>>>(din, dout, in, size);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void embedding_backward_kernel(float* dtable, const float* dout,
                                          const uint16_t* tokens, int num_tokens, int d_model) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_tokens * d_model;
  if (idx >= total) return;
  int tok = idx / d_model;
  int dim = idx % d_model;
  atomicAdd(&dtable[tokens[tok] * d_model + dim], dout[idx]);
}

void gpu_embedding_backward(float* dtable, const float* dout, const uint16_t* tokens,
                            int num_tokens, int d_model) {
  int total = num_tokens * d_model;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  embedding_backward_kernel<<<blocks, threads>>>(dtable, dout, tokens, num_tokens, d_model);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Dropout kernels
// ============================================================================

// Stateless dropout using Philox counter-based RNG (no shared mutable state).
// Each thread creates its own local RNG state from (seed, thread_index).
// The seed increments each call so different dropout invocations produce different masks.
static unsigned long long d_dropout_seed = 42;

void gpu_dropout_init(unsigned long long seed, int /*max_size*/) {
  d_dropout_seed = seed;
}

void gpu_dropout_free() {
  // Nothing to free — no persistent GPU state
}

__global__ void dropout_forward_kernel(float* out, const float* in, uint8_t* mask, int size,
                                       float rate, unsigned long long seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, i, 0, &state);
  float r = curand_uniform(&state);
  float scale = 1.0f / (1.0f - rate);
  if (r < rate) {
    mask[i] = 0;
    out[i] = 0.0f;
  } else {
    mask[i] = 1;
    out[i] = in[i] * scale;
  }
}

void gpu_dropout_forward(float* out, const float* in, uint8_t* mask, int size, float rate) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  dropout_forward_kernel<<<blocks, threads>>>(out, in, mask, size, rate, d_dropout_seed++);
  CUDA_CHECK(cudaGetLastError());
}

// In-place dropout backward: dout[i] *= mask[i] * scale (masks gradient at same positions)
__global__ void dropout_backward_kernel(float* dout, const uint8_t* mask, int size, float rate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  float scale = 1.0f / (1.0f - rate);
  dout[i] *= mask[i] * scale;
}

void gpu_dropout_backward(float* dout, const uint8_t* mask, int size, float rate) {
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  dropout_backward_kernel<<<blocks, threads>>>(dout, mask, size, rate);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Optimizer kernels
// ============================================================================

__global__ void adamw_kernel(float* params, const float* grads, float* m, float* v,
                            const uint8_t* decay_flags, size_t num_params, float lr, float beta1,
                            float beta2, float eps, float weight_decay, float bc1, float bc2) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= num_params) return;
  // Decoupled weight decay (before Adam update)
  if (decay_flags[i]) {
    params[i] -= lr * weight_decay * params[i];
  }
  float g = grads[i];
  m[i] = beta1 * m[i] + (1.0f - beta1) * g;
  v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
  float m_hat = m[i] / bc1;
  float v_hat = v[i] / bc2;
  params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void gpu_adamw_step(float* params, const float* grads, float* m, float* v,
                    const uint8_t* decay_flags, size_t num_params, float lr, float beta1,
                    float beta2, float eps, float weight_decay, int step) {
  float bc1 = 1.0f - powf(beta1, (float)step);
  float bc2 = 1.0f - powf(beta2, (float)step);
  int threads = 256;
  int blocks = (int)((num_params + threads - 1) / threads);
  adamw_kernel<<<blocks, threads>>>(params, grads, m, v, decay_flags, num_params, lr, beta1, beta2,
                                    eps, weight_decay, bc1, bc2);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void adam_kernel(float* params, const float* grads, float* m, float* v,
                           size_t num_params, float lr, float beta1, float beta2, float eps,
                           float bc1, float bc2) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= num_params) return;
  float g = grads[i];
  m[i] = beta1 * m[i] + (1.0f - beta1) * g;
  v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
  float m_hat = m[i] / bc1;
  float v_hat = v[i] / bc2;
  params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

void gpu_adam_step(float* params, const float* grads, float* m, float* v, size_t num_params,
                  float lr, float beta1, float beta2, float eps, int step) {
  float bc1 = 1.0f - powf(beta1, (float)step);
  float bc2 = 1.0f - powf(beta2, (float)step);
  int threads = 256;
  int blocks = (int)((num_params + threads - 1) / threads);
  adam_kernel<<<blocks, threads>>>(params, grads, m, v, num_params, lr, beta1, beta2, eps, bc1,
                                  bc2);
  CUDA_CHECK(cudaGetLastError());
}

// Gradient norm: two-pass — reduce squared sum, then sqrt on host
__global__ void norm_sq_kernel(const float* grads, float* partial_sums, size_t num_params) {
  extern __shared__ float sdata[];
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  sdata[threadIdx.x] = (i < num_params) ? grads[i] * grads[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if ((int)threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial_sums[blockIdx.x] = sdata[0];
}

float gpu_gradient_norm(const float* grads, size_t num_params) {
  int threads = 256;
  int blocks = (int)((num_params + threads - 1) / threads);
  float* d_partial = gpu_alloc(blocks);
  norm_sq_kernel<<<blocks, threads, threads * sizeof(float)>>>(grads, d_partial, num_params);
  CUDA_CHECK(cudaGetLastError());

  std::vector<float> h_partial(blocks);
  gpu_copy_to_host(h_partial.data(), d_partial, blocks);
  gpu_free(d_partial);

  float sum = 0.0f;
  for (float v : h_partial) sum += v;
  return sqrtf(sum);
}

__global__ void scale_kernel(float* data, size_t size, float scale) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= size) return;
  data[i] *= scale;
}

void gpu_gradient_scale(float* grads, size_t num_params, float scale) {
  int threads = 256;
  int blocks = (int)((num_params + threads - 1) / threads);
  scale_kernel<<<blocks, threads>>>(grads, num_params, scale);
  CUDA_CHECK(cudaGetLastError());
}
