// Fused causal self-attention (FlashAttention-2 style) in fp32.
//
// Forward walks key tiles with an online softmax, so the [S, S] score matrix never exists in
// memory; it saves only the per-row logsumexp. Backward recomputes the probabilities from
// Q, K and that logsumexp: one kernel accumulates dK/dV per key row, another dQ per query row,
// so no atomics are needed. Layout everywhere is [BH, S, D] (heads already split).

#include <cfloat>

#include "gpu_ops.cuh"

namespace {

constexpr int kRows = 32;  // query rows per block (one thread each)
constexpr int kCols = 32;  // key rows per shared-memory tile

template <int D>
__global__ void flash_fwd_kernel(float* __restrict__ out, float* __restrict__ lse,
                                 const float* __restrict__ q, const float* __restrict__ k,
                                 const float* __restrict__ v, int S, float scale) {
  __shared__ float ks[kCols][D];
  __shared__ float vs[kCols][D];
  size_t base = static_cast<size_t>(blockIdx.y) * S * D;
  int row = blockIdx.x * kRows + threadIdx.x;
  bool valid = row < S;

  float qr[D], o[D];
  for (int d = 0; d < D; ++d) {
    qr[d] = valid ? q[base + static_cast<size_t>(row) * D + d] * scale : 0.0f;
    o[d] = 0.0f;
  }
  float m = -FLT_MAX, l = 0.0f;

  int last_row = min(S, (blockIdx.x + 1) * kRows) - 1;  // causal: no key beyond the last row
  for (int start = 0; start <= last_row; start += kCols) {
    __syncthreads();
    for (int i = threadIdx.x; i < kCols * D; i += blockDim.x) {
      int r = start + i / D;
      ks[i / D][i % D] = r < S ? k[base + static_cast<size_t>(r) * D + i % D] : 0.0f;
      vs[i / D][i % D] = r < S ? v[base + static_cast<size_t>(r) * D + i % D] : 0.0f;
    }
    __syncthreads();
    if (!valid) continue;

    float s[kCols];
    float tile_max = -FLT_MAX;
    for (int j = 0; j < kCols; ++j) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += qr[d] * ks[j][d];
      s[j] = start + j <= row ? dot : -FLT_MAX;
      tile_max = fmaxf(tile_max, s[j]);
    }
    if (tile_max == -FLT_MAX) continue;  // tile entirely in the future

    float m_new = fmaxf(m, tile_max);
    float correction = expf(m - m_new);
    l *= correction;
    for (int d = 0; d < D; ++d) o[d] *= correction;
    for (int j = 0; j < kCols; ++j) {
      if (s[j] == -FLT_MAX) continue;
      float p = expf(s[j] - m_new);
      l += p;
      for (int d = 0; d < D; ++d) o[d] += p * vs[j][d];
    }
    m = m_new;
  }

  if (valid) {
    float inv_l = 1.0f / l;
    for (int d = 0; d < D; ++d) out[base + static_cast<size_t>(row) * D + d] = o[d] * inv_l;
    lse[static_cast<size_t>(blockIdx.y) * S + row] = m + logf(l);
  }
}

// delta[r] = sum_d dout[r, d] * out[r, d]  (the softmax-backward row term)
template <int D>
__global__ void flash_delta_kernel(float* delta, const float* dout, const float* out,
                                   int rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float sum = 0.0f;
  for (int d = 0; d < D; ++d) {
    sum += dout[static_cast<size_t>(r) * D + d] * out[static_cast<size_t>(r) * D + d];
  }
  delta[r] = sum;
}

// One thread per key row j: dV_j = sum_{r>=j} P_rj dO_r,  dK_j = scale * sum_{r>=j} dS_rj Q_r
template <int D>
__global__ void flash_bwd_dkdv_kernel(float* __restrict__ dk, float* __restrict__ dv,
                                      const float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v, const float* __restrict__ dout,
                                      const float* __restrict__ lse,
                                      const float* __restrict__ delta, int S, float scale) {
  __shared__ float kj_s[kCols][D + 1];  // +1 pad: each thread reads its own row
  __shared__ float vj_s[kCols][D + 1];
  __shared__ float qs[kRows][D];        // broadcast reads: no pad needed
  __shared__ float dos[kRows][D];
  __shared__ float lse_s[kRows], delta_s[kRows];

  size_t base = static_cast<size_t>(blockIdx.y) * S * D;
  size_t row_base = static_cast<size_t>(blockIdx.y) * S;
  int t = threadIdx.x;
  int j = blockIdx.x * kCols + t;
  bool valid = j < S;
  for (int d = 0; d < D; ++d) {
    kj_s[t][d] = valid ? k[base + static_cast<size_t>(j) * D + d] : 0.0f;
    vj_s[t][d] = valid ? v[base + static_cast<size_t>(j) * D + d] : 0.0f;
  }
  float dk_acc[D], dv_acc[D];
  for (int d = 0; d < D; ++d) dk_acc[d] = dv_acc[d] = 0.0f;

  // Query rows that can see this key tile start at the tile's first key
  for (int start = blockIdx.x * kCols; start < S; start += kRows) {
    __syncthreads();
    for (int i = t; i < kRows * D; i += blockDim.x) {
      int r = start + i / D;
      bool in = r < S;
      qs[i / D][i % D] = in ? q[base + static_cast<size_t>(r) * D + i % D] : 0.0f;
      dos[i / D][i % D] = in ? dout[base + static_cast<size_t>(r) * D + i % D] : 0.0f;
    }
    if (t < kRows) {
      int r = start + t;
      lse_s[t] = r < S ? lse[row_base + r] : 0.0f;
      delta_s[t] = r < S ? delta[row_base + r] : 0.0f;
    }
    __syncthreads();
    if (!valid) continue;

    for (int i = 0; i < kRows; ++i) {
      int r = start + i;
      if (r >= S) break;
      if (r < j) continue;  // causal: query r cannot see key j
      float s = 0.0f, dp = 0.0f;
      for (int d = 0; d < D; ++d) {
        s += qs[i][d] * kj_s[t][d];
        dp += dos[i][d] * vj_s[t][d];
      }
      float p = expf(s * scale - lse_s[i]);
      float ds = p * (dp - delta_s[i]) * scale;
      for (int d = 0; d < D; ++d) {
        dv_acc[d] += p * dos[i][d];
        dk_acc[d] += ds * qs[i][d];
      }
    }
  }
  if (valid) {
    for (int d = 0; d < D; ++d) {
      dk[base + static_cast<size_t>(j) * D + d] = dk_acc[d];
      dv[base + static_cast<size_t>(j) * D + d] = dv_acc[d];
    }
  }
}

// One thread per query row r: dQ_r = scale * sum_{j<=r} dS_rj K_j
template <int D>
__global__ void flash_bwd_dq_kernel(float* __restrict__ dq, const float* __restrict__ q,
                                    const float* __restrict__ k, const float* __restrict__ v,
                                    const float* __restrict__ dout, const float* __restrict__ lse,
                                    const float* __restrict__ delta, int S, float scale) {
  __shared__ float qr_s[kRows][D + 1];
  __shared__ float dor_s[kRows][D + 1];
  __shared__ float ks[kCols][D];
  __shared__ float vs[kCols][D];

  size_t base = static_cast<size_t>(blockIdx.y) * S * D;
  int t = threadIdx.x;
  int r = blockIdx.x * kRows + t;
  bool valid = r < S;
  for (int d = 0; d < D; ++d) {
    qr_s[t][d] = valid ? q[base + static_cast<size_t>(r) * D + d] : 0.0f;
    dor_s[t][d] = valid ? dout[base + static_cast<size_t>(r) * D + d] : 0.0f;
  }
  float row_lse = valid ? lse[static_cast<size_t>(blockIdx.y) * S + r] : 0.0f;
  float row_delta = valid ? delta[static_cast<size_t>(blockIdx.y) * S + r] : 0.0f;
  float dq_acc[D];
  for (int d = 0; d < D; ++d) dq_acc[d] = 0.0f;

  int last_row = min(S, (blockIdx.x + 1) * kRows) - 1;
  for (int start = 0; start <= last_row; start += kCols) {
    __syncthreads();
    for (int i = t; i < kCols * D; i += blockDim.x) {
      int c = start + i / D;
      ks[i / D][i % D] = c < S ? k[base + static_cast<size_t>(c) * D + i % D] : 0.0f;
      vs[i / D][i % D] = c < S ? v[base + static_cast<size_t>(c) * D + i % D] : 0.0f;
    }
    __syncthreads();
    if (!valid) continue;

    for (int j = 0; j < kCols; ++j) {
      int c = start + j;
      if (c > r) break;  // causal
      float s = 0.0f, dp = 0.0f;
      for (int d = 0; d < D; ++d) {
        s += qr_s[t][d] * ks[j][d];
        dp += dor_s[t][d] * vs[j][d];
      }
      float p = expf(s * scale - row_lse);
      float ds = p * (dp - row_delta) * scale;
      for (int d = 0; d < D; ++d) dq_acc[d] += ds * ks[j][d];
    }
  }
  if (valid) {
    for (int d = 0; d < D; ++d) dq[base + static_cast<size_t>(r) * D + d] = dq_acc[d];
  }
}

template <int D>
void launch_forward(float* out, float* lse, const float* q, const float* k, const float* v,
                    int BH, int S, float scale) {
  dim3 grid((S + kRows - 1) / kRows, BH);
  flash_fwd_kernel<D><<<grid, kRows>>>(out, lse, q, k, v, S, scale);
  CUDA_CHECK(cudaGetLastError());
}

template <int D>
void launch_backward(float* dq, float* dk, float* dv, float* delta, const float* dout,
                     const float* out, const float* lse, const float* q, const float* k,
                     const float* v, int BH, int S, float scale) {
  int rows = BH * S;
  flash_delta_kernel<D><<<(rows + 255) / 256, 256>>>(delta, dout, out, rows);
  CUDA_CHECK(cudaGetLastError());
  dim3 grid_k((S + kCols - 1) / kCols, BH);
  flash_bwd_dkdv_kernel<D><<<grid_k, kCols>>>(dk, dv, q, k, v, dout, lse, delta, S, scale);
  CUDA_CHECK(cudaGetLastError());
  dim3 grid_q((S + kRows - 1) / kRows, BH);
  flash_bwd_dq_kernel<D><<<grid_q, kRows>>>(dq, q, k, v, dout, lse, delta, S, scale);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

bool gpu_flash_attention_supported(int d_head) { return d_head == 32 || d_head == 64; }

void gpu_flash_attention_forward(float* out, float* lse, const float* q, const float* k,
                                 const float* v, int BH, int S, int d_head, float scale) {
  if (d_head == 64) return launch_forward<64>(out, lse, q, k, v, BH, S, scale);
  if (d_head == 32) return launch_forward<32>(out, lse, q, k, v, BH, S, scale);
  throw std::runtime_error("flash attention: unsupported d_head " + std::to_string(d_head));
}

void gpu_flash_attention_backward(float* dq, float* dk, float* dv, float* delta,
                                  const float* dout, const float* out, const float* lse,
                                  const float* q, const float* k, const float* v, int BH, int S,
                                  int d_head, float scale) {
  if (d_head == 64) {
    return launch_backward<64>(dq, dk, dv, delta, dout, out, lse, q, k, v, BH, S, scale);
  }
  if (d_head == 32) {
    return launch_backward<32>(dq, dk, dv, delta, dout, out, lse, q, k, v, BH, S, scale);
  }
  throw std::runtime_error("flash attention: unsupported d_head " + std::to_string(d_head));
}
