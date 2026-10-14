#include <gtest/gtest.h>

#include <cstdio>
#include <functional>
#include <vector>

#include "gpu_ops.cuh"
#include "model.hpp"

// Per-op GPU timings at the "small" preset's training shapes (B*S = 64*128 rows, d = 512).
// Disabled by default; run with:
//   dad-gpt-tests --gtest_also_run_disabled_tests --gtest_filter=*Bench*

namespace {

float time_ms(const std::function<void()>& fn, int reps = 20) {
  fn();  // warm up (cuBLAS heuristics, first-touch)
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < reps; ++i) fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / reps;
}

}  // namespace

TEST(Bench, DISABLED_OpTimings) {
  const int B = 64, S = 128, d = 512, ff = 2048, H = 8, dh = 64, V = 512;
  const int BS = B * S;
  cublasHandle_t h;
  CUBLAS_CHECK(cublasCreate(&h));

  float* x = gpu_alloc(static_cast<size_t>(BS) * ff);
  float* y = gpu_alloc(static_cast<size_t>(BS) * ff);
  float* z = gpu_alloc(static_cast<size_t>(BS) * ff);
  float* w = gpu_alloc(static_cast<size_t>(d) * ff);
  float* dw = gpu_alloc(static_cast<size_t>(d) * ff);
  float* vec = gpu_alloc(ff);
  float* vec2 = gpu_alloc(ff);
  float* scores = gpu_alloc(static_cast<size_t>(B) * H * S * S);
  float* scores2 = gpu_alloc(static_cast<size_t>(B) * H * S * S);
  float* logits = gpu_alloc(static_cast<size_t>(BS) * V);
  float* dlogits = gpu_alloc(static_cast<size_t>(BS) * V);
  uint16_t* targets = gpu_alloc_u16(BS);
  gpu_zero(x, static_cast<size_t>(BS) * ff);
  gpu_zero(logits, static_cast<size_t>(BS) * V);
  CUDA_CHECK(cudaMemset(targets, 0, BS * sizeof(uint16_t)));
  gpu_zero(vec, ff);

  struct Row {
    const char* name;
    float ms;
    int per_step;  // calls per training step for the small preset (8 layers)
  };
  std::vector<Row> rows = {
      {"sgemm BSxd * dxd (QKV/Wo fwd)", time_ms([&] { gpu_matmul(h, y, x, w, BS, d, d); }), 4 * 8},
      {"sgemm BSxd * dxff (fc1 fwd)", time_ms([&] { gpu_matmul(h, y, x, w, BS, d, ff); }), 2 * 8},
      {"matmul_backward BSxd*dxd", time_ms([&] { gpu_matmul_backward(h, y, x, w, z, dw, BS, d, d); }), 4 * 8},
      {"matmul_backward BSxd*dxff", time_ms([&] { gpu_matmul_backward(h, y, x, w, z, dw, BS, d, ff); }), 2 * 8},
      {"attn scores batched (QK^T)", time_ms([&] { gpu_matmul_strided_batched_bt(h, scores, x, y, S, dh, S, S * dh, S * dh, S * S, B * H); }), 3 * 8},
      {"softmax attention rows", time_ms([&] { gpu_softmax(scores2, scores, B * H * S, S); }), 8},
      {"softmax_backward attention", time_ms([&] { gpu_softmax_backward(scores2, scores, scores, B * H * S, S); }), 8},
      {"layernorm fwd", time_ms([&] { gpu_layernorm(y, x, vec, vec2, BS, d); }), 17},
      {"layernorm_backward (atomics)", time_ms([&] { gpu_layernorm_backward(z, dw, dw + d, y, x, vec, BS, d); }), 17},
      {"reduce_rows (bias grad) d=512", time_ms([&] { gpu_reduce_rows(vec2, y, BS, d); }), 6 * 8},
      {"reduce_rows (bias grad) ff", time_ms([&] { gpu_reduce_rows(vec2, y, BS, ff); }), 8},
      {"gelu fwd", time_ms([&] { gpu_gelu(y, x, BS * ff); }), 8},
      {"cross_entropy (alloc+sync)", time_ms([&] { gpu_cross_entropy_loss(dlogits, logits, targets, BS, V); }), 1},
      {"gradient_norm 25.5M (alloc+sync)", time_ms([&] { gpu_gradient_norm(x, 25547776 / 4); }), 1},
      {"add_vectors BSxd", time_ms([&] { gpu_add_vectors(y, x, z, BS * d); }), 6 * 8},
      {"split_heads", time_ms([&] { gpu_split_heads(y, x, B, S, H, dh); }), 6 * 8},
  };

  float total = 0.0f;
  std::printf("\n%-36s %9s %6s %10s\n", "op", "ms/call", "calls", "ms/step");
  for (const auto& r : rows) {
    std::printf("%-36s %9.3f %6d %10.2f\n", r.name, r.ms, r.per_step, r.ms * r.per_step);
    total += r.ms * r.per_step;
  }
  std::printf("%-36s %27.2f\n", "sum of listed ops", total);

  for (void* p : {static_cast<void*>(x), static_cast<void*>(y), static_cast<void*>(z),
                  static_cast<void*>(w), static_cast<void*>(dw), static_cast<void*>(vec),
                  static_cast<void*>(vec2), static_cast<void*>(scores), static_cast<void*>(scores2),
                  static_cast<void*>(logits), static_cast<void*>(dlogits), static_cast<void*>(targets)}) {
    gpu_free(p);
  }
  cublasDestroy(h);
}

TEST(Bench, DISABLED_TrainingStepBreakdown) {
  ModelConfig c = model_preset("small");
  c.batch_size = 64;
  Model model{};
  model_alloc(model, c);
  const int B = 64, S = 128, BS = B * S;
  std::vector<uint16_t> tokens(BS);
  for (int i = 0; i < BS; ++i) tokens[i] = static_cast<uint16_t>((i * 37) % c.vocab_size);
  uint16_t* d_targets = gpu_alloc_u16(BS);
  gpu_copy_u16_to_device(d_targets, tokens.data(), BS);
  size_t n = model.param_offsets.total;
  float* m = gpu_alloc(n);
  float* v = gpu_alloc(n);
  gpu_zero(m, n);
  gpu_zero(v, n);

  auto ce = [&] {
    gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                           model.act(model.act_offsets.logits), d_targets, BS, c.vocab_size);
  };
  for (float rate : {0.0f, 0.1f}) {
    model.training = rate > 0.0f;
    model.dropout_rate = rate;
    float fwd = time_ms([&] { model_forward(model, tokens.data(), B, S); }, 5);
    ce();
    float bwd = time_ms([&] { model_backward(model, tokens.data(), B, S); }, 5);
    std::printf("dropout %.1f: forward %.1f ms, backward %.1f ms\n", rate, fwd, bwd);
  }
  model.training = true;
  model.dropout_rate = 0.1f;
  for (auto [p, name] : {std::pair{MatmulPrecision::FP32, "fp32"},
                         std::pair{MatmulPrecision::TF32, "tf32"},
                         std::pair{MatmulPrecision::BF16, "bf16"}}) {
    gpu_set_matmul_precision(p);
    float fwd = time_ms([&] { model_forward(model, tokens.data(), B, S); }, 5);
    ce();
    float bwd = time_ms([&] { model_backward(model, tokens.data(), B, S); }, 5);
    std::printf("matmul %s: forward %.1f ms, backward %.1f ms\n", name, fwd, bwd);
  }
  gpu_set_matmul_precision(MatmulPrecision::FP32);
  std::printf("zero activations + grad activations (%.2f GB): %.1f ms\n",
              2.0 * model.act_offsets.total * 4 / 1e9, time_ms([&] {
                gpu_zero(model.activations, model.act_offsets.total);
                gpu_zero(model.grad_activations, model.act_offsets.total);
              }, 5));
  std::printf("zero_grad: %.1f ms\n", time_ms([&] { model_zero_grad(model); }, 5));
  std::printf("adamw step: %.1f ms\n", time_ms([&] {
                gpu_adamw_step(model.params, model.gradients, m, v, model.decay_flags, n, 1e-4f,
                               0.9f, 0.95f, 1e-8f, 0.1f, 1);
              }, 5));
  gpu_free(m);
  gpu_free(v);
  gpu_free(d_targets);
  model_free(model);
}

TEST(Bench, DISABLED_FusedVsMaterializedAttention) {
  // Whole-model forward+backward with dropout off, at the small preset's context and at
  // GPT-2's, so the attention share of the step grows with S
  for (auto [preset, B] : {std::pair{"small", 64}, std::pair{"gpt2-124m", 4}}) {
    ModelConfig c = model_preset(preset);
    c.vocab_size = 512;
    c.batch_size = B;
    int S = c.ctx_len, BS = B * S;
    std::vector<uint16_t> tokens(BS);
    for (int i = 0; i < BS; ++i) tokens[i] = static_cast<uint16_t>((i * 37) % c.vocab_size);
    uint16_t* d_targets = gpu_alloc_u16(BS);
    gpu_copy_u16_to_device(d_targets, tokens.data(), BS);

    for (bool materialized : {true, false}) {
      Model model{};
      model_alloc(model, c);
      model.training = false;
      model.attention = materialized ? AttentionImpl::Materialized : AttentionImpl::Fused;
      float fwd = time_ms([&] { model_forward(model, tokens.data(), B, S); }, 3);
      gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                             model.act(model.act_offsets.logits), d_targets, BS, c.vocab_size);
      float bwd = time_ms([&] { model_backward(model, tokens.data(), B, S); }, 3);
      size_t free_b = 0, total_b = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
      std::printf("%-9s B=%-2d S=%-4d %-12s forward %7.1f ms  backward %7.1f ms  GPU mem used %.2f GB\n",
                  preset, B, S, materialized ? "materialized" : "fused", fwd, bwd,
                  (total_b - free_b) / 1e9);
      model_free(model);
    }
    gpu_free(d_targets);
  }
}
