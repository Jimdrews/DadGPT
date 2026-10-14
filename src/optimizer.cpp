#include "optimizer.hpp"

#include <cmath>
#include <cstring>
#include <numbers>

void adam_alloc(Adam& opt, size_t num_params) {
  opt.num_params = num_params;
  opt.m = new float[num_params]();
  opt.v = new float[num_params]();
  opt.step = 0;
}

void adam_free(Adam& opt) {
  delete[] opt.m;
  delete[] opt.v;
  opt.m = nullptr;
  opt.v = nullptr;
}

float adam_lr(const Adam& opt) {
  if (opt.step < opt.warmup_steps) {
    return opt.max_lr * static_cast<float>(opt.step) / static_cast<float>(opt.warmup_steps);
  }
  // Cosine decay from max_lr to min_lr
  float min_lr = opt.max_lr * 0.1f;
  int decay_steps = opt.total_steps - opt.warmup_steps;
  if (decay_steps <= 0) return opt.max_lr;
  float progress =
      static_cast<float>(opt.step - opt.warmup_steps) / static_cast<float>(decay_steps);
  if (progress > 1.0f) progress = 1.0f;
  return min_lr + 0.5f * (opt.max_lr - min_lr) * (1.0f + std::cos(std::numbers::pi_v<float> * progress));
}

void adam_step(Adam& opt, float* params, const float* grads) {
  ++opt.step;
  float lr = adam_lr(opt);
  float bc1 = 1.0f - std::pow(opt.beta1, static_cast<float>(opt.step));
  float bc2 = 1.0f - std::pow(opt.beta2, static_cast<float>(opt.step));

  for (size_t i = 0; i < opt.num_params; ++i) {
    opt.m[i] = opt.beta1 * opt.m[i] + (1.0f - opt.beta1) * grads[i];
    opt.v[i] = opt.beta2 * opt.v[i] + (1.0f - opt.beta2) * grads[i] * grads[i];
    float m_hat = opt.m[i] / bc1;
    float v_hat = opt.v[i] / bc2;
    params[i] -= lr * m_hat / (std::sqrt(v_hat) + opt.eps);
  }
}

float gradient_clip(float* grads, size_t num_params, float max_norm) {
  float norm_sq = 0.0f;
  for (size_t i = 0; i < num_params; ++i) {
    norm_sq += grads[i] * grads[i];
  }
  float norm = std::sqrt(norm_sq);

  if (norm > max_norm) {
    float scale = max_norm / norm;
    for (size_t i = 0; i < num_params; ++i) {
      grads[i] *= scale;
    }
  }

  return norm;
}
