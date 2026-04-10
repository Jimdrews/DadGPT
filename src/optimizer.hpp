#pragma once

#include <cstddef>

struct Adam {
  float* m = nullptr;  // first moment
  float* v = nullptr;  // second moment
  size_t num_params = 0;
  int step = 0;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float max_lr = 3e-4f;
  float weight_decay = 0.1f;
  int warmup_steps = 100;
  int total_steps = 1;
};

void adam_alloc(Adam& opt, size_t num_params);
void adam_free(Adam& opt);

// Get current learning rate based on schedule
float adam_lr(const Adam& opt);

// Perform one update step: modifies params using grads, updates m/v
void adam_step(Adam& opt, float* params, const float* grads);

// Clip gradient global norm to max_norm, returns the original norm
float gradient_clip(float* grads, size_t num_params, float max_norm);
