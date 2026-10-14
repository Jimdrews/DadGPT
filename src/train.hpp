#pragma once

#include <cstdint>
#include <string>

struct TrainConfig {
  std::string data_dir = "data";

  // Architecture: a preset (see model_preset) with optional overrides (0 = keep preset).
  // Ignored when resuming, since the checkpoint fixes the architecture.
  std::string model = "small";
  int d_model = 0;
  int n_layers = 0;
  int n_heads = 0;
  int seq_len = 0;  // context length; 0 = preset ctx_len

  // Optimization
  int batch_size = 64;
  int accumulation_steps = 1;
  int max_steps = 0;    // optimizer steps; 0 = derive from epochs
  float epochs = 10.0f;  // passes over the training tokens, used when max_steps == 0
  int warmup_steps = 500;
  float max_lr = 3e-4f;
  float weight_decay = 0.1f;
  float dropout_rate = 0.1f;
  uint64_t seed = 1337;

  // Logging, evaluation and checkpoints (all in optimizer steps)
  int log_interval = 10;
  int eval_interval = 250;
  int eval_batches = 50;  // validation batches per eval; 0 = the whole validation set
  int checkpoint_interval = 1000;  // 0 = only at the end
  int keep_checkpoints = 2;        // older checkpoint_step_*.bin files are deleted
};

void run_training(const TrainConfig& config);

// Highest-step checkpoint_step_N.bin in dir, or "" if there is none
std::string find_latest_checkpoint(const std::string& dir);
