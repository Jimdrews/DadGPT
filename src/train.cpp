#include "train.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <regex>

#include "dataloader.hpp"
#include "gpu_ops.cuh"
#include "model.hpp"
#include "optimizer.hpp"

namespace fs = std::filesystem;

void run_training(const TrainConfig& tc) {
  std::cout << "=== dad-gpt train ===\n\n";

  // Check for existing checkpoint
  std::string resume_path;
  int resume_epoch = 0;
  {
    std::regex ckpt_re("checkpoint_epoch_(\\d+)\\.bin");
    for (const auto& entry : fs::directory_iterator(tc.data_dir)) {
      std::smatch match;
      std::string name = entry.path().filename().string();
      if (std::regex_match(name, match, ckpt_re)) {
        int epoch = std::stoi(match[1].str());
        if (epoch > resume_epoch) {
          resume_epoch = epoch;
          resume_path = entry.path().string();
        }
      }
    }
  }

  // Initialize model
  ModelConfig mc;
  mc.ctx_len = tc.seq_len;
  mc.batch_size = tc.batch_size;
  Model model{};
  Adam opt;  // m and v live on GPU

  if (!resume_path.empty()) {
    // Restores weights, Adam moments and step; buffers sized for the current --batch
    model_load_with_optimizer(model, opt, resume_path, tc.batch_size);
    std::cout << std::format("Resumed from {} (epoch {}, step {})\n", resume_path, resume_epoch,
                             opt.step);
    // Don't replay the dropout masks of the first run (seeds advance once per dropout call)
    gpu_dropout_init(42ULL + static_cast<unsigned long long>(opt.step) * 1000003ULL, 0);
  } else {
    model_alloc(model, mc);
    opt.num_params = model.param_offsets.total;
    opt.m = gpu_alloc(opt.num_params);
    opt.v = gpu_alloc(opt.num_params);
    gpu_zero(opt.m, opt.num_params);
    gpu_zero(opt.v, opt.num_params);
  }
  mc = model.config;
  if (tc.seq_len > mc.ctx_len) {
    throw std::runtime_error(
        std::format("--seq-len {} exceeds checkpoint ctx_len {}", tc.seq_len, mc.ctx_len));
  }
  model.training = true;
  model.dropout_rate = tc.dropout_rate;

  std::cout << std::format("Model: {} params, d_model={}, n_layers={}, n_heads={}\n",
                           model.param_offsets.total, mc.d_model, mc.n_layers, mc.n_heads);

  opt.max_lr = 3e-4f;
  opt.warmup_steps = 500;
  opt.weight_decay = tc.weight_decay;

  // Open data files
  DataLoader train_loader(tc.data_dir + "/train.bin", tc.batch_size, tc.seq_len);
  DataLoader val_loader(tc.data_dir + "/val.bin", 1, tc.seq_len);
  std::cout << std::format("Train tokens: {}, Val tokens: {}\n", train_loader.num_tokens(),
                           val_loader.num_tokens());
  std::cout << std::format("Batch size: {}, Seq len: {}\n\n", tc.batch_size, tc.seq_len);

  // Device buffer for targets (batch_size * seq_len)
  size_t batch_tokens = static_cast<size_t>(tc.batch_size) * tc.seq_len;
  uint16_t* d_targets = gpu_alloc_u16(batch_tokens);

  size_t tokens_per_batch =
      static_cast<size_t>(tc.batch_size) * static_cast<size_t>(tc.seq_len + 1);
  size_t batches_per_epoch = train_loader.num_tokens() / tokens_per_batch;
  // One optimizer step consumes accumulation_steps micro-batches (the last may be partial)
  size_t accum = static_cast<size_t>(std::max(tc.accumulation_steps, 1));
  size_t steps_per_epoch = (batches_per_epoch + accum - 1) / accum;
  opt.total_steps = std::max(1, static_cast<int>(steps_per_epoch * tc.epochs));

  int global_step = opt.step;

  for (int epoch = resume_epoch; epoch < tc.epochs; ++epoch) {
    train_loader.reset();
    float epoch_loss = 0.0f;
    int epoch_steps = 0;

    while (train_loader.has_next()) {
      model_zero_grad(model);
      float accum_loss = 0.0f;
      int n_micro = 0;

      for (int acc = 0; acc < tc.accumulation_steps && train_loader.has_next(); ++acc) {
        Batch batch = train_loader.next_batch();

        gpu_zero(model.activations, model.act_offsets.total);
        gpu_zero(model.grad_activations, model.act_offsets.total);

        // Single batched forward pass
        model_forward(model, batch.tokens.data(), tc.batch_size, tc.seq_len);

        // Copy all targets to device
        gpu_copy_u16_to_device(d_targets, batch.targets.data(), batch_tokens);

        // Loss + dlogits on GPU (averaged over B*S positions)
        float batch_loss = gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                                                  model.act(model.act_offsets.logits), d_targets,
                                                  static_cast<int>(batch_tokens), mc.vocab_size);

        // Backward pass (gradients accumulate across accumulation steps)
        model_backward(model, batch.tokens.data(), tc.batch_size, tc.seq_len);
        accum_loss += batch_loss;
        ++n_micro;
      }

      // Average gradients over the micro-batches actually run (the last group may be short)
      if (n_micro > 1) {
        float inv_accum = 1.0f / static_cast<float>(n_micro);
        gpu_gradient_scale(model.gradients, model.param_offsets.total, inv_accum);
        accum_loss *= inv_accum;
      }

      // Gradient clipping on GPU. A non-finite norm would poison the params and Adam state.
      float norm = gpu_gradient_norm(model.gradients, model.param_offsets.total);
      if (!std::isfinite(norm) || !std::isfinite(accum_loss)) {
        std::cerr << std::format("  step {:>5d} | non-finite grad norm/loss, skipping update\n",
                                 global_step + 1);
        continue;
      }
      if (norm > 1.0f) {
        gpu_gradient_scale(model.gradients, model.param_offsets.total, 1.0f / norm);
      }

      // AdamW step on GPU
      ++opt.step;
      float lr = adam_lr(opt);
      gpu_adamw_step(model.params, model.gradients, opt.m, opt.v, model.decay_flags,
                     model.param_offsets.total, lr, opt.beta1, opt.beta2, opt.eps,
                     opt.weight_decay, opt.step);

      epoch_loss += accum_loss;
      ++epoch_steps;
      ++global_step;

      if (global_step % tc.log_interval == 0) {
        float avg_loss = epoch_loss / static_cast<float>(epoch_steps);
        std::cout << std::format("  step {:>5d} | loss {:.4f} | lr {:.6f}\n", global_step,
                                 avg_loss, lr);
      }
    }

    // Validation (batch_size=1, dropout disabled)
    model.training = false;
    val_loader.reset();
    float val_loss = 0.0f;
    int val_steps = 0;
    float* val_dlogits = gpu_alloc(static_cast<size_t>(tc.seq_len) * mc.vocab_size);
    while (val_loader.has_next()) {
      Batch vb = val_loader.next_batch();
      gpu_zero(model.activations, model.act_offsets.total);
      model_forward(model, vb.tokens.data(), 1, tc.seq_len);

      gpu_copy_u16_to_device(d_targets, vb.targets.data(), tc.seq_len);
      val_loss += gpu_cross_entropy_loss(val_dlogits, model.act(model.act_offsets.logits),
                                         d_targets, tc.seq_len, mc.vocab_size);
      ++val_steps;
    }
    gpu_free(val_dlogits);
    model.training = true;

    float avg_train = epoch_loss / static_cast<float>(std::max(epoch_steps, 1));
    float avg_val = val_loss / static_cast<float>(std::max(val_steps, 1));
    std::cout << std::format("Epoch {:>2d} | train_loss {:.4f} | val_loss {:.4f}\n", epoch + 1,
                             avg_train, avg_val);

    std::string ckpt_path =
        tc.data_dir + "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
    model_save_with_optimizer(model, opt, ckpt_path);
    std::cout << std::format("  Saved {}\n\n", ckpt_path);
  }

  gpu_free(d_targets);
  gpu_free(opt.m);
  gpu_free(opt.v);
  model_free(model);
  std::cout << "Training complete!\n";
}
