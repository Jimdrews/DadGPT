#include "train.hpp"

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
  Model model{};

  if (!resume_path.empty()) {
    model_load(model, resume_path);
    std::cout << std::format("Resumed from {} (epoch {})\n", resume_path, resume_epoch);
  } else {
    model_alloc(model, mc);
  }
  mc = model.config;

  std::cout << std::format("Model: {} params, d_model={}, n_layers={}, n_heads={}\n",
                           model.param_offsets.total, mc.d_model, mc.n_layers, mc.n_heads);

  // Initialize optimizer — m and v on GPU
  Adam opt;
  opt.max_lr = 3e-4f;
  opt.warmup_steps = 500;
  opt.num_params = model.param_offsets.total;
  opt.m = gpu_alloc(model.param_offsets.total);
  opt.v = gpu_alloc(model.param_offsets.total);
  gpu_zero(opt.m, model.param_offsets.total);
  gpu_zero(opt.v, model.param_offsets.total);

  // Open data files
  DataLoader train_loader(tc.data_dir + "/train.bin", tc.batch_size, tc.seq_len);
  DataLoader val_loader(tc.data_dir + "/val.bin", 1, tc.seq_len);
  std::cout << std::format("Train tokens: {}, Val tokens: {}\n", train_loader.num_tokens(),
                           val_loader.num_tokens());
  std::cout << std::format("Batch size: {}, Seq len: {}\n\n", tc.batch_size, tc.seq_len);

  // Device buffer for targets
  uint16_t* d_targets = gpu_alloc_u16(tc.seq_len);

  size_t tokens_per_batch =
      static_cast<size_t>(tc.batch_size) * static_cast<size_t>(tc.seq_len + 1);
  size_t batches_per_epoch = train_loader.num_tokens() / tokens_per_batch;
  opt.total_steps = static_cast<int>(batches_per_epoch * tc.epochs);

  int global_step = 0;

  for (int epoch = resume_epoch; epoch < tc.epochs; ++epoch) {
    train_loader.reset();
    float epoch_loss = 0.0f;
    int epoch_steps = 0;

    while (train_loader.has_next()) {
      Batch batch = train_loader.next_batch();

      model_zero_grad(model);
      float batch_loss = 0.0f;

      for (int b = 0; b < tc.batch_size; ++b) {
        gpu_zero(model.activations, model.act_offsets.total);
        gpu_zero(model.grad_activations, model.act_offsets.total);

        // Forward pass (tokens copied to device inside model_forward)
        model_forward(model, batch.tokens[b].data(), tc.seq_len);

        // Copy targets to device for cross-entropy
        gpu_copy_u16_to_device(d_targets, batch.targets[b].data(), tc.seq_len);

        // Loss + dlogits on GPU
        float loss = gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                                            model.act(model.act_offsets.logits), d_targets,
                                            tc.seq_len, mc.vocab_size);

        model_backward(model, batch.tokens[b].data(), tc.seq_len);
        batch_loss += loss;
      }

      // Average gradients
      float inv_batch = 1.0f / static_cast<float>(tc.batch_size);
      gpu_gradient_scale(model.gradients, model.param_offsets.total, inv_batch);
      batch_loss *= inv_batch;

      // Gradient clipping on GPU
      float norm = gpu_gradient_norm(model.gradients, model.param_offsets.total);
      if (norm > 1.0f) {
        gpu_gradient_scale(model.gradients, model.param_offsets.total, 1.0f / norm);
      }

      // Adam step on GPU
      ++opt.step;
      float lr = adam_lr(opt);
      gpu_adam_step(model.params, model.gradients, opt.m, opt.v, model.param_offsets.total, lr,
                    opt.beta1, opt.beta2, opt.eps, opt.step);

      epoch_loss += batch_loss;
      ++epoch_steps;
      ++global_step;

      if (global_step % tc.log_interval == 0) {
        float avg_loss = epoch_loss / static_cast<float>(epoch_steps);
        std::cout << std::format("  step {:>5d} | loss {:.4f} | lr {:.6f}\n", global_step,
                                 avg_loss, lr);
      }
    }

    // Validation
    val_loader.reset();
    float val_loss = 0.0f;
    int val_steps = 0;
    float* val_dlogits = gpu_alloc(static_cast<size_t>(tc.seq_len) * mc.vocab_size);
    while (val_loader.has_next()) {
      Batch vb = val_loader.next_batch();
      gpu_zero(model.activations, model.act_offsets.total);
      model_forward(model, vb.tokens[0].data(), tc.seq_len);

      gpu_copy_u16_to_device(d_targets, vb.targets[0].data(), tc.seq_len);
      val_loss += gpu_cross_entropy_loss(val_dlogits, model.act(model.act_offsets.logits),
                                         d_targets, tc.seq_len, mc.vocab_size);
      ++val_steps;
    }
    gpu_free(val_dlogits);

    float avg_train = epoch_loss / static_cast<float>(std::max(epoch_steps, 1));
    float avg_val = val_loss / static_cast<float>(std::max(val_steps, 1));
    std::cout << std::format("Epoch {:>2d} | train_loss {:.4f} | val_loss {:.4f}\n", epoch + 1,
                             avg_train, avg_val);

    std::string ckpt_path =
        tc.data_dir + "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
    model_save(model, ckpt_path);
    std::cout << std::format("  Saved {}\n\n", ckpt_path);
  }

  gpu_free(d_targets);
  gpu_free(opt.m);
  gpu_free(opt.v);
  model_free(model);
  std::cout << "Training complete!\n";
}
