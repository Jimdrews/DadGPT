#include "train.hpp"

#include <cstring>
#include <format>
#include <iostream>

#include "backward.hpp"
#include "dataloader.hpp"
#include "model.hpp"
#include "optimizer.hpp"

void run_training(const TrainConfig& tc) {
  std::cout << "=== dad-gpt train ===\n\n";

  // Initialize model
  ModelConfig mc;
  mc.ctx_len = tc.seq_len;
  Model model{};
  model_alloc(model, mc);
  std::cout << std::format("Model: {} params, d_model={}, n_layers={}, n_heads={}\n",
                           model.param_offsets.total, mc.d_model, mc.n_layers, mc.n_heads);

  // Initialize optimizer
  Adam opt;
  opt.max_lr = 3e-4f;
  opt.warmup_steps = 100;
  adam_alloc(opt, model.param_offsets.total);

  // Open data files
  DataLoader train_loader(tc.data_dir + "/train.bin", tc.batch_size, tc.seq_len);
  DataLoader val_loader(tc.data_dir + "/val.bin", 1, tc.seq_len);
  std::cout << std::format("Train tokens: {}, Val tokens: {}\n", train_loader.num_tokens(),
                           val_loader.num_tokens());
  std::cout << std::format("Batch size: {}, Seq len: {}\n\n", tc.batch_size, tc.seq_len);

  // Estimate total steps for LR schedule
  size_t tokens_per_batch =
      static_cast<size_t>(tc.batch_size) * static_cast<size_t>(tc.seq_len + 1);
  size_t batches_per_epoch = train_loader.num_tokens() / tokens_per_batch;
  opt.total_steps = static_cast<int>(batches_per_epoch * tc.epochs);

  int global_step = 0;

  for (int epoch = 0; epoch < tc.epochs; ++epoch) {
    train_loader.reset();
    float epoch_loss = 0.0f;
    int epoch_steps = 0;

    while (train_loader.has_next()) {
      Batch batch = train_loader.next_batch();

      // Zero gradients for this batch
      model_zero_grad(model);
      float batch_loss = 0.0f;

      // Forward + backward for each sequence in the batch
      for (int b = 0; b < tc.batch_size; ++b) {
        std::memset(model.activations, 0, model.act_offsets.total * sizeof(float));
        std::memset(model.grad_activations, 0, model.act_offsets.total * sizeof(float));

        model_forward(model, batch.tokens[b].data(), tc.seq_len);

        float loss = cross_entropy_loss(model.act(model.act_offsets.logits),
                                        batch.targets[b].data(),
                                        model.grad_act(model.act_offsets.logits), tc.seq_len,
                                        mc.vocab_size);

        model_backward(model, batch.tokens[b].data(), tc.seq_len);
        batch_loss += loss;
      }

      // Average gradients over batch
      float inv_batch = 1.0f / static_cast<float>(tc.batch_size);
      for (size_t i = 0; i < model.param_offsets.total; ++i) {
        model.gradients[i] *= inv_batch;
      }
      batch_loss *= inv_batch;

      // Gradient clipping
      gradient_clip(model.gradients, model.param_offsets.total, 1.0f);

      // Optimizer step
      adam_step(opt, model.params, model.gradients);

      epoch_loss += batch_loss;
      ++epoch_steps;
      ++global_step;

      if (global_step % tc.log_interval == 0) {
        float avg_loss = epoch_loss / static_cast<float>(epoch_steps);
        std::cout << std::format("  step {:>5d} | loss {:.4f} | lr {:.6f}\n", global_step,
                                 avg_loss, adam_lr(opt));
      }
    }

    // Validation
    val_loader.reset();
    float val_loss = 0.0f;
    int val_steps = 0;
    while (val_loader.has_next()) {
      Batch vb = val_loader.next_batch();
      std::memset(model.activations, 0, model.act_offsets.total * sizeof(float));
      model_forward(model, vb.tokens[0].data(), tc.seq_len);

      std::vector<float> dummy_dlogits(static_cast<size_t>(tc.seq_len) * mc.vocab_size, 0.0f);
      val_loss += cross_entropy_loss(model.act(model.act_offsets.logits), vb.targets[0].data(),
                                     dummy_dlogits.data(), tc.seq_len, mc.vocab_size);
      ++val_steps;
    }

    float avg_train = epoch_loss / static_cast<float>(std::max(epoch_steps, 1));
    float avg_val = val_loss / static_cast<float>(std::max(val_steps, 1));
    std::cout << std::format("Epoch {:>2d} | train_loss {:.4f} | val_loss {:.4f}\n", epoch + 1,
                             avg_train, avg_val);

    // Save checkpoint
    std::string ckpt_path = tc.data_dir + "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".bin";
    model_save(model, ckpt_path);
    std::cout << std::format("  Saved {}\n\n", ckpt_path);
  }

  adam_free(opt);
  model_free(model);
  std::cout << "Training complete!\n";
}
