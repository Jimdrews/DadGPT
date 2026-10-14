#include "train.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dataloader.hpp"
#include "gpu_ops.cuh"
#include "hf_gpt2.hpp"
#include "model.hpp"
#include "optimizer.hpp"

namespace fs = std::filesystem;

namespace {

// (step, path) for every checkpoint_step_N.bin in dir, sorted by step
std::vector<std::pair<int, fs::path>> list_checkpoints(const std::string& dir) {
  std::vector<std::pair<int, fs::path>> found;
  if (!fs::is_directory(dir)) return found;
  std::regex ckpt_re("checkpoint_step_(\\d+)\\.bin");
  for (const auto& entry : fs::directory_iterator(dir)) {
    std::smatch match;
    std::string name = entry.path().filename().string();
    if (std::regex_match(name, match, ckpt_re)) {
      found.emplace_back(std::stoi(match[1].str()), entry.path());
    }
  }
  std::sort(found.begin(), found.end());
  return found;
}

ModelConfig build_config(const TrainConfig& tc, const DatasetMeta& meta) {
  ModelConfig mc = model_preset(tc.model);
  if (tc.d_model > 0) {
    mc.d_model = tc.d_model;
    mc.d_ff = 4 * tc.d_model;
  }
  if (tc.n_layers > 0) mc.n_layers = tc.n_layers;
  if (tc.n_heads > 0) mc.n_heads = tc.n_heads;
  if (tc.seq_len > 0) mc.ctx_len = tc.seq_len;
  mc.vocab_size = meta.vocab_size;
  mc.batch_size = tc.batch_size;
  return mc;
}

// Mean loss over up to max_batches sequential validation batches (dropout off)
float evaluate(Model& model, DataLoader& val, uint16_t* d_targets, int batch, int seq_len,
               int max_batches) {
  bool was_training = model.training;
  model.training = false;
  val.reset();
  double total = 0.0;
  int n = 0;
  size_t rows = static_cast<size_t>(batch) * seq_len;
  while (val.has_next() && (max_batches <= 0 || n < max_batches)) {
    Batch vb = val.next_batch();
    model_forward(model, vb.tokens.data(), batch, seq_len);
    gpu_copy_u16_to_device(d_targets, vb.targets.data(), rows);
    // dlogits are written to the gradient buffer as scratch; it is zeroed before backward
    total += gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                                    model.act(model.act_offsets.logits), d_targets,
                                    static_cast<int>(rows), model.config.vocab_size);
    ++n;
  }
  model.training = was_training;
  return n > 0 ? static_cast<float>(total / n) : NAN;
}

void save_checkpoint(const Model& model, const Adam& opt, const TrainConfig& tc) {
  std::string path = std::format("{}/checkpoint_step_{}.bin", tc.data_dir, opt.step);
  model_save_with_optimizer(model, opt, path);
  std::cout << std::format("  saved {}\n", path);

  auto all = list_checkpoints(tc.data_dir);
  int keep = std::max(tc.keep_checkpoints, 1);
  for (size_t i = 0; i + keep < all.size(); ++i) {
    fs::remove(all[i].second);
  }
}

}  // namespace

std::string find_latest_checkpoint(const std::string& dir) {
  auto all = list_checkpoints(dir);
  return all.empty() ? std::string() : all.back().second.string();
}

void run_training(const TrainConfig& tc) {
  std::cout << "=== dad-gpt train ===\n\n";

  DatasetMeta meta = read_dataset_meta(tc.data_dir);
  gpu_set_matmul_precision(tc.precision == "fp32"   ? MatmulPrecision::FP32
                           : tc.precision == "tf32" ? MatmulPrecision::TF32
                                                    : MatmulPrecision::BF16);
  Model model{};
  Adam opt;  // m and v live on GPU

  std::string resume_path = find_latest_checkpoint(tc.data_dir);
  if (!resume_path.empty()) {
    // Restores weights, Adam moments and step; buffers sized for the current --batch
    model_load_with_optimizer(model, opt, resume_path, tc.batch_size);
    std::cout << std::format("Resumed from {} (step {})\n", resume_path, opt.step);
    if (model.config.vocab_size != meta.vocab_size) {
      throw std::runtime_error(std::format("Checkpoint vocab {} != dataset vocab {} ({})",
                                           model.config.vocab_size, meta.vocab_size,
                                           meta.tokenizer));
    }
    // Don't replay the dropout masks of the first run (seeds advance once per dropout call)
    gpu_dropout_init(42ULL + static_cast<unsigned long long>(opt.step) * 1000003ULL, 0);
  } else {
    if (!tc.init_from.empty()) {
      if (meta.tokenizer != "gpt2") {
        throw std::runtime_error("--init-from needs data prepared with --tokenizer gpt2");
      }
      model_load_hf_gpt2(model, tc.init_from, tc.batch_size, tc.seq_len);
      std::cout << std::format("Initialized from Hugging Face GPT-2 in {}\n", tc.init_from);
    } else {
      model_alloc(model, build_config(tc, meta));
    }
    opt.num_params = model.param_offsets.total;
    opt.m = gpu_alloc(opt.num_params);
    opt.v = gpu_alloc(opt.num_params);
    gpu_zero(opt.m, opt.num_params);
    gpu_zero(opt.v, opt.num_params);
  }
  const ModelConfig& mc = model.config;
  int seq_len = tc.seq_len > 0 ? tc.seq_len : mc.ctx_len;
  if (seq_len > mc.ctx_len) {
    throw std::runtime_error(
        std::format("--seq-len {} exceeds the model's context {}", seq_len, mc.ctx_len));
  }
  model.training = true;
  model.dropout_rate = tc.dropout_rate;
  model.attention = tc.attention == "fused"          ? AttentionImpl::Fused
                    : tc.attention == "materialized" ? AttentionImpl::Materialized
                                                     : AttentionImpl::Auto;

  opt.max_lr = tc.max_lr;
  opt.warmup_steps = tc.warmup_steps;
  opt.weight_decay = tc.weight_decay;

  DataLoader train_loader(tc.data_dir + "/train.bin", tc.batch_size, seq_len);
  uint64_t val_tokens = DataLoader(tc.data_dir + "/val.bin", 1, seq_len).num_tokens();
  // Evaluate with the training batch size when the validation set is big enough
  int val_batch =
      val_tokens >= static_cast<uint64_t>(tc.batch_size) * (seq_len + 1) ? tc.batch_size : 1;
  DataLoader val_loader(tc.data_dir + "/val.bin", val_batch, seq_len);

  size_t batch_tokens = static_cast<size_t>(tc.batch_size) * seq_len;
  size_t tokens_per_step = batch_tokens * static_cast<size_t>(tc.accumulation_steps);
  int max_steps = tc.max_steps;
  if (max_steps <= 0) {
    max_steps =
        static_cast<int>(std::ceil(tc.epochs * static_cast<double>(train_loader.num_tokens()) /
                                   static_cast<double>(tokens_per_step)));
  }
  opt.total_steps = std::max(max_steps, 1);

  std::cout << std::format(
      "Model: {} params, d_model={}, n_layers={}, n_heads={}, ctx={}, "
      "vocab={} ({})\n",
      model.param_offsets.total, mc.d_model, mc.n_layers, mc.n_heads, mc.ctx_len, mc.vocab_size,
      meta.tokenizer);
  std::cout << std::format("Train tokens: {}, Val tokens: {}\n", train_loader.num_tokens(),
                           val_tokens);
  std::cout << std::format(
      "Batch {} x {} x accum {} = {} tokens/step, {} steps "
      "({:.2f} epochs), lr {} warmup {}\n\n",
      tc.batch_size, seq_len, tc.accumulation_steps, tokens_per_step, max_steps,
      static_cast<double>(max_steps) * tokens_per_step /
          static_cast<double>(train_loader.num_tokens()),
      tc.max_lr, tc.warmup_steps);

  if (opt.step >= max_steps) {
    std::cout << std::format("Already trained to step {} (max_steps {}).\n", opt.step, max_steps);
  }

  uint16_t* d_targets = gpu_alloc_u16(batch_tokens);
  // A different random stream per resume point, so resumed runs don't revisit the same batches
  std::mt19937_64 rng(tc.seed ^ (static_cast<uint64_t>(opt.step) * 0x9E3779B97F4A7C15ULL));

  double interval_loss = 0.0;
  int interval_steps = 0;
  const int start_step = opt.step;
  int consecutive_skips = 0;
  auto interval_start = std::chrono::steady_clock::now();

  while (opt.step < max_steps) {
    gpu_zero(model.gradients, model.param_offsets.total);
    float step_loss = 0.0f;

    for (int micro = 0; micro < tc.accumulation_steps; ++micro) {
      Batch batch = train_loader.random_batch(rng);
      // Forward overwrites every activation, but backward accumulates (+=) into activation
      // gradients, so only those need clearing per micro-batch
      gpu_zero(model.grad_activations, model.act_offsets.total);

      model_forward(model, batch.tokens.data(), tc.batch_size, seq_len);
      gpu_copy_u16_to_device(d_targets, batch.targets.data(), batch_tokens);
      // Loss + dlogits on GPU (averaged over B*S positions)
      step_loss += gpu_cross_entropy_loss(model.grad_act(model.act_offsets.logits),
                                          model.act(model.act_offsets.logits), d_targets,
                                          static_cast<int>(batch_tokens), mc.vocab_size);
      // Gradients accumulate across micro-batches
      model_backward(model, batch.tokens.data(), tc.batch_size, seq_len);
    }

    if (tc.accumulation_steps > 1) {
      float inv_accum = 1.0f / static_cast<float>(tc.accumulation_steps);
      gpu_gradient_scale(model.gradients, model.param_offsets.total, inv_accum);
      step_loss *= inv_accum;
    }

    // A non-finite norm would poison the params and Adam state: skip the update instead
    float norm = gpu_gradient_norm(model.gradients, model.param_offsets.total);
    if (!std::isfinite(norm) || !std::isfinite(step_loss)) {
      std::cerr << std::format("  step {:>6d} | non-finite grad norm/loss, skipping update\n",
                               opt.step + 1);
      if (++consecutive_skips >= 20) {
        throw std::runtime_error("20 consecutive non-finite steps; aborting");
      }
      continue;
    }
    consecutive_skips = 0;
    if (norm > 1.0f) {
      gpu_gradient_scale(model.gradients, model.param_offsets.total, 1.0f / norm);
    }

    ++opt.step;
    float lr = adam_lr(opt);
    gpu_adamw_step(model.params, model.gradients, opt.m, opt.v, model.decay_flags,
                   model.param_offsets.total, lr, opt.beta1, opt.beta2, opt.eps, opt.weight_decay,
                   opt.step);

    interval_loss += step_loss;
    ++interval_steps;
    if (interval_steps == 1 && opt.step == start_step + 1) {
      // All buffers (including lazily allocated attention ones) exist after the first step.
      // When they exceed VRAM, the Windows driver's sysmem fallback pages to system RAM
      // instead of failing, which silently makes training several times slower.
      size_t free_b = 0, total_b = 0;
      gpu_memory_info(free_b, total_b);
      std::cout << std::format("  GPU memory: {:.2f} / {:.2f} GB used\n", (total_b - free_b) / 1e9,
                               total_b / 1e9);
      if (free_b < total_b / 50) {
        std::cerr << "  warning: GPU memory is (nearly) full. If the driver spills to system "
                     "RAM, training slows down sharply; reduce --batch or use --attention fused.\n";
      }
    }
    if (opt.step % tc.log_interval == 0) {
      auto now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(now - interval_start).count();
      std::cout << std::format(
          "  step {:>6d} | loss {:.4f} | lr {:.2e} | norm {:.3f} | {:.0f} ms/step | {:.0f} tok/s\n",
          opt.step, interval_loss / interval_steps, lr, norm, 1000.0 * secs / interval_steps,
          static_cast<double>(tokens_per_step) * interval_steps / secs);
      interval_loss = 0.0;
      interval_steps = 0;
      interval_start = now;
    }

    bool last = opt.step == max_steps;
    if (opt.step % tc.eval_interval == 0 || last) {
      float val = evaluate(model, val_loader, d_targets, val_batch, seq_len, tc.eval_batches);
      std::cout << std::format("  step {:>6d} | val_loss {:.4f}\n", opt.step, val);
    }
    if ((tc.checkpoint_interval > 0 && opt.step % tc.checkpoint_interval == 0) || last) {
      save_checkpoint(model, opt, tc);
    }
  }

  gpu_free(d_targets);
  gpu_free(opt.m);
  gpu_free(opt.v);
  model_free(model);
  std::cout << "Training complete!\n";
}
