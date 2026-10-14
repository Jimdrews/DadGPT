#pragma once

#include <string>

#include "model.hpp"

// Loads a Hugging Face GPT-2 checkpoint directory (config.json + model.safetensors, fp32, e.g.
// openai-community/gpt2) into a freshly allocated model. The architecture comes from the
// files, so gpt2-medium/large/xl load too.
//
// Layout mapping: HF's Conv1D stores weights as [in, out], which is already our row-major
// W[K x N]; the fused attn.c_attn [d, 3d] is split column-wise into Wq, Wk, Wv. lm_head is
// tied to wte in both models.
//
// batch_size sizes the activation buffers. ctx_len (0 = the checkpoint's n_positions) may be
// smaller to save memory; only the first ctx_len position embeddings are kept.
void model_load_hf_gpt2(Model& model, const std::string& dir, int batch_size, int ctx_len = 0);
