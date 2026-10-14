#pragma once

#include <cstdint>

// Backward for C = A × B: dA += dC × B^T, dB += A^T × dC
void matmul_backward(const float* dout, const float* a, const float* b, float* da, float* db, int M,
                     int K, int N);

// Backward for row-wise softmax: din_i = out_i * (dout_i - sum(dout * out))
void softmax_backward(const float* dout, const float* out, float* din, int rows, int cols);

// Backward for layernorm
void layernorm_backward(const float* dout, const float* in, const float* weight, float* din,
                        float* dweight, float* dbias, int rows, int cols, float eps = 1e-5f);

// Backward for GELU (tanh approximation)
void gelu_backward(const float* dout, const float* in, float* din, int size);

// Backward for embedding: scatter-add dout into dtable at token positions
void embedding_backward(const float* dout, const uint16_t* tokens, float* dtable, int num_tokens,
                        int d_model);

// Cross-entropy loss and its gradient w.r.t. logits
// Returns the average loss. Writes dlogits = softmax(logits) - one_hot(target)
float cross_entropy_loss(const float* logits, const uint16_t* targets, float* dlogits, int seq_len,
                         int vocab_size);
