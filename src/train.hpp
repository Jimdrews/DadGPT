#pragma once

#include <string>

struct TrainConfig {
  std::string data_dir = "data";
  int epochs = 20;
  int batch_size = 32;
  int seq_len = 128;
  int log_interval = 10;  // print loss every N steps
};

void run_training(const TrainConfig& config);
