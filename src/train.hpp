#pragma once

#include <string>

struct TrainConfig {
  std::string data_dir = "data";
  int epochs = 100;
  int batch_size = 64;
  int seq_len = 128;
  int log_interval = 10;  // print loss every N steps
};

void run_training(const TrainConfig& config);
