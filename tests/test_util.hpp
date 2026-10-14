#pragma once

#include <atomic>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <process.h>
#define DAD_GPT_GETPID _getpid
#else
#include <unistd.h>
#define DAD_GPT_GETPID getpid
#endif

namespace test_util {

// A unique path in the system temp directory. ctest runs each test in its own process,
// possibly in parallel, so fixed names like /tmp/x.bin would race.
inline std::string temp_path(const std::string& name) {
  static std::atomic<int> counter{0};
  auto file = std::to_string(DAD_GPT_GETPID()) + "_" + std::to_string(counter++) + "_" + name;
  return (std::filesystem::temp_directory_path() / ("dadgpt_test_" + file)).string();
}

}  // namespace test_util
