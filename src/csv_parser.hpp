#pragma once

#include <string>
#include <utility>
#include <vector>

struct Joke {
  std::string question;
  std::string response;
};

std::vector<Joke> parse_jokes_csv(const std::string& path);
