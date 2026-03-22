#include "csv_parser.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// Parse a single CSV field starting at pos. Handles quoted fields with escaped quotes ("").
// Returns the field value and advances pos past the field (including trailing comma or newline).
std::string parse_field(const std::string& data, size_t& pos) {
  if (pos >= data.size()) return "";

  if (data[pos] == '"') {
    // Quoted field
    ++pos;  // skip opening quote
    std::string field;
    while (pos < data.size()) {
      if (data[pos] == '"') {
        if (pos + 1 < data.size() && data[pos + 1] == '"') {
          // Escaped quote
          field += '"';
          pos += 2;
        } else {
          // Closing quote
          ++pos;
          break;
        }
      } else {
        field += data[pos];
        ++pos;
      }
    }
    // Skip trailing comma or newline
    if (pos < data.size() && (data[pos] == ',' || data[pos] == '\n' || data[pos] == '\r')) {
      if (data[pos] == '\r' && pos + 1 < data.size() && data[pos + 1] == '\n') {
        pos += 2;
      } else {
        ++pos;
      }
    }
    return field;
  }

  // Unquoted field
  std::string field;
  while (pos < data.size() && data[pos] != ',' && data[pos] != '\n' && data[pos] != '\r') {
    field += data[pos];
    ++pos;
  }
  if (pos < data.size() && (data[pos] == ',' || data[pos] == '\n' || data[pos] == '\r')) {
    if (data[pos] == '\r' && pos + 1 < data.size() && data[pos + 1] == '\n') {
      pos += 2;
    } else {
      ++pos;
    }
  }
  return field;
}

}  // namespace

std::vector<Joke> parse_jokes_csv(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Could not open " + path +
        "\nDownload from: "
        "https://huggingface.co/datasets/shuttie/dadjokes/resolve/main/train.csv");
  }

  // Read entire file into memory (small enough)
  std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();

  size_t pos = 0;

  // Skip header row
  std::string h1 = parse_field(data, pos);
  std::string h2 = parse_field(data, pos);

  std::vector<Joke> jokes;
  while (pos < data.size()) {
    // Skip any trailing whitespace/newlines at end of file
    if (pos < data.size() && (data[pos] == '\n' || data[pos] == '\r')) {
      ++pos;
      continue;
    }
    if (pos >= data.size()) break;

    std::string question = parse_field(data, pos);
    std::string response = parse_field(data, pos);

    if (!question.empty() || !response.empty()) {
      jokes.push_back({std::move(question), std::move(response)});
    }
  }

  return jokes;
}
