#include <format>
#include <iostream>
#include <string>

#include "hello.cuh"

int main() {
    std::cout << std::format("dad-gpt v0.1.0 | C++{}\n", __cplusplus);

    int device_count = 0;
    std::string device_name;

    if (cuda_hello(device_count, device_name)) {
        std::cout << std::format("CUDA: {} device(s) found\n", device_count);
        std::cout << std::format("GPU:  {}\n", device_name);
        std::cout << "Kernel test: passed\n";
    } else {
        std::cerr << "CUDA: no devices found or kernel failed\n";
        return 1;
    }

    return 0;
}
