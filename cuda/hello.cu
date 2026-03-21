#include "hello.cuh"

#include <cuda_runtime.h>
#include <cstdio>

__global__ void hello_kernel(int* result) {
    *result = 42;
}

bool cuda_hello(int& device_count, std::string& device_name) {
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    device_name = props.name;

    int* d_result = nullptr;
    int h_result = 0;

    cudaMalloc(&d_result, sizeof(int));
    hello_kernel<<<1, 1>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result == 42;
}
