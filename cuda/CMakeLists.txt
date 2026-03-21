# CUDA kernel library
# New .cu files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "*.cu")

add_library(dad-gpt-cuda STATIC ${CUDA_SOURCES})
target_include_directories(dad-gpt-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dad-gpt-cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
