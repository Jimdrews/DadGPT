# CUDA kernel library
# New .cu files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "*.cu")

add_library(${PROJECT_NAME}-cuda STATIC ${CUDA_SOURCES})
target_include_directories(${PROJECT_NAME}-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}-cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
