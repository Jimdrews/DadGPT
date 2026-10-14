cmake_minimum_required(VERSION 3.28)
project(dad-gpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA is required: the model, training and generation all run on the GPU.
# Override the target GPU with -DCMAKE_CUDA_ARCHITECTURES=<sm> (default 86, RTX 30xx).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

if(MSVC)
  # Sources contain UTF-8 comments; keep M_PI-style macros and min/max sane on Windows
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(cuda)

find_package(GTest CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(PCRE2 CONFIG REQUIRED COMPONENTS 8BIT)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)

# --- Developer tooling targets ---

# Collect all source files for formatting/linting
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp" "src/*.hpp"
    "cuda/*.cu" "cuda/*.cuh"
    "tests/*.cpp" "tests/*.hpp"
)
file(GLOB_RECURSE CXX_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp" "src/*.hpp"
    "tests/*.cpp" "tests/*.hpp"
)

# format: reformat all source files in place
add_custom_target(format
    COMMAND clang-format-18 -i ${ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on all source files"
    VERBATIM
)

# format-check: CI-friendly check that fails if any file is unformatted
add_custom_target(format-check
    COMMAND clang-format-18 --dry-run --Werror ${ALL_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking formatting (fails if any file needs formatting)"
    VERBATIM
)

# lint: run clang-tidy on C++ source files
add_custom_target(lint
    COMMAND clang-tidy-18 -p ${CMAKE_BINARY_DIR} ${CXX_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-tidy on C++ source files"
    VERBATIM
)

# memcheck: run valgrind on the test executable
add_custom_target(memcheck
    COMMAND valgrind --tool=memcheck --leak-check=full
        --suppressions=${CMAKE_SOURCE_DIR}/valgrind.supp
        $<TARGET_FILE:${PROJECT_NAME}-tests>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running valgrind memcheck on tests"
    DEPENDS ${PROJECT_NAME}-tests
    VERBATIM
)

# gpu-memcheck: run compute-sanitizer memcheck on the test executable
add_custom_target(gpu-memcheck
    COMMAND compute-sanitizer --tool memcheck --leak-check full
        $<TARGET_FILE:${PROJECT_NAME}-tests>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running compute-sanitizer memcheck on tests"
    DEPENDS ${PROJECT_NAME}-tests
    VERBATIM
)

# gpu-racecheck: detect shared-memory races in kernels (e.g. reductions missing __syncthreads)
add_custom_target(gpu-racecheck
    COMMAND compute-sanitizer --tool racecheck --racecheck-report all
        $<TARGET_FILE:${PROJECT_NAME}-tests>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running compute-sanitizer racecheck on tests"
    DEPENDS ${PROJECT_NAME}-tests
    VERBATIM
)
