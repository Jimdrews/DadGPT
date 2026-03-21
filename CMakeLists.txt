cmake_minimum_required(VERSION 3.28)
project(dad-gpt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

enable_testing()

add_subdirectory(cuda)
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
        $<TARGET_FILE:dad-gpt-tests>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running valgrind memcheck on tests"
    DEPENDS dad-gpt-tests
    VERBATIM
)

# GPU profiling (not a target — run manually):
#   compute-sanitizer --tool memcheck ./build/default/src/dad-gpt
#   compute-sanitizer --tool racecheck ./build/default/src/dad-gpt
