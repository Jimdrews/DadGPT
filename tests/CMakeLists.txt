# Test executable
# New test .cpp files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(${PROJECT_NAME}-tests ${TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}-tests PRIVATE GTest::gtest_main ${PROJECT_NAME}-lib)

include(GoogleTest)
# PRE_TEST: discover at ctest time, so building never needs the CUDA runtime DLLs on PATH
gtest_discover_tests(${PROJECT_NAME}-tests DISCOVERY_MODE PRE_TEST)

# Lets tests find committed fixtures and the (gitignored) data/gpt2 model files
target_compile_definitions(${PROJECT_NAME}-tests PRIVATE DADGPT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
