# Test executable
# New test .cpp files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(dad-gpt-tests ${TEST_SOURCES})
target_link_libraries(dad-gpt-tests PRIVATE GTest::gtest_main dad-gpt-cuda)
target_include_directories(dad-gpt-tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(dad-gpt-tests)
