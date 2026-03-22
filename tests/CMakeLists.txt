# Test executable
# New test .cpp files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(${PROJECT_NAME}-tests ${TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}-tests PRIVATE GTest::gtest_main)

if(DAD_GPT_CUDA_ENABLED)
  target_link_libraries(${PROJECT_NAME}-tests PRIVATE ${PROJECT_NAME}-cuda)
  target_compile_definitions(${PROJECT_NAME}-tests PRIVATE DAD_GPT_CUDA_ENABLED)
endif()
target_include_directories(${PROJECT_NAME}-tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-tests)
