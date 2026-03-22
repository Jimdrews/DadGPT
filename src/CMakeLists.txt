# Main executable
# New .cpp files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)

if(DAD_GPT_CUDA_ENABLED)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}-cuda)
  target_compile_definitions(${PROJECT_NAME} PRIVATE DAD_GPT_CUDA_ENABLED)
endif()
