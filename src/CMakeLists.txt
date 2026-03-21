# Main executable
# New .cpp files in this directory (or subdirectories) are picked up automatically.
# Re-run cmake configure after adding new files.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(dad-gpt ${SOURCES})
target_link_libraries(dad-gpt PRIVATE dad-gpt-cuda nlohmann_json::nlohmann_json)
