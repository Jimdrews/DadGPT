# Library of all source files except main.cpp (shared between executable and tests)
file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS "*.cpp")
list(FILTER LIB_SOURCES EXCLUDE REGEX "main\\.cpp$")

add_library(${PROJECT_NAME}-lib STATIC ${LIB_SOURCES})
target_include_directories(${PROJECT_NAME}-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME}-lib PUBLIC nlohmann_json::nlohmann_json)

if(DAD_GPT_CUDA_ENABLED)
  target_link_libraries(${PROJECT_NAME}-lib PUBLIC ${PROJECT_NAME}-cuda)
  target_compile_definitions(${PROJECT_NAME}-lib PUBLIC DAD_GPT_CUDA_ENABLED)
endif()

# Main executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}-lib)
