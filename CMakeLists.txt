cmake_minimum_required(VERSION 3.12)
project(MiniML-CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ASAN "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

add_compile_options(-Wall -Wextra)

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
endif()

# src/ is left untouched and unchanged: only tensor.cpp is a compiled translation
# unit; every other file under src/ is a header consumed directly.
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(miniml_core STATIC src/tensor.cpp)
target_include_directories(miniml_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

enable_testing()

add_executable(test_gradcheck tests/test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE miniml_core)
set_target_properties(test_gradcheck PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_serialization tests/test_serialization.cpp)
target_link_libraries(test_serialization PRIVATE miniml_core)
set_target_properties(test_serialization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(xor_demo examples/xor_demo.cpp)
target_link_libraries(xor_demo PRIVATE miniml_core)
set_target_properties(xor_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

add_test(NAME gradcheck COMMAND test_gradcheck
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
add_test(NAME serialization COMMAND test_serialization
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
