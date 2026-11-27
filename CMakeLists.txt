cmake_minimum_required(VERSION 3.12)
project(MiniML-CPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ASAN "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
# Step 6 features are opt-in headers under include/; this option gates their
# test targets so a downstream build can omit them entirely.
option(ENABLE_O3_EXTRAS "Build opt-in scheduler/RMSprop tests" ON)

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

# Step 3: deterministic mini-batch loader.
add_executable(test_minibatch tests/test_minibatch.cpp)
target_link_libraries(test_minibatch PRIVATE miniml_core)
set_target_properties(test_minibatch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_test(NAME minibatch_loader COMMAND test_minibatch
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Step 5: richer binary-classification metrics.
add_executable(test_metrics tests/test_metrics.cpp)
target_link_libraries(test_metrics PRIVATE miniml_core)
set_target_properties(test_metrics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_test(NAME metrics_binary COMMAND test_metrics
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

# Step 7: negative control for the tightened gradcheck gate.
add_test(NAME gradcheck_falsification COMMAND test_gradcheck --falsify
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(gradcheck_falsification PROPERTIES WILL_FAIL TRUE)

# Step 6: opt-in schedulers and RMSprop.
if(ENABLE_O3_EXTRAS)
    add_executable(test_scheduler tests/test_scheduler.cpp)
    target_link_libraries(test_scheduler PRIVATE miniml_core)
    set_target_properties(test_scheduler PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME scheduler_lr COMMAND test_scheduler
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_executable(test_rmsprop tests/test_rmsprop.cpp)
    target_link_libraries(test_rmsprop PRIVATE miniml_core)
    set_target_properties(test_rmsprop PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    add_test(NAME rmsprop_toy COMMAND test_rmsprop
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()
