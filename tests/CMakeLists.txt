find_package(GTest REQUIRED)

# Helper: add a test executable linked against sparsity + gtest
function(sparsity_add_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE sparsity GTest::gtest_main)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_subdirectory(metrics)
