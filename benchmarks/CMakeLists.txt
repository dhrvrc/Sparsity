add_executable(bench_dense bench_dense.cpp)

target_link_libraries(bench_dense PRIVATE sparsity)

target_compile_options(bench_dense PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

add_executable(bench_gpu bench_gpu.cpp)

target_link_libraries(bench_gpu PRIVATE sparsity)

target_compile_options(bench_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

add_executable(bench_3d bench_3d.cpp)

target_link_libraries(bench_3d PRIVATE sparsity)

target_compile_options(bench_3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)
