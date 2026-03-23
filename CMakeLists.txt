cmake_minimum_required(VERSION 3.18)

project(Sparsity
    VERSION 0.1.0
    DESCRIPTION "Natively hybrid C++ vector similarity search library"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(WITH_CUDA         "Enable CUDA acceleration"   OFF)
option(BUILD_TESTS       "Build unit tests"            ON)
option(BUILD_BENCHMARKS  "Build benchmarks"            ON)
option(BUILD_PYTHON      "Build Python bindings"       OFF)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------
if(WITH_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(WITH_CUDA)
    message(STATUS "CUDA enabled — toolkit: ${CUDAToolkit_VERSION}")
else()
    message(STATUS "CUDA disabled (pass -DWITH_CUDA=ON to enable)")
endif()

# ---------------------------------------------------------------------------
# Core library sources
# ---------------------------------------------------------------------------
set(SPARSITY_SOURCES
    src/core/metrics/l2.cpp
    src/core/metrics/cosine.cpp
    src/core/metrics/tanimoto.cpp
    src/core/metrics/sparse.cpp
    src/core/packing.cpp
    src/core/sparse.cpp
    src/core/binary.cpp
    src/core/blocked_ell.cpp
    src/core/index.cpp
    src/core/dispatch.cpp
    src/core/kmeans.cpp
    src/core/ivf.cpp
    src/core/tiled_dense.cpp
)

if(WITH_CUDA)
    list(APPEND SPARSITY_SOURCES
        src/cuda/metrics/l2.cu
        src/cuda/metrics/l2_tiled.cu
        src/cuda/metrics/cosine.cu
        src/cuda/metrics/tanimoto.cu
        src/cuda/topk.cu
        src/cuda/sparse_dot.cu
    )
endif()

add_library(sparsity STATIC ${SPARSITY_SOURCES})

target_include_directories(sparsity PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(sparsity PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

if(WITH_CUDA)
    target_link_libraries(sparsity PRIVATE CUDA::cudart)
    target_compile_options(sparsity PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3>
    )
endif()

# ---------------------------------------------------------------------------
# clang-format target
# ---------------------------------------------------------------------------
find_program(CLANG_FORMAT clang-format)
if(CLANG_FORMAT)
    file(GLOB_RECURSE ALL_CXX_FILES
        "${CMAKE_SOURCE_DIR}/include/*.h"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/src/*.cu"
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_SOURCE_DIR}/python/bindings/*.cpp"
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT} -i ${ALL_CXX_FILES}
        COMMENT "Running clang-format on all source files"
        VERBATIM
    )
else()
    message(STATUS "clang-format not found — 'make format' target unavailable")
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()
