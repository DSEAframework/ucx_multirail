cmake_minimum_required(VERSION 3.10)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.30)
    cmake_policy(SET CMP0167 NEW)
endif()

# Minimum CUDA architecture is 61 because of use of atomicAdd(double *, double)
set(CMAKE_CUDA_ARCHITECTURES "61")

project(UCX_MULTIRAIL
    VERSION 0.1.0
    DESCRIPTION "UCX multirail proof of concept"
    LANGUAGES C CUDA
)

enable_language(CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find dependencies
find_package(CUDAToolkit REQUIRED)
find_package(UCX 1.17 REQUIRED)

# Output directory (optional)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Include directory
set(PROJECT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# UCX libraries
set(UCX_LIBS uct ucs ucp stdc++)

# Helper function to define executables
function(add_ucx_cuda_executable target_name source_path)
    set_source_files_properties(${source_path} PROPERTIES LANGUAGE CUDA)
    add_executable(${target_name} ${source_path})
    #target_compile_definitions(${target_name} PRIVATE DEBUG)
    target_link_libraries(${target_name} PRIVATE ${UCX_LIBS})
    target_include_directories(${target_name} PRIVATE ${PROJECT_INCLUDE_DIR})
endfunction()

# === Basic PoC ===
add_ucx_cuda_executable(mr_basic_receiver apps/ucx_mr_basic_recv.c)
add_ucx_cuda_executable(mr_basic_sender   apps/ucx_mr_basic_send.c)

# === Benchmark ===
add_ucx_cuda_executable(mr_bench_receiver apps/ucx_mr_bench_recv.c)
add_ucx_cuda_executable(mr_bench_sender   apps/ucx_mr_bench_send.c)