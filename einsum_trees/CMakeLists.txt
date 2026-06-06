cmake_minimum_required(VERSION 3.10)
project(EINSUM_TREES)





Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)








set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O0)









add_executable(ops_tests.out 
    ops_tests.cpp 
    ops.cpp
    kernels.cpp
)
target_include_directories(ops_tests.out PRIVATE libxsmm/include /opt/homebrew/opt/libomp/include)
target_link_directories(ops_tests.out PRIVATE /opt/homebrew/opt/libomp/lib libxsmm/lib)
target_link_libraries(ops_tests.out PRIVATE Catch2::Catch2 xsmm omp)
target_compile_options(ops_tests.out PRIVATE ${CXX_FLAGS})


