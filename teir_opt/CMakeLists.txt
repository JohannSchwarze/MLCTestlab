cmake_minimum_required(VERSION 3.10)
project(TEIR_OPT)

Include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)


set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O3 -fopenmp)



add_executable(tests 
    tests.cpp 
    opts.cpp
    kernels.cpp
)
target_include_directories(tests PRIVATE libxsmm/include /opt/homebrew/opt/libomp/include)
target_link_directories(tests PRIVATE /opt/homebrew/opt/libomp/lib libxsmm/lib)
target_link_libraries(tests PRIVATE Catch2::Catch2 xsmm omp)
target_compile_options(tests PRIVATE ${CXX_FLAGS})

add_executable(benchmarks
    benchmarks.cpp
    opts.cpp
    kernels.cpp
)
target_include_directories(benchmarks PRIVATE libxsmm/include /opt/homebrew/opt/libomp/include)
target_link_directories(benchmarks PRIVATE /opt/homebrew/opt/libomp/lib libxsmm/lib)
target_compile_options(benchmarks PRIVATE ${CXX_FLAGS})
target_link_libraries(benchmarks PRIVATE xsmm omp)
