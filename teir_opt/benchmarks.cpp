#include <iostream>

#include "opts.h"
#include "kernels.h"

typedef void(op_func_t)(float const*, float const*, float*);







void benchmark_operation(op_func_t* op, uint64_t const reps) {
    std::vector<float> in0(in0_size, 0.0f);
    std::vector<float> in1(in1_size, 0.0f);
    std::vector<float> out(out_size, 0.0f);

    for (uint64_t i = 0; i < 10; i++) {
        op(in0.data(), in1.data(), out.data());
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        op(in0.data(), in1.data(), out.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double gflops = ((reps * 1e-9) * flops) / duration.count();

    std::cout << gflops << "\t\t";
    std::cout << duration.count() << "\t\t";
    std::cout << std::endl;
}













int main() {
    generate_kernels();

    uint64_t reps = 100;

    std::cout << "GFlops\t\tDuration [s]\t\t" << std::endl;
    benchmark_operation(&abcd_ebf_aefcd_v1, 10);
    benchmark_operation(&abcd_ebf_aefcd_v2, 10000);
    benchmark_operation(&abcd_ebf_aefcd_v2_parallel, 10000);
    benchmark_operation(&abcd_ebf_aefcd_v2_reordered, 10000);
    benchmark_operation(&abcd_ebf_aefcd_v4, 10000);

    return 0;
}
