#include <iostream>
#include <chrono>
#include "ops.h"
#include "kernels.h"






void benchmark_opv2_bacd_cdab(uint64_t const reps) {
    std::vector<float> in0(db * da * dc * dd, 0.0f);
    std::vector<float> out(db * da * dc * dd, 0.0f);
    
    for (uint64_t i = 0; i < 10; i++) {
        opv2_bacd_cdab(in0.data(), out.data());
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        opv2_bacd_cdab(in0.data(), out.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    uint64_t bytes_transferred_per_rep = 2 * da * db * dc * dd;
    double gibs_transferred = reps * (bytes_transferred_per_rep / (1024.0 * 1024.0 * 1024.0));
    double gibs = gibs_transferred / duration.count();

    std::cout << gibs << "\t\t" << gibs_transferred << "\t\t" << duration.count() << "\t\t" << duration.count() / reps << std::endl;
}





void benchmark_opv2_cdab_cdef_efab(uint64_t const reps) {
    std::vector<float> in0(dc * dd * da * db, 0.0f);
    std::vector<float> in1(dc * dd * de * df, 0.0f);
    std::vector<float> out(de * df * da * db, 0.0f);

    opv2_cdab_cdef_efab(in0.data(), in1.data(), out.data());
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        opv2_cdab_cdef_efab(in0.data(), in1.data(), out.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    uint64_t flops_computed_per_rep = de * df * da * db * dc * dd * 2;
    double gflops_computed = reps * (flops_computed_per_rep * 1e-9);
    double gflops = gflops_computed / duration.count();

    std::cout << gflops << "\t\t" << gflops_computed << "\t\t" << duration.count() << "\t\t" << duration.count() / reps << std::endl;
}


void benchmark_opv2_efgh_efab_gabh(uint64_t const reps) {
    std::vector<float> in0(de * df * dg * dh, 0.0f);
    std::vector<float> in1(de * df * da * db, 0.0f);
    std::vector<float> out(dg * da * db * dh, 0.0f);

    opv2_efgh_efab_gabh(in0.data(), in1.data(), out.data());
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        opv2_efgh_efab_gabh(in0.data(), in1.data(), out.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    uint64_t flops_computed_per_rep = dg * da * db * dh * de * df * 2;
    double gflops_computed = reps * (flops_computed_per_rep * 1e-9);
    double gflops = gflops_computed / duration.count();

    std::cout << gflops << "\t\t" << gflops_computed << "\t\t" << duration.count() << "\t\t" << duration.count() / reps << std::endl;
}






void benchmark_opv2(uint64_t const reps) {
    std::vector<float> efgh(de * df * dg * dh, 0.0f);
    std::vector<float> bacd(db * da * dc * dd, 0.0f);
    std::vector<float> cdef(dc * dd * de * df, 0.0f);
    std::vector<float> cdab(dc * dd * da * db, 0.0f);
    std::vector<float> efab(de * df * da * db, 0.0f);
    std::vector<float> gabh(dg * da * db * dh, 0.0f);

    opv2_bacd_cdab(bacd.data(), cdab.data());
    opv2_cdab_cdef_efab(cdab.data(), cdef.data(), efab.data());
    opv2_efgh_efab_gabh(efgh.data(), efab.data(), gabh.data());
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        opv2_bacd_cdab(bacd.data(), cdab.data());
        opv2_cdab_cdef_efab(cdab.data(), cdef.data(), efab.data());
        opv2_efgh_efab_gabh(efgh.data(), efab.data(), gabh.data());
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double gflops_computed_efab = reps * ((2 * de * df * da * db * dc * dd) * 1e-9);
    double gflops_computed_gabh = reps * ((2 * dg * da * db * dh * de * df) * 1e-9);
    double gflops_computed = gflops_computed_efab + gflops_computed_gabh;
    double gflops = gflops_computed / duration.count();

    std::cout << gflops << "\t\t" << gflops_computed << "\t\t" << duration.count() << "\t\t" << duration.count() / reps << std::endl;
}










int main() {
    generate_kernels();

    std::cout << "GiBs\t\tBytes transferred [GiB]\t\tDuration [s]\t\tAverage time per rep [s]" <<  std::endl;
    benchmark_opv2_bacd_cdab(250);

    std::cout << std::endl;
    std::cout << "GFlops\t\tFlops computed [GFlops]\t\tDuration [s]\t\tAverage time per rep [s]" << std::endl;
    benchmark_opv2_cdab_cdef_efab(5);
    benchmark_opv2_efgh_efab_gabh(5);
    benchmark_opv2(5);

    return 0;
}

