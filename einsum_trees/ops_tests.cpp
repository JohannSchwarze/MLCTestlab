#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#include "ops.h"
#include "kernels.h"





double max_abs_diff(float const* in0, float const* in1, uint64_t const size) {
    double result = 0.0;
    for (uint64_t i = 0; i < size; ++i) {
        double diff = abs(in0[i] - in1[i]);
        result = std::max(result, diff);
    }
    return result;
}



void load_tensor(float* out, uint64_t size, const char* fp) {
    std::ifstream file(fp, std::ios::binary);
    file.read((char*) out, size);
}






TEST_CASE("test v1", "[test]") {
    /*
    std::vector<float> bacd(db * da * dc * dd, 0.0f);
    std::vector<float> cdef(dc * dd * de * df, 0.0f);
    std::vector<float> abef(da * db * de * df, 0.0f);
    std::vector<float> efgh(de * df * dg * dh, 0.0f);
    std::vector<float> gabh(dg * da * db * dh, 0.0f);
    std::vector<float> gabh_exp(dg * da * db * dh, 0.0f);

    load_tensor(bacd.data(), bacd.size(), "data/bacd.bin");
    load_tensor(cdef.data(), cdef.size(), "data/cdef.bin");
    load_tensor(efgh.data(), efgh.size(), "data/efgh.bin");
    load_tensor(gabh_exp.data(), gabh_exp.size(), "data/gabh.bin");

    opv1_bacd_cdef_abef(bacd.data(), cdef.data(), abef.data());
    opv1_abef_efgh_gabh(abef.data(), efgh.data(), gabh.data());

    double diff = max_abs_diff(gabh.data(), gabh_exp.data(), gabh.size());
    REQUIRE(diff < 1e-5);
    */
}


TEST_CASE("test transpose", "[test]") {
    transposefunction func(3, 3, 3, 3);

    std::vector<float> in0 = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    std::vector<float> out(3 * 3, 0.0f);
    std::vector<float> out_exp = {
        1, 4, 7,
        2, 5, 8,
        3, 6, 9,
    };

    func(in0.data(), out.data());

    bool result = out == out_exp;
    REQUIRE(result);
}


TEST_CASE("test v2", "[test]") {
    std::vector<float> bacd(db * da * dc * dd, 0.0f);
    std::vector<float> cdab(dc * dd * da * db, 0.0f);
    std::vector<float> cdef(dc * dd * de * df, 0.0f);
    std::vector<float> efab(de * df * da * db, 0.0f);
    std::vector<float> efgh(de * df * dg * dh, 0.0f);
    std::vector<float> gabh(dg * da * db * dh, 0.0f);
    std::vector<float> gabh_exp(dg * da * db * dh, 0.0f);

    load_tensor(bacd.data(), bacd.size(), "data/bacd.bin");
    load_tensor(cdef.data(), cdef.size(), "data/cdef.bin");
    load_tensor(efgh.data(), efgh.size(), "data/efgh.bin");
    load_tensor(gabh_exp.data(), gabh_exp.size(), "data/gabh.bin");

    opv2_bacd_cdab(bacd.data(), cdab.data());
    opv2_cdab_cdef_efab(cdab.data(), cdef.data(), efab.data());
    opv2_efgh_efab_gabh(efgh.data(), efab.data(), gabh.data());

    std::cout << gabh[0] << std::endl;
    std::cout << gabh_exp[0] << std::endl;

    double diff = max_abs_diff(gabh.data(), gabh_exp.data(), gabh.size());
    REQUIRE(diff < 1e-5);
}





int main(int argc, char* argv[]) {
    generate_kernels();
    return Catch::Session().run(argc, argv);
}











