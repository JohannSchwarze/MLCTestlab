#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>

#include "ops.h"
#include "kernels.h"





float max_abs_diff(float const* in0, float const* in1, uint64_t const size) {
    float result = 0.0;
    for (uint64_t i = 0; i < size; ++i) {
        float diff = abs(in0[i] - in1[i]);
        result = std::max(result, diff);
    }
    return result;
}



void load_tensor(float* out, uint64_t size, const char* fp) {
    std::ifstream file(fp, std::ios::binary);
    file.read((char*) out, size * sizeof(float));
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

TEST_CASE("test transpose 2", "[test]") {
    transposefunction func(3, 3, 6, 6);

    std::vector<float> in0 = {
        1, 2, 3, 0, 0, 0,
        4, 5, 6, 0, 0, 0,
        7, 8, 9, 0, 0, 0,
    };
    std::vector<float> out(3 * 6, 0.0f);
    std::vector<float> out_exp = {
        1, 4, 7, 0, 0, 0,
        2, 5, 8, 0, 0, 0,
        3, 6, 9, 0, 0, 0,
    };

    func(in0.data(), out.data());

    bool result = out == out_exp;
    REQUIRE(result);
}


TEST_CASE("test bacd->cdab", "[test]") {
    std::vector<float> bacd(db * da * dc * dd, 0.0f);
    std::vector<float> cdab(dc * dd * da * db, 0.0f);
    std::vector<float> cdab_exp(dc * dd * da * db, 0.0f);

    load_tensor(bacd.data(), bacd.size(), "data/bacd_cdab/bacd.bin");
    load_tensor(cdab_exp.data(), cdab_exp.size(), "data/bacd_cdab/cdab.bin");

    opv2_bacd_cdab(bacd.data(), cdab.data());

    float diff = max_abs_diff(cdab.data(), cdab_exp.data(), cdab.size());
    REQUIRE(diff < 1e-6);
}



TEST_CASE("test cdab_cdef_efab", "[test]") {
    std::vector<float> cdab(dc * dd * da * db, 0.0f);
    std::vector<float> cdef(dc * dd * de * df, 0.0f);
    std::vector<float> efab(de * df * da * db, 0.0f);
    std::vector<float> efab_exp(de * df * da * db, 0.0f);

    load_tensor(cdab.data(), cdab.size(), "data/cdab_cdef_efab/cdab.bin");
    load_tensor(cdef.data(), cdef.size(), "data/cdab_cdef_efab/cdef.bin");
    load_tensor(efab_exp.data(), efab_exp.size(), "data/cdab_cdef_efab/efab.bin");

    opv2_cdab_cdef_efab(cdab.data(), cdef.data(), efab.data());

    float diff = max_abs_diff(efab.data(), efab_exp.data(), efab.size());
    REQUIRE(diff < 1e-6);
}

TEST_CASE("test efgh_efab_gabh", "[test]") {
    std::vector<float> efgh(de * df * dg * dh, 0.0f);
    std::vector<float> efab(de * df * da * db, 0.0f);
    std::vector<float> gabh(dg * da * db * dh, 0.0f);
    std::vector<float> gabh_exp(dg * da * db * dh, 0.0f);

    load_tensor(efgh.data(), efgh.size(), "data/efgh_efab_gabh/efgh.bin");
    load_tensor(efab.data(), efab.size(), "data/efgh_efab_gabh/efab.bin");
    load_tensor(gabh_exp.data(), gabh_exp.size(), "data/efgh_efab_gabh/gabh.bin");

    opv2_efgh_efab_gabh(efgh.data(), efab.data(), gabh.data());

    float diff = max_abs_diff(gabh.data(), gabh_exp.data(), gabh.size());
    REQUIRE(diff < 1e-6);
}




TEST_CASE("test v2", "[test]") {
    std::vector<float> bacd(db * da * dc * dd, 0.0f);
    std::vector<float> cdab(dc * dd * da * db, 0.0f);
    std::vector<float> cdef(dc * dd * de * df, 0.0f);
    std::vector<float> efab(de * df * da * db, 0.0f);
    std::vector<float> efgh(de * df * dg * dh, 0.0f);
    std::vector<float> gabh(dg * da * db * dh, 0.0f);
    std::vector<float> gabh_exp(dg * da * db * dh, 0.0f);

    load_tensor(bacd.data(), bacd.size(), "data/main/bacd.bin");
    load_tensor(cdef.data(), cdef.size(), "data/main/cdef.bin");
    load_tensor(efgh.data(), efgh.size(), "data/main/efgh.bin");
    load_tensor(gabh_exp.data(), gabh_exp.size(), "data/main/gabh.bin");

    opv2_bacd_cdab(bacd.data(), cdab.data());
    opv2_cdab_cdef_efab(cdab.data(), cdef.data(), efab.data());
    opv2_efgh_efab_gabh(efgh.data(), efab.data(), gabh.data());

    float diff = max_abs_diff(gabh.data(), gabh_exp.data(), gabh.size());
    REQUIRE(diff < 1e-6);
}





int main(int argc, char* argv[]) {
    generate_kernels();
    return Catch::Session().run(argc, argv);
}











