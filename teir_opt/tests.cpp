#include <iostream>
#include <math.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include "kernels.h"
#include "opts.h"


double max_abs_diff(float const* a, float const* b, size_t size) {
    double max_diff = 0.0;
    for (size_t i = 0; i < size; ++i) {
        max_diff = std::max(max_diff, std::abs(static_cast<double>(a[i] - b[i])));
    }
    return max_diff;
}



typedef void(op_func_t)(float const*, float const*, float*);




void test_op_func(op_func_t* func) {
    std::vector<float> in0(in0_size);
    std::vector<float> in1(in1_size);
    std::vector<float> out(out_size);
    std::vector<float> out_ref(out_size);

    for (size_t i = 0; i < in0.size(); ++i) {
        in0[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (size_t i = 0; i < in1.size(); ++i) {
        in1[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = -1.0f;
        out_ref[i] = -1.0f;
    }

    abcd_ebf_aefcd_v1(in0.data(), in1.data(), out.data());
    func(in0.data(), in1.data(), out_ref.data());

    double diff = max_abs_diff(out.data(), out_ref.data(), out.size());
    std::cout << "Max absolute difference: " << diff << std::endl;

    REQUIRE(diff < 1e-5);
}





TEST_CASE( "test01", "[test]" ) {
    test_op_func(&abcd_ebf_aefcd_v2);
}

TEST_CASE( "test02", "[test]" ) {
    test_op_func(&abcd_ebf_aefcd_v2_parallel);
}

TEST_CASE( "test03", "[test]" ) {
    test_op_func(&abcd_ebf_aefcd_v2_reordered);
}

TEST_CASE( "test04", "[test]" ) {
    test_op_func(&abcd_ebf_aefcd_v4);
}

TEST_CASE( "test05", "[test]" ) {
    test_op_func(&abcd_ebf_aefcd_v1_lowered);
}



int main(int argc, char* argv[]) {
    generate_kernels();
    return Catch::Session().run(argc, argv);
}