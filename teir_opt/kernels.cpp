#include <iostream>

#include "kernels.h"
#include "libxsmm.h"




zerofunction::zerofunction() {
    m_function = nullptr;
}

zerofunction::zerofunction(uint64_t m, uint64_t n, uint64_t lda) {
    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(m, n, lda, lda, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    m_function = libxsmm_dispatch_meltw_unary(libxsmm_meltw_unary_type::LIBXSMM_MELTW_TYPE_UNARY_XOR, shape, 0);
}

void zerofunction::operator()(float* out) {
    libxsmm_meltw_unary_param params;
    params.in.primary = out;
    params.out.primary = out;
    LIBXSMM_ASSERT(NULL != m_function);
    m_function(&params);
}




static zerofunction _zero_32_64_96;
void zero_32_64_96(float* out) {
    _zero_32_64_96(out);
}

static zerofunction _zero_96_256_96;
void zero_96_256_96(float* out) {
    _zero_96_256_96(out);
}

static libxsmm_mmfunction<float> _gemm_32_64_64_96_256_96;
void gemm_32_64_64_96_256_96(float const* in0, float const* in1, float* out) {
    _gemm_32_64_64_96_256_96(in0, in1, out);
}

static libxsmm_mmfunction<float> _gemm_96_256_64_96_256_96;
void gemm_96_256_64_96_256_96(float const* in0, float const* in1, float* out) {
    _gemm_96_256_64_96_256_96(in0, in1, out);
}


void generate_kernels() {
    _gemm_32_64_64_96_256_96 = libxsmm_mmfunction<float>(LIBXSMM_GEMM_FLAG_TRANS_B, 32, 64, 64, 96, 256, 96, 1.0, 1.0);
    _gemm_96_256_64_96_256_96 = libxsmm_mmfunction<float>(LIBXSMM_GEMM_FLAG_TRANS_B, 96, 256, 64, 96, 256, 96, 1.0, 1.0);
    
    _zero_32_64_96 = zerofunction(32, 64, 96);
    _zero_96_256_96 = zerofunction(96, 256, 96);
}




