#include "kernels.h"

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




transposefunction::transposefunction() {
    m_function = nullptr;
}

transposefunction::transposefunction(uint64_t m, uint64_t n, uint64_t ldi, uint64_t ldo) {
    libxsmm_meltw_unary_shape shape = libxsmm_create_meltw_unary_shape(m, n, ldi, ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    m_function = libxsmm_dispatch_meltw_unary(libxsmm_meltw_unary_type::LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, shape, 0);
}

void transposefunction::operator()(float const* in0, float* out) {
    libxsmm_meltw_unary_param params;
    params.in.primary = (void*)in0;
    params.out.primary = out;
    LIBXSMM_ASSERT(NULL != m_function);
    m_function(&params);
}













static transposefunction _transpose_m48_n64_ldi294912_ldo4096;
void transpose_m48_n64_ldi294912_ldo4096(const float* in0, float* out) {
    _transpose_m48_n64_ldi294912_ldo4096(in0, out);
}

static zerofunction _zero_m64_n64_ldo4096;
void zero_m64_n64_ldo4096(float* out) {
    _zero_m64_n64_ldo4096(out);
}

static zerofunction _zero_m48_n64_ldo48;
void zero_m48_n64_ldo48(float* out) {
    _zero_m48_n64_ldo48(out);
}

static libxsmm_mmfunction<float> _gemm_m64_n64_k4608_lda4096_ldb4096_ldc4096;
void gemm_m64_n64_k4608_lda4096_ldb4096_ldc4096(const float* in0, const float* in1, float* out) {
    _gemm_m64_n64_k4608_lda4096_ldb4096_ldc4096(in0, in1, out);
}

static libxsmm_mmfunction<float> _gemm_m48_n64_k4096_lda4608_ldb4096_ldc48;
void gemm_m48_n64_k4096_lda4608_ldb4096_ldc48(const float* in0, const float* in1, float* out) {
    _gemm_m48_n64_k4096_lda4608_ldb4096_ldc48(in0, in1, out);
}












void generate_kernels() {
    _transpose_m48_n64_ldi294912_ldo4096 = transposefunction(48, 64, 294912, 4096);

    _zero_m48_n64_ldo48 = zerofunction(48, 64, 48);
    _zero_m64_n64_ldo4096 = zerofunction(64, 64, 4096);

    _gemm_m48_n64_k4096_lda4608_ldb4096_ldc48 = libxsmm_mmfunction<float>(LIBXSMM_GEMM_FLAG_TRANS_B, 48, 64, 4096, 4608, 4096, 48, 1.0, 1.0);
    _gemm_m64_n64_k4608_lda4096_ldb4096_ldc4096 = libxsmm_mmfunction<float>(LIBXSMM_GEMM_FLAG_TRANS_B, 64, 64, 4608, 4096, 4096, 4096, 1.0, 1.0);
}