#ifndef KERNELS_H
#define KERNELS_H



#include <libxsmm.h>


struct zerofunction {
private:
    libxsmm_meltwfunction_unary m_function;

public:
    zerofunction();
    zerofunction(uint64_t m, uint64_t n, uint64_t lda);

    void operator()(float* out);
};






struct transposefunction {
private:
    libxsmm_meltwfunction_unary m_function;

public:
    transposefunction();
    transposefunction(uint64_t m, uint64_t n, uint64_t ldi, uint64_t ldo);

    void operator()(float const* in0, float* out);
};






void transpose_m48_n64_ldi294912_ldo4096(const float* in0, float* out);

void zero_m64_n64_ldo4096(float* out);
void zero_m48_n64_ldo48(float* out);

void gemm_m64_n64_k4608_lda4096_ldb4096_ldc4096(const float* in0, const float* in1, float* out);
void gemm_m48_n64_k4096_lda4608_ldb4096_ldc48(const float* in0, const float* in1, float* out);





void generate_kernels();




#endif