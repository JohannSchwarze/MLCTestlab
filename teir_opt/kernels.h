#ifndef KERNELS_H
#define KERNELS_H

#include "libxsmm.h"

struct zerofunction {
private:
    libxsmm_meltwfunction_unary m_function;

public:
    zerofunction();
    zerofunction(uint64_t m, uint64_t n, uint64_t lda);

    void operator()(float* out);
};



void zero_4_256_96(float* out);
void zero_32_64_96(float* out);
void zero_96_256_96(float* out);

void gemm_4_256_64_96_256_96(float const* in0, float const* in1, float* out);
void gemm_32_64_64_96_256_96(float const* in0, float const* in1, float* out);
void gemm_96_256_64_96_256_96(float const* in0, float const* in1, float* out);


void generate_kernels();

#endif