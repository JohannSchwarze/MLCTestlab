#ifndef OPTS_H
#define OPTS_H


extern uint64_t const in0_size;
extern uint64_t const in1_size;
extern uint64_t const out_size;
extern uint64_t const flops;

void abcd_ebf_aefcd_v1(float const* in0, float const* in1, float* out);
void abcd_ebf_aefcd_v2(float const* in0, float const* in1, float* out);
void abcd_ebf_aefcd_v2_parallel(float const* in0, float const* in1, float* out);
void abcd_ebf_aefcd_v2_reordered(float const* in0, float const* in1, float* out);
void abcd_ebf_aefcd_v4(float const* in0, float const* in1, float* out);





#endif 