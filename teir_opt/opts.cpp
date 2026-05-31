#include "kernels.h"
#include "opts.h"
#include "libxsmm.h"

uint64_t const in0_size = 8 * 64 * 24 * 4;
uint64_t const in1_size = 64 * 12 * 256;
uint64_t const out_size = 8 * 24 * 4 * 12 * 256;
uint64_t const flops = 8 * 64 * 24 * 4 * 12 * 256 * 2;







void abcd_ebf_aefcd_v1(float const* in0, float const* in1, float* out) {
    uint64_t const da = 8;
    uint64_t const db = 64;
    uint64_t const dc = 24;
    uint64_t const dd = 4;
    uint64_t const de = 12;
    uint64_t const df = 256;

    uint64_t const in0_sd = 1;
    uint64_t const in0_sc = in0_sd * dd;
    uint64_t const in0_sb = in0_sc * dc;
    uint64_t const in0_sa = in0_sb * db;
    uint64_t const in0_se = 0;
    uint64_t const in0_sf = 0;

    uint64_t const in1_sf = 1;
    uint64_t const in1_sb = in1_sf * df;
    uint64_t const in1_se = in1_sb * db;
    uint64_t const in1_sd = 0;
    uint64_t const in1_sc = 0;
    uint64_t const in1_sa = 0;

    uint64_t const out_sd = 1;
    uint64_t const out_sc = out_sd * dd;
    uint64_t const out_sf = out_sc * dc;
    uint64_t const out_se = out_sf * df;
    uint64_t const out_sa = out_se * de;
    uint64_t const out_sb = 0;

    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xb = 0; xb < db; ++xb) {
            for (uint64_t xc = 0; xc < dc; ++xc) {
                for (uint64_t xd = 0; xd < dd; ++xd) {
                    for (uint64_t xe = 0; xe < de; ++xe) {
                        for (uint64_t xf = 0; xf < df; ++xf) {
                            float const* in0_ptr = in0 + xa * in0_sa + xb * in0_sb + xc * in0_sc + xd * in0_sd + xe * in0_se + xf * in0_sf;
                            float const* in1_ptr = in1 + xa * in1_sa + xb * in1_sb + xc * in1_sc + xd * in1_sd + xe * in1_se + xf * in1_sf;
                            float* out_ptr = out + xa * out_sa + xb * out_sb + xc * out_sc + xd * out_sd + xe * out_se + xf * out_sf;

                            if (xb == 0) {
                                *out_ptr = 0.0f;
                            }
                            *out_ptr += *in0_ptr * *in1_ptr;
                        }
                    }
                }
            }
        }
    }
}





void abcd_ebf_aefcd_v2(float const* in0, float const* in1, float* out) {
    uint64_t const da = 8;
    uint64_t const db = 64;
    uint64_t const dg = 96;
    uint64_t const de = 12;
    uint64_t const df = 256;

    uint64_t const in0_sg = 1;
    uint64_t const in0_sb = in0_sg * dg;
    uint64_t const in0_sa = in0_sb * db;
    uint64_t const in0_se = 0;
    uint64_t const in0_sf = 0;

    uint64_t const in1_sf = 1;
    uint64_t const in1_sb = in1_sf * df;
    uint64_t const in1_se = in1_sb * db;
    uint64_t const in1_sa = 0;
    uint64_t const in1_sg = 0;

    uint64_t const out_sg = 1;
    uint64_t const out_sf = out_sg * dg;
    uint64_t const out_se = out_sf * df;
    uint64_t const out_sa = out_se * de;
    uint64_t const out_sb = 0;

    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xe = 0; xe < de; ++xe) {
            float const* in0_ptr = in0 + xa * in0_sa + xe * in0_se;
            float const* in1_ptr = in1 + xa * in1_sa + xe * in1_se;
            float* out_ptr       = out + xa * out_sa + xe * out_se;

            zero_96_256_96(out_ptr);
            gemm_96_256_64_96_256_96(in0_ptr, in1_ptr, out_ptr);
        }
    }
}


void abcd_ebf_aefcd_v2_parallel(float const* in0, float const* in1, float* out) {
    uint64_t const da = 8;
    uint64_t const db = 64;
    uint64_t const dg = 96;
    uint64_t const de = 12;
    uint64_t const df = 256;

    uint64_t const in0_sg = 1;
    uint64_t const in0_sb = in0_sg * dg;
    uint64_t const in0_sa = in0_sb * db;
    uint64_t const in0_se = 0;
    uint64_t const in0_sf = 0;

    uint64_t const in1_sf = 1;
    uint64_t const in1_sb = in1_sf * df;
    uint64_t const in1_se = in1_sb * db;
    uint64_t const in1_sa = 0;
    uint64_t const in1_sg = 0;

    uint64_t const out_sg = 1;
    uint64_t const out_sf = out_sg * dg;
    uint64_t const out_se = out_sf * df;
    uint64_t const out_sa = out_se * de;
    uint64_t const out_sb = 0;

    #pragma omp parallel for collapse(2)
    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xe = 0; xe < de; ++xe) {
            float const* in0_ptr = in0 + xa * in0_sa + xe * in0_se;
            float const* in1_ptr = in1 + xa * in1_sa + xe * in1_se;
            float* out_ptr       = out + xa * out_sa + xe * out_se;

            zero_96_256_96(out_ptr);
            gemm_96_256_64_96_256_96(in0_ptr, in1_ptr, out_ptr);
        }
    }
}


void abcd_ebf_aefcd_v2_reordered(float const* in0, float const* in1, float* out) {
    uint64_t const da = 8;
    uint64_t const db = 64;
    uint64_t const dg = 96;
    uint64_t const de = 12;
    uint64_t const df = 256;

    uint64_t const in0_sg = 1;
    uint64_t const in0_sb = in0_sg * dg;
    uint64_t const in0_sa = in0_sb * db;
    uint64_t const in0_se = 0;
    uint64_t const in0_sf = 0;

    uint64_t const in1_sf = 1;
    uint64_t const in1_sb = in1_sf * df;
    uint64_t const in1_se = in1_sb * db;
    uint64_t const in1_sa = 0;
    uint64_t const in1_sg = 0;

    uint64_t const out_sg = 1;
    uint64_t const out_sf = out_sg * dg;
    uint64_t const out_se = out_sf * df;
    uint64_t const out_sa = out_se * de;
    uint64_t const out_sb = 0;

    for (uint64_t xe = 0; xe < de; ++xe) {
        for (uint64_t xa = 0; xa < da; ++xa) {
            float const* in0_ptr = in0 + xa * in0_sa + xe * in0_se;
            float const* in1_ptr = in1 + xa * in1_sa + xe * in1_se;
            float* out_ptr       = out + xa * out_sa + xe * out_se;

            zero_96_256_96(out_ptr);
            gemm_96_256_64_96_256_96(in0_ptr, in1_ptr, out_ptr);
        }
    }
}


void abcd_ebf_aefcd_v4(float const* in0, float const* in1, float* out) {
    uint64_t const da = 8;
    uint64_t const db = 64;
    uint64_t const dj = 3;
    uint64_t const dk = 32;
    uint64_t const de = 12;
    uint64_t const dh = 4;
    uint64_t const di = 64;

    uint64_t const in0_sk = 1;
    uint64_t const in0_sj = in0_sk * dk;
    uint64_t const in0_sb = in0_sj * dj;
    uint64_t const in0_sa = in0_sb * db;
    uint64_t const in0_se = 0;
    uint64_t const in0_sh = 0;
    uint64_t const in0_si = 0;

    uint64_t const in1_si = 1;
    uint64_t const in1_sh = in1_si * di;
    uint64_t const in1_sb = in1_sh * dh;
    uint64_t const in1_se = in1_sb * db;
    uint64_t const in1_sj = 0;
    uint64_t const in1_sk = 0;
    uint64_t const in1_sa = 0;

    uint64_t const out_sk = 1;
    uint64_t const out_sj = out_sk * dk;
    uint64_t const out_si = out_sj * dj;
    uint64_t const out_sh = out_si * di;
    uint64_t const out_se = out_sh * dh;
    uint64_t const out_sa = out_se * de;
    uint64_t const out_sb = 0;

    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xe = 0; xe < de; ++xe) {
            for (uint64_t xj = 0; xj < dj; ++xj) {
                for (uint64_t xh = 0; xh < dh; ++xh) {
                    float const* in0_ptr = in0 + xa * in0_sa + xj * in0_sj + xe * in0_se + xh * in0_sh;
                    float const* in1_ptr = in1 + xa * in1_sa + xj * in1_sj + xe * in1_se + xh * in1_sh;
                    float* out_ptr       = out + xa * out_sa + xj * out_sj + xe * out_se + xh * out_sh;

                    zero_32_64_96(out_ptr);
                    gemm_32_64_64_96_256_96(in0_ptr, in1_ptr, out_ptr);
                }
            }
        }
    }
}