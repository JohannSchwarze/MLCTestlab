#include "ops.h"
#include "kernels.h"




const uint64_t da = 64;
const uint64_t db = 64;
const uint64_t dc = 96;
const uint64_t dd = 48;
const uint64_t de = 64;
const uint64_t df = 64;
const uint64_t dg = 96;
const uint64_t dh = 48;






void opv1_bacd_cdef_abef(const float* in0, const float* in1, float* out) {
    uint64_t const in0_sd = 1;
    uint64_t const in0_sc = dd * in0_sd;
    uint64_t const in0_sa = dc * in0_sc;
    uint64_t const in0_sb = da * in0_sa;
    
    uint64_t const in1_sf = 1;
    uint64_t const in1_se = df * in1_sf;
    uint64_t const in1_sd = de * in1_se;
    uint64_t const in1_sc = dd * in1_sd;

    uint64_t const out_sf = 1;
    uint64_t const out_se = df * out_sf;
    uint64_t const out_sb = de * out_se;
    uint64_t const out_sa = db * out_sb;

    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xb = 0; xb < db; ++xb) {
            for (uint64_t xe = 0; xe < de; ++xe) {
                for (uint64_t xf = 0; xf < df; ++xf) {
                    float* out_ptr = out + xa * out_sa + xb * out_sb + xe * out_se + xf * out_sf;
                    *out_ptr = 0.0f;
                    for (uint64_t xc = 0; xc < dc; ++xc) {
                        for (uint64_t xd = 0; xd < dd; ++xd) {
                            float const* in0_ptr = in0 + xb * in0_sb + xa * in0_sa + xc * in0_sc + xd * in0_sd;
                            float const* in1_ptr = in1 + xc * in1_sc + xd * in1_sd + xe * in1_se + xf * in1_sf;
                            *out_ptr += *in0_ptr * *in1_ptr;
                        }
                    }
                }
            }
        }
    }
}



void opv1_abef_efgh_gabh(const float* in0, const float* in1, float* out) {
    uint64_t const in0_sf = 1;
    uint64_t const in0_se = df * in0_sf;
    uint64_t const in0_sb = de * in0_se;
    uint64_t const in0_sa = db * in0_sb;

    uint64_t const in1_sh = 1;
    uint64_t const in1_sg = dh * in1_sh;
    uint64_t const in1_sf = dg * in1_sg;
    uint64_t const in1_se = df * in1_sf;

    uint64_t const out_sh = 1;
    uint64_t const out_sb = dh * out_sh;
    uint64_t const out_sa = db * out_sb;
    uint64_t const out_sg = da * out_sa;

    for (uint64_t xg = 0; xg < dg; ++xg) {
        for (uint64_t xa = 0; xa < da; ++xa) {
            for (uint64_t xb = 0; xb < db; ++xb) {
                for (uint64_t xh = 0; xh < dh; ++xh) {
                    float* out_ptr = out + xg * out_sg + xa * out_sa + xb * out_sb + xh * out_sh;
                    *out_ptr = 0.0f;
                    for (uint64_t xe = 0; xe < de; ++xe) {
                        for (uint64_t xf = 0; xf < df; ++xf) {
                            float const* in0_ptr = in0 + xa * in0_sa + xb * in0_sb + xe * in0_se + xf * in0_sf;
                            float const* in1_ptr = in1 + xe * in1_se + xf * in1_sf + xg * in1_sg + xh * in1_sh;
                            *out_ptr += *in0_ptr * *in1_ptr;
                        }
                    }
                }
            }
        }
    }
}




























void opv2_bacd_cdab(const float* in0, float* out) {
    uint64_t const in0_sd = 1;
    uint64_t const in0_sc = dd * in0_sd;
    uint64_t const in0_sa = dc * in0_sc;
    uint64_t const in0_sb = da * in0_sa;

    uint64_t const out_sb = 1;
    uint64_t const out_sa = db * out_sb;
    uint64_t const out_sd = da * out_sa;
    uint64_t const out_sc = dd * out_sd;

    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xc = 0; xc < dc; ++xc) {
            float const* in0_ptr = in0 + xa * in0_sa + xc * in0_sc;
            float* out_ptr       = out + xa * out_sa + xc * out_sc;
            transpose_m48_n64_ldi294912_ldo4096(in0_ptr, out_ptr);
        }
    }
}

void opv2_cdab_cdef_efab(const float* in0, const float* in1, float* out) {
    uint64_t const dt = dc * dd;

    uint64_t const in0_sb = 1;
    uint64_t const in0_sa = db * in0_sb;
    uint64_t const in0_sd = da * in0_sa;
    uint64_t const in0_sc = dd * in0_sd;
    uint64_t const in0_st = da * in0_sa;

    uint64_t const in1_sf = 1;
    uint64_t const in1_se = df * in1_sf;
    uint64_t const in1_sd = de * in1_se;
    uint64_t const in1_sc = dd * in1_sd;
    uint64_t const in1_st = de * in1_se;

    uint64_t const out_sb = 1;
    uint64_t const out_sa = db * out_sb;
    uint64_t const out_sf = da * out_sa;
    uint64_t const out_se = df * out_sf;

    for (uint64_t xa = 0; xa < da; ++xa) {
        for (uint64_t xe = 0; xe < de; ++xe) {
            float const* in0_ptr = in0 + xa * in0_sa;
            float const* in1_ptr = in1 + xe * in1_se;
            float* out_ptr       = out + xa * out_sa + xe * out_se;

            zero_m64_n64_ldo4096(out_ptr);
            gemm_m64_n64_k4608_lda4096_ldb4096_ldc4096(in0_ptr, in1_ptr, out_ptr);
        }
    }
}

void opv2_efgh_efab_gabh(const float* in0, const float* in1, float* out) {
    uint64_t dr = de * df;

    uint64_t const in0_sh = 1;
    uint64_t const in0_sg = dh * in0_sh;
    uint64_t const in0_sf = dg * in0_sg;
    uint64_t const in0_se = df * in0_sf;
    uint64_t const in0_sr = dg * in0_sg;

    uint64_t const in1_sb = 1;
    uint64_t const in1_sa = db * in1_sb;
    uint64_t const in1_sf = da * in1_sa;
    uint64_t const in1_se = df * in1_sf;
    uint64_t const in1_sr = da * in1_sa;

    uint64_t const out_sh = 1;
    uint64_t const out_sb = dh * out_sh;
    uint64_t const out_sa = db * out_sb;
    uint64_t const out_sg = da * out_sa;

    for (uint64_t xg = 0; xg < dg; ++xg) {
        for (uint64_t xa = 0; xa < da; ++xa) {
            float const* in0_ptr = in0 + xg * in0_sg;
            float const* in1_ptr = in1 + xa * in1_sa;
            float* out_ptr       = out + xg * out_sg + xa * out_sa;

            zero_m48_n64_ldo48(out_ptr);
            gemm_m48_n64_k4096_lda4608_ldb4096_ldc48(in0_ptr, in1_ptr, out_ptr);
        }
    }
}