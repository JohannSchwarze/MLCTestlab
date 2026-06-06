#ifndef OPS_H
#define OPS_H

#include <cstdint>



extern const uint64_t da;
extern const uint64_t db;
extern const uint64_t dc;
extern const uint64_t dd;
extern const uint64_t de;
extern const uint64_t df;
extern const uint64_t dg;
extern const uint64_t dh;





void opv2_bacd_cdab(const float* in0, float* out);
void opv2_cdab_cdef_efab(const float* in0, const float* in1, float* out);
void opv2_efgh_efab_gabh(const float* in0, const float* in1, float* out);










#endif