#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace dsv4f_engram_fp8 {
inline float value(uint8_t b) {
    const int e=(b>>3)&15,mant=b&7;
    float v=e?std::ldexp(1.f+mant/8.f,e-7):std::ldexp(float(mant),-9);
    if(e==15 && mant==7)v=NAN;
    return b&128?-v:v;
}
inline void baseline(const uint8_t * w,const uint8_t * sc,float * out,int dim) {
    for(int j=0;j<dim;++j)out[j]=value(w[j])*std::ldexp(1.f,int(sc[j/32])-127);
}
struct tables {
    std::array<float,256> fp8{},scale{};
    tables() {
        for(int b=0;b<256;++b) {
            fp8[b]=value(uint8_t(b));
            // Normal powers of two are exact, independent of thread fenv.
            const uint32_t bits=uint32_t(b)<<23;
            std::memcpy(&scale[b],&bits,sizeof(bits));
        }
    }
};
inline void lut(const uint8_t * w,const uint8_t * sc,float * out,int dim) {
    static const tables lookup;
    for(int g=0;g<dim/32;++g) {
        const unsigned s=sc[g];
        // Preserve ldexpf's FTZ and rounding behavior at the two exceptional
        // scale encodings. The ordinary path performs no ldexpf calls.
        const float scale=s==0 || s==255?std::ldexp(1.f,int(s)-127):lookup.scale[s];
        for(int j=0;j<32;++j)out[32*g+j]=lookup.fp8[w[32*g+j]]*scale;
    }
}
}
