#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static const ggml_type types[] = {GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
                                  GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};

static void weights(ggml_type type, int n, int rows, std::mt19937 & rng, std::vector<unsigned char> & packed) {
    packed.resize(ggml_row_size(type,n)*rows);
    if(type==GGML_TYPE_PTQ1_0 || type==GGML_TYPE_Q2_0) {
        for(auto & byte:packed)byte=rng()%256;
        const int block=ggml_blck_size(type),bytes=ggml_type_size(type);
        for(size_t b=0;b<packed.size()/bytes;++b){auto d=ggml_fp32_to_fp16(.003f+.01f*(b%7));std::memcpy(packed.data()+b*bytes+(type==GGML_TYPE_PTQ1_0?26:0),&d,2);}
        (void) block;
    } else {
        std::normal_distribution<float> normal(0.f,.3f);std::vector<float> values(size_t(n)*rows);
        for(float & v:values)v=normal(rng);
        const size_t written=ggml_quantize_chunk(type,values.data(),packed.data(),0,rows,n,nullptr);
        if(written!=packed.size())std::abort();
    }
}

// Exercise every packed Q3 scale/sign bit, including the +/-127 activation
// endpoints and nonzero group sums used by the unsigned-dot correction.
static bool q3_extremes(std::mt19937 & rng) {
    const auto * dot = ggml_get_type_traits_cpu(GGML_TYPE_Q3_K);
    bool ok = true;
    for (int n : {256, 512, 2304, 5120}) {
        std::vector<block_q3_K> w(n / QK_K);
        std::vector<block_q8_K> q(n / QK_K);
        std::vector<float> a(n), b(n);
        for (int trial = 0; trial < 64; ++trial) {
            for (size_t k = 0; k < w.size(); ++k) {
                auto * raw = reinterpret_cast<unsigned char *>(&w[k]);
                for (size_t j = 0; j < sizeof(w[k]); ++j) raw[j] = rng() % 256;
                if (trial < 4) {
                    std::memset(w[k].scales, trial & 1 ? 255 : 0, sizeof(w[k].scales));
                    std::memset(w[k].hmask, trial & 2 ? 255 : 0, sizeof(w[k].hmask));
                    std::memset(w[k].qs, trial & 1 ? 255 : 0, sizeof(w[k].qs));
                }
                w[k].d = ggml_fp32_to_fp16(n == 256 ? 1.f : (trial & 1 ? -.02f : .03f));
                q[k].d = n == 256 ? 1.f : (trial & 2 ? -.01f : .04f);
                for (int j = 0; j < QK_K; ++j) {
                    q[k].qs[j] = trial < 4 ? (trial & 1 ? -127 : 127) :
                                  trial == 4 ? 0 : trial == 5 ? (j & 1 ? -127 : 127) : int(rng() % 255) - 127;
                }
                for (int j = 0; j < QK_K / 16; ++j) {
                    int total = 0;
                    for (int t = 0; t < 16; ++t) total += q[k].qs[16*j + t];
                    q[k].bsums[j] = total;
                }
            }
            dequantize_row_q3_K(w.data(), a.data(), n);
            dequantize_row_q8_K(q.data(), b.data(), n);
            double want = 0, l1 = 0;
            for (int j = 0; j < n; ++j) {
                want += double(a[j]) * b[j];
                l1 += std::abs(double(a[j]) * b[j]);
            }
            float got = 0;
            dot->vec_dot(n, &got, 0, w.data(), 0, q.data(), 0, 1);
            if (!std::isfinite(got) || std::abs(got - want) > 2e-6 * std::max(1., l1) ||
                (n == 256 && got != want)) {
                std::fprintf(stderr, "q3_K extremes n=%d trial=%d got=%g expected=%g\n", n, trial, got, want);
                ok = false;
            }
        }
    }
    return ok;
}

// Finite half scales include signed zeros, subnormals and +/-65504.
// The hash permits bit-for-bit baseline/candidate comparison in each ISA.
static bool ptq_scales() {
    const auto * dot = ggml_get_type_traits_cpu(GGML_TYPE_PTQ1_0);
    unsigned char weights[28];
    for (int j = 0; j < 26; ++j) weights[j] = (j * 73 + 19) & 255;
    const auto one = ggml_fp32_to_fp16(1.f);
    std::memcpy(weights + 26, &one, 2);
    float trits[128];
    ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float(weights, trits, 128);
    block_q8_0 act[4];
    int dots[4] = {}, l1 = 0;
    for (int j = 0; j < 128; ++j) {
        act[j/32].qs[j%32] = (j * 37 % 255) - 127;
        dots[j/32] += int(trits[j]) * act[j/32].qs[j%32];
        l1 += std::abs(int(trits[j]) * act[j/32].qs[j%32]);
    }
    uint64_t hash = 14695981039346656037ULL;
    for (int mode = 0; mode < 2; ++mode) for (unsigned raw = 0; raw < 65536; ++raw) {
        if ((raw & 0x7c00) == 0x7c00) continue;
        const ggml_fp16_t half = raw;
        const auto wscale = mode ? one : half;
        std::memcpy(weights + 26, &wscale, 2);
        for (auto & q : act) q.d = mode ? half : one;
        float got;
        dot->vec_dot(128, &got, 0, weights, 0, act, 0, 1);
        const double scale = ggml_fp16_to_fp32(half);
        const double want = scale * (dots[0] + dots[1] + dots[2] + dots[3]);
        if (!std::isfinite(got) || std::abs(got - want) > 2e-6 * std::max(1., std::abs(scale) * l1)) {
            std::fprintf(stderr, "PTQ half-scale mode=%d half=%04x got=%g expected=%g\n",mode,raw,got,want);
            return false;
        }
        uint32_t bits; std::memcpy(&bits, &got, 4);
        hash = (hash ^ bits) * 1099511628211ULL;
    }
    // Exercise all 256 packed-byte values across every digit/tail position.
    std::memcpy(weights + 26, &one, 2);
    for (auto & q : act) q.d = one;
    for (int raw = 0; raw < 256; ++raw) {
        std::memset(weights, raw, 26);
        ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float(weights, trits, 128);
        int want = 0;
        for (int j = 0; j < 128; ++j) want += int(trits[j]) * act[j/32].qs[j%32];
        float got;
        dot->vec_dot(128, &got, 0, weights, 0, act, 0, 1);
        if (got != want) return false;
        uint32_t bits; std::memcpy(&bits, &got, 4);
        hash = (hash ^ bits) * 1099511628211ULL;
    }
    std::printf("[dsv4f-ptq-test] 63488 finite half scales x2 and all packed bytes PASS hash=%016llx\n", (unsigned long long)hash);
    return true;
}

int main(int argc,char **) {
    const char * isa=std::getenv("DSV4F_CPU_PTQ_ISA");
    const char * quant=std::getenv("DSV4F_CPU_QUANT_ISA");
    if(((isa&&!std::strcmp(isa,"avx512")) || (quant&&!std::strcmp(quant,"avx512"))) &&
       (!ggml_cpu_has_avx512() || !ggml_cpu_has_avx512_vnni()))return 77;
    const char * ptq_kernel=std::getenv("DSV4F_CPU_PTQ_KERNEL");
    if(ptq_kernel && !std::strcmp(ptq_kernel,"f16c") && (!ggml_cpu_has_avx2() || !ggml_cpu_has_f16c()))return 77;
    if(ptq_kernel && !std::strcmp(ptq_kernel,"threshold") && !ggml_cpu_has_avx512_vbmi())return 77;
    ggml_cpu_init();auto * ctx=ggml_init({1<<20,nullptr,true});
    std::mt19937 rng(4101);std::normal_distribution<float> normal(0.f,.3f);bool ok=true;
    for(auto type:types) {
        const auto * packed=ggml_get_type_traits(type);const auto * dot=ggml_get_type_traits_cpu(type);
        for(int n:{32,64,96,128,256,512,2304,5120}) {
            if(n%ggml_blck_size(type))continue;
            std::vector<unsigned char>w,q(ggml_row_size(dot->vec_dot_type,n));std::vector<float>x(n),a(n),b(n);
            for(int trial=0;trial<24;++trial){
                weights(type,n,1,rng,w);
                for(int j=0;j<n;++j)x[j]=trial==0?0.f:trial==1?(j%2?-1.f:1.f):normal(rng);
                std::fill(q.begin(), q.end(), 0xa5);
                ggml_get_type_traits_cpu(dot->vec_dot_type)->from_float(x.data(),q.data(),n);
                if (trial == 0 && dot->vec_dot_type == GGML_TYPE_Q8_K) {
                    const auto * blocks = reinterpret_cast<const block_q8_K *>(q.data());
                    for (int k = 0; k < n / QK_K; ++k) for (int j = 0; j < QK_K / 16; ++j) {
                        if (blocks[k].bsums[j] != 0) {
                            std::fprintf(stderr, "Q8_K zero activation left stale sums\n");
                            ok = false;
                        }
                    }
                }
                packed->to_float(w.data(),a.data(),n);if(dot->vec_dot_type==GGML_TYPE_Q8_K)dequantize_row_q8_K((const block_q8_K*)q.data(),b.data(),n);else ggml_get_type_traits(dot->vec_dot_type)->to_float(q.data(),b.data(),n);
                double want=0,l1=0;for(int j=0;j<n;++j){want+=double(a[j])*b[j];l1+=std::abs(double(a[j])*b[j]);}
                float got=0;dot->vec_dot(n,&got,0,w.data(),0,q.data(),0,1);
                if(!std::isfinite(got)||std::abs(got-want)>2e-6*std::max(1.,l1)){
                    std::fprintf(stderr,"%s n=%d trial=%d got=%g expected=%g l1=%g\n",ggml_type_name(type),n,trial,got,want,l1);ok=false;
                }
            }
        }
    }
    ok = q3_extremes(rng) && ptq_scales() && ok;
    if(ok&&argc>1)for(auto type:types)for(int n:{5120,2304}){
        const int rows=n==5120?2304:5120;const auto * dot=ggml_get_type_traits_cpu(type);
        std::vector<unsigned char>w,q(ggml_row_size(dot->vec_dot_type,n));weights(type,n,rows,rng,w);
        std::vector<float>x(n),out(rows);for(float & v:x)v=normal(rng);
        ggml_get_type_traits_cpu(dot->vec_dot_type)->from_float(x.data(),q.data(),n);
        const size_t stride=ggml_row_size(type,n);std::vector<double> times;
        for(int rep=0;rep<31;++rep){auto start=std::chrono::steady_clock::now();for(int r=0;r<rows;++r)dot->vec_dot(n,&out[r],0,w.data()+r*stride,0,q.data(),0,1);auto end=std::chrono::steady_clock::now();if(rep)times.push_back(std::chrono::duration<double,std::milli>(end-start).count());}
        std::sort(times.begin(),times.end());std::printf("[dsv4f-quant-bench] type=%s rows=%d columns=%d median_ms=%.6f min_ms=%.6f checksum=%g\n",ggml_type_name(type),rows,n,(times[14]+times[15])/2,times.front(),out[0]);
    }
    ggml_free(ctx);std::puts(ok?"[dsv4f-test] eight CPU quant dots vs float dequantization PASS":"[dsv4f-test] CPU quant dots FAIL");return ok?0:1;
}
