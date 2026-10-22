#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Compare the actual backend dispatch and graph fusion against the preceding
// eight-warp matvec + separate clamp/GLU graph. Include channel/sample strides
// and multi-token inputs, which must continue to take their old dispatch.
static bool check(ggml_backend_t gpu,int ni,int no,int channels,int samples,int nt,bool shared) {
    struct state {ggml_context * ctx;ggml_backend_buffer_t buf;ggml_cgraph * graph;
                  ggml_tensor * w,*g,*x,*out;};
    std::vector<state> modes(shared?4:2);
    for(size_t mode=0;mode<modes.size();++mode) {
        auto & s=modes[mode];s.ctx=ggml_init({4<<20,nullptr,true});
        s.w=ggml_new_tensor_4d(s.ctx,GGML_TYPE_Q8_0,ni,no,channels,1);
        s.g=shared?ggml_dup_tensor(s.ctx,s.w):nullptr;
        s.x=ggml_new_tensor_4d(s.ctx,GGML_TYPE_F32,ni,nt,channels,samples);
        auto * up=ggml_mul_mat(s.ctx,s.w,s.x);
        if(shared) {
            auto * gate=ggml_mul_mat(s.ctx,s.g,s.x);
            s.out=mode>=2?ggml_swiglu_clamp(s.ctx,gate,up,1.25f):
                ggml_swiglu_split(s.ctx,ggml_clamp(s.ctx,gate,-INFINITY,1.25f),ggml_clamp(s.ctx,up,-1.25f,1.25f));
        } else s.out=up;
        s.graph=ggml_new_graph_custom(s.ctx,1024,false);ggml_build_forward_expand(s.graph,s.out);
        ggml_set_input(s.x);ggml_set_output(s.out);
        s.buf=ggml_backend_alloc_ctx_tensors(s.ctx,gpu);if(!s.buf)return false;
    }
    std::mt19937 rng(85431);std::normal_distribution<float> normal(0,.03f);
    for(int gate=0;gate<(shared?2:1);++gate) {
        auto * w=gate?modes[0].g:modes[0].w;
        std::vector<float> values(ggml_nelements(w));for(auto & v:values)v=normal(rng);
        std::vector<uint8_t> bytes(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),bytes.data(),0,ggml_nrows(w),ni,nullptr);
        for(auto & s:modes)ggml_backend_tensor_set(gate?s.g:s.w,bytes.data(),0,bytes.size());
    }
    std::vector<float> input(ggml_nelements(modes[0].x)),expected(ggml_nelements(modes[0].out)),actual(expected.size());
    bool ok=true;double worst=0;
    for(float scale:{.01f,1.f,20.f}) {
        for(auto & v:input)v=normal(rng)*scale;
        for(size_t mode=0;mode<modes.size();++mode) {
            auto & s=modes[mode];setenv("GGML_HIP_Q8_TILE",mode%2?"1":"0",1);
            ggml_backend_tensor_set(s.x,input.data(),0,input.size()*4);
            for(int i=0;i<3;++i)ok&=ggml_backend_graph_compute(gpu,s.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(s.out,actual.data(),0,actual.size()*4);
            if(!mode)expected=actual;
            double e=0,n=0;for(size_t i=0;i<actual.size();++i){const double d=actual[i]-expected[i];e+=d*d;n+=double(expected[i])*expected[i];ok&=std::isfinite(actual[i]);}
            const double rel=std::sqrt(e/std::max(n,1e-30));worst=std::max(worst,rel);ok&=rel<2e-6;
        }
    }
    for(size_t mode=0;mode<modes.size();++mode) {
        auto & s=modes[mode];setenv("GGML_HIP_Q8_TILE",mode%2?"1":"0",1);
        std::vector<double> times;
        for(int i=-3;i<20;++i) {
            auto start=std::chrono::steady_clock::now();
            ok&=ggml_backend_graph_compute(gpu,s.graph)==GGML_STATUS_SUCCESS;ggml_backend_synchronize(gpu);
            if(i>=0)times.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(times.begin(),times.end());
        std::printf("Q8 graph K=%d M=%d channels=%d samples=%d tokens=%d shared=%d mode=%zu median_us=%.3f worst_relative=%.9g %s\n",
            ni,no,channels,samples,nt,int(shared),mode,times[10],worst,ok?"PASS":"FAIL");
    }
    for(auto & s:modes){ggml_backend_buffer_free(s.buf);ggml_free(s.ctx);}
    return ok;
}
int main() {
    setenv("GGML_CUDA_MOE_CACHE","0",1);ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 77;
    bool ok=check(gpu,5120,2304,1,1,1,true);
    ok&=check(gpu,4096,1024,8,2,1,false);
    ok&=check(gpu,1280,32768,1,1,1,false);
    ok&=check(gpu,5120,1280,1,1,2,false);
    ok&=check(gpu,5120,512,1,1,4,false);
    ggml_backend_free(gpu);return ok?0:1;
}
