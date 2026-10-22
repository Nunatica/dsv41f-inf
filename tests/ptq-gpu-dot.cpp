#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/dsv4f-v41.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Exercise the actual pointer-cache kernels, including gate/up fusion, distinct
// activation rows, inactive cache slots and non-multiple-of-four output tails.
static bool check(ggml_backend_t gpu,int ni,int no,bool fused,bool act_rows) {
    constexpr int nk=3;
    auto & ctx=*static_cast<ggml_backend_cuda_context*>(gpu->context);
    const auto stream=ctx.stream();
    const size_t rb=ni/128*28, one=rb*no;
    std::mt19937 rng(72401+ni+no);
    const float scales[]={0.f,.03f,-.125f,.0001f,.75f};
    std::vector<uint8_t> w(nk*2*one);
    for(size_t b=0;b<w.size()/28;++b) {
        for(int j=0;j<26;++j)w[b*28+j]=rng()%256;
        const auto h=ggml_fp32_to_fp16(scales[b%5]);
        std::memcpy(w.data()+b*28+26,&h,2);
    }
    std::vector<block_q8_1> q(nk*ni/32);
    for(size_t b=0;b<q.size();++b) {
        // Include zero, non-power-of-two and extreme signed activation values.
        ggml_fp16_t halves[]={ggml_fp32_to_fp16(scales[(b+1)%5]),0};
        std::memcpy(&q[b].ds,halves,sizeof(halves));
        for(int k=0;k<32;++k)q[b].qs[k]=int(rng()%255)-127;
    }
    char * dw=nullptr;char ** ptrs=nullptr;block_q8_1 * dq=nullptr;
    int32_t * plan=nullptr;float * output=nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dw,w.size()));
    CUDA_CHECK(cudaMalloc((void**)&ptrs,nk*sizeof(char*)));
    CUDA_CHECK(cudaMalloc((void**)&dq,q.size()*sizeof(block_q8_1)));
    CUDA_CHECK(cudaMalloc((void**)&plan,nk*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc((void**)&output,nk*no*sizeof(float)));
    std::vector<char*> pointers;for(int e=0;e<nk;++e)pointers.push_back(dw+2*e*one);
    const int32_t selected[]={2,-1,0};
    CUDA_CHECK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ptrs,pointers.data(),nk*sizeof(char*),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq,q.data(),q.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(plan,selected,sizeof(selected),cudaMemcpyHostToDevice));
    const auto launch=[&](bool dot4) {
        const dim3 grid((no+3)/4,nk);
        if(dot4)dsv4f_ptr_mmv<GGML_TYPE_PTQ1_0,true><<<grid,128,0,stream>>>(ptrs,one,plan,dq,output,ni,no,ni/32,rb,act_rows,fused,1.25f);
        else dsv4f_ptr_mmv<GGML_TYPE_PTQ1_0><<<grid,128,0,stream>>>(ptrs,one,plan,dq,output,ni,no,ni/32,rb,act_rows,fused,1.25f);
        CUDA_CHECK(cudaGetLastError());
    };
    std::vector<float> reference(nk*no,-1234.5f),actual=reference;
    CUDA_CHECK(cudaMemcpy(output,reference.data(),reference.size()*4,cudaMemcpyHostToDevice));
    launch(false);CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(reference.data(),output,reference.size()*4,cudaMemcpyDeviceToHost));
    launch(true);CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
    bool ok=true;double error=0,norm=0,old_error=0,old_norm=0,maximum=0;
    for(int e=0;e<nk;++e)for(int r=0;r<no;++r) {
        const int index=e*no+r;
        if(selected[e]<0){ok&=actual[index]==-1234.5f;continue;}
        const auto dot=[&](size_t offset) {
            double sum=0;const auto * row=w.data()+selected[e]*2*one+offset+r*rb;
            for(int k=0;k<ni;++k) {
                const auto * b=row+k/128*28;const int j=k%128;
                const int at=j<80?j%16:j<120?16+(j-80)%8:24+(j-120)%2;
                const int n=j<80?j/16:j<120?(j-80)/8:(j-120)/2;
                unsigned byte=b[at];for(int d=0;d<n;++d)byte=(byte*3)&255;
                const int digit=int(byte*3>>8)-1;
                ggml_fp16_t h,aq;std::memcpy(&h,b+26,2);
                const auto & a=q[(act_rows?e*ni/32:0)+k/32];std::memcpy(&aq,&a.ds,2);
                sum+=double(digit)*ggml_fp16_to_fp32(h)*ggml_fp16_to_fp32(aq)*int(a.qs[k%32]);
            }
            return sum;
        };
        double expected=dot(one);
        if(fused) {const double g=std::min(dot(0),1.25);expected=g/(1+std::exp(-g))*std::clamp(expected,-1.25,1.25);}
        const double delta=actual[index]-expected,old_delta=actual[index]-reference[index];
        error+=delta*delta;norm+=expected*expected;
        old_error+=old_delta*old_delta;old_norm+=double(reference[index])*reference[index];
        maximum=std::max(maximum,std::abs(delta));ok&=std::isfinite(actual[index]);
    }
    const double rel=std::sqrt(error/std::max(norm,1e-30));
    const double old_rel=std::sqrt(old_error/std::max(old_norm,1e-30));
    ok&=rel<5e-6 && old_rel<1e-5 && maximum<.003;
    cudaGraph_t graphs[2];cudaGraphExec_t executables[2];
    for(int mode=0;mode<2;++mode) {
        CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeRelaxed));
        for(int i=0;i<32;++i)launch(mode);
        CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode]));
        CUDA_CHECK(cudaGraphInstantiate(&executables[mode],graphs[mode],nullptr,nullptr,0));
    }
    cudaEvent_t begin,end;CUDA_CHECK(hipEventCreate(&begin));CUDA_CHECK(hipEventCreate(&end));
    std::vector<double> times[2];
    for(int i=-4;i<20;++i)for(int k=0;k<2;++k) {
        const int mode=(i+k)&1;
        CUDA_CHECK(cudaEventRecord(begin,stream));CUDA_CHECK(cudaGraphLaunch(executables[mode],stream));
        CUDA_CHECK(cudaEventRecord(end,stream));CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;CUDA_CHECK(hipEventElapsedTime(&ms,begin,end));
        if(i>=0)times[mode].push_back(ms/32);
    }
    for(auto & t:times)std::sort(t.begin(),t.end());
    std::printf("PTQ GPU K=%d M=%d fused=%d act_rows=%d oracle_rmse=%.9g old_rmse=%.9g max_abs=%.9g old_ms=%.6f dot4_ms=%.6f speedup=%.3f %s\n",
        ni,no,int(fused),int(act_rows),rel,old_rel,maximum,times[0][10],times[1][10],times[0][10]/times[1][10],ok?"PASS":"FAIL");
    for(int k=0;k<2;++k){CUDA_CHECK(cudaGraphExecDestroy(executables[k]));CUDA_CHECK(cudaGraphDestroy(graphs[k]));}
    CUDA_CHECK(cudaEventDestroy(begin));CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaFree(output));CUDA_CHECK(cudaFree(plan));CUDA_CHECK(cudaFree(dq));CUDA_CHECK(cudaFree(ptrs));CUDA_CHECK(cudaFree(dw));
    return ok;
}
int main() {
    ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;
    bool ok=check(gpu,5120,2304,true,false);
    ok&=check(gpu,2304,5120,false,true);
    ok&=check(gpu,128,7,false,false);
    ggml_backend_free(gpu);return ok?0:1;
}
