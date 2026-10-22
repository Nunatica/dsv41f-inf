#pragma once
#include "vecdotq.cuh"
#include <cstring>
// On-disk PTQ1_0 stays compact in RAM and the expert cache. Only the prompt
// staging ring expands it losslessly to the MMQ-supported Q2_0 representation.
static __device__ __forceinline__ int dsv4f_ptq_digit(const uint8_t * b,int j) {
    int at=j<80?j%16:j<120?16+(j-80)%8:24+(j-120)%2;
    int n=j<80?j/16:j<120?(j-80)/8:(j-120)/2;
    unsigned v=b[at];for(int i=0;i<n;++i)v=(v*3u)&255u;return (v*3u)>>8;
}
static __global__ void dsv4f_ptq_to_q2(const uint8_t * src,block_q2_0 * dst,int64_t blocks) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=blocks*32)return;
    int64_t b=i/32;int h=(i%32)/16,k=i%16;unsigned v=0;
    for(int j=0;j<4;++j)v|=dsv4f_ptq_digit(src+b*28,h*64+k*4+j)<<(2*j);
    dst[2*b+h].qs[k]=v;if(!k)dst[2*b+h].d=*(const half*)(src+b*28+26);
}
template<int TYPE> static __device__ __forceinline__ float dsv4f_row_dot(const char * row,const block_q8_1 * q,int n,int lane) {
    float s=0;
    if constexpr(TYPE==GGML_TYPE_PTQ1_0){for(int j=lane;j<n;j+=32){const auto * b=(const uint8_t*)row+j/128*28;s+=(dsv4f_ptq_digit(b,j%128)-1)*__half2float(*(const half*)(b+26))*__low2float(q[j/32].ds)*q[j/32].qs[j%32];}}
    if constexpr(TYPE==GGML_TYPE_Q2_0){constexpr int qi=QI2_0,vdr=VDR_Q2_0_Q8_1_MMVQ;for(int k=lane/(qi/vdr);k<n/QK2_0;k+=vdr*32/qi)s+=vec_dot_q2_0_q8_1(row,q+k*(QK2_0/32),k,vdr*(lane%(qi/vdr)));}
    if constexpr(TYPE==GGML_TYPE_Q3_K){constexpr int qi=QI3_K,vdr=VDR_Q3_K_Q8_1_MMVQ;for(int k=lane/(qi/vdr);k<n/QK_K;k+=vdr*32/qi)s+=vec_dot_q3_K_q8_1(row,q+k*(QK_K/32),k,vdr*(lane%(qi/vdr)));}
    return warp_reduce_sum<32>(s);
}
// Four adjacent ternary weights share one Q8 activation scale. Accumulate their
// exact integer dot before applying the two scales, instead of decoding and
// multiplying every scalar separately. The original path remains available for
// numerical/performance comparisons; the changed FP32 reduction is tested
// independently against both that path and a host double-precision oracle.
static __device__ __forceinline__ float dsv4f_ptq_dot4(const char * row,const block_q8_1 * q,int n,int lane) {
    const int j=4*lane;
    float s=0;
    for(int b=0;b<n/128;++b) {
        const auto * w=(const uint8_t*)row+b*28;
        unsigned packed=0;
#pragma unroll
        for(int k=0;k<4;++k) {
            const int t=j+k;
            const int at=t<80?t%16:t<120?16+(t-80)%8:24+(t-120)%2;
            const int digit=t<80?t/16:t<120?(t-80)/8:(t-120)/2;
            const unsigned power=digit==0?1:digit==1?3:digit==2?9:digit==3?27:81;
            const int v=int((((unsigned)w[at]*power)&255u)*3u>>8)-1;
            packed|=(unsigned(v)&255u)<<(8*k);
        }
        const auto & a=q[b*4+lane/8];
        const int x=*reinterpret_cast<const int*>(a.qs+4*(lane%8));
        const int dot=ggml_cuda_dp4a(int(packed),x,0);
        s+=float(dot)*(__half2float(*(const half*)(w+26))*__low2float(a.ds));
    }
    return warp_reduce_sum<32>(s);
}
template<int TYPE,bool PTQ_DOT4=false> static __global__ void dsv4f_ptr_mmv(char * const * cache,size_t offset,const int32_t * plan,const block_q8_1 * q,float * out,int ni,int no,int aq_stride,size_t row_bytes,bool act_rows,bool fused,float limit) {
    static_assert(!PTQ_DOT4 || TYPE==GGML_TYPE_PTQ1_0,"PTQ dot4 requires PTQ1_0 weights");
    const int e=blockIdx.y,slot=plan[e];if(slot<0)return;int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=no)return;
    const auto * aq=q+(act_rows?e*aq_stride:0);const char * base=cache[slot];
    float v=PTQ_DOT4?dsv4f_ptq_dot4(base+offset+row*row_bytes,aq,ni,lane):dsv4f_row_dot<TYPE>(base+offset+row*row_bytes,aq,ni,lane);
    if(fused){float g=PTQ_DOT4?dsv4f_ptq_dot4(base+row*row_bytes,aq,ni,lane):dsv4f_row_dot<TYPE>(base+row*row_bytes,aq,ni,lane);g=fminf(g,limit);v=g/(1.f+expf(-g))*fminf(fmaxf(v,-limit),limit);}
    if(!lane)out[e*no+row]=v;
}
static void dsv4f_ptr_product(ggml_type type,char * const * cache,size_t offset,const int32_t * plan,const void * q,float * out,int ni,int no,int nk,bool act_rows,bool fused,float limit,cudaStream_t stream) {
    dim3 grid((no+3)/4,nk);int stride=ni/32;size_t rb=ggml_row_size(type,ni);
    switch(type){
#define DSV4F_PTR(T) case T:dsv4f_ptr_mmv<T><<<grid,128,0,stream>>>(cache,offset,plan,(const block_q8_1*)q,out,ni,no,stride,rb,act_rows,fused,limit);break;
    DSV4F_PTR(GGML_TYPE_Q3_K) DSV4F_PTR(GGML_TYPE_Q2_0)
    case GGML_TYPE_PTQ1_0: {
        static const bool dot4=[] {const char * v=std::getenv("DSV4F_V41_PTQ_DOT");return !v || std::strcmp(v,"0");}();
        if(dot4)dsv4f_ptr_mmv<GGML_TYPE_PTQ1_0,true><<<grid,128,0,stream>>>(cache,offset,plan,(const block_q8_1*)q,out,ni,no,stride,rb,act_rows,fused,limit);
        else dsv4f_ptr_mmv<GGML_TYPE_PTQ1_0><<<grid,128,0,stream>>>(cache,offset,plan,(const block_q8_1*)q,out,ni,no,stride,rb,act_rows,fused,limit);
        break;
    }
#undef DSV4F_PTR
    default:GGML_ABORT("unsupported DSV4F pointer expert type");}
    CUDA_CHECK(cudaGetLastError());
}
