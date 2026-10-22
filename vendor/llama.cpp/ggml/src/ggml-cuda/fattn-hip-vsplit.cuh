#pragma once

#if defined(GGML_USE_HIP)
// Split the 512 V outputs across four waves instead of keeping 512 FP32
// outputs per wave. Each wave retains 128 outputs (64 accumulator VGPRs).
// The four waves also partition the QK reduction, combining FP32 partials in
// LDS. Shared K/V data stays in LDS for both GEMMs.
template<int queries, int kv_rows>
__launch_bounds__(queries*64)
static __global__ void hip_fattn_wmma_vsplit(
        const char * Q, const char * K, const char * V, const char * mask, const char * sinks,
        const int * KV_max, float * dst, float2 * dst_meta,
        float scale, float max_bias, float m0, float m1, uint32_t n_head_log2, float logit_softcap,
        int32_t ne00, uint3 ne01, int32_t ne02, int32_t ne03,
        int32_t nb01, int32_t nb02, int32_t nb03,
        int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,
        int32_t nb11, int32_t nb12, int64_t nb13,
        int32_t nb21, int32_t nb22, int64_t nb23,
        int32_t ne31, int32_t ne32, int32_t ne33,
        int32_t nb31, int32_t nb32, int64_t nb33) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    using namespace ggml_cuda_mma;
    constexpr int cols=queries*8, stride=260, nthreads=queries*64;
    const int lane=threadIdx.x, wave=threadIdx.y, tid=wave*32+lane;
    const int group=wave/4, vpart=wave%4;
    const int gqa=ne02/ne12, head_tiles=gqa/8;
    const int sequence=blockIdx.z/(head_tiles*ne12);
    const int kv_head=(blockIdx.z/head_tiles)%ne12;
    const int head0=kv_head*gqa+(blockIdx.z%head_tiles)*8;
    const int query0=blockIdx.x*queries;
    const int column=group*16+lane%16;
    extern __shared__ half2 shared[];
    half2 * qs=shared;
    half2 * kv=qs+cols*stride;
    half * biases=(half *)(kv+kv_rows*stride);
    constexpr int partial_stride=kv_rows+4;
    float * partials=(float *)(biases+queries*kv_rows);
    const half2 * kbase=(const half2 *)(K+sequence*nb13+kv_head*nb12);
    const half2 * vbase=(const half2 *)(V+sequence*nb23+kv_head*nb22);
    const half * masks=(const half *)(mask+(sequence%ne33)*nb33);
    const bool shared_kv=kbase==vbase && nb11==nb21;
    for(int i=tid;i<cols*256;i+=nthreads) {
        const int c=i/256,d=i%256,q=query0+c/8,h=head0+c%8;
        float2 value=make_float2(0,0);
        if(q<int(ne01.z))value=((const float2 *)(Q+sequence*nb03+q*nb01+h*nb02))[d];
        qs[c*stride+d]=make_half2(value.x*scale,value.y*scale);
    }
    using input=tile<16,8,half2,DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using accum=tile<16,16,float,DATA_LAYOUT_I_MAJOR>;
    accum output[8];
    float maximum=-FLT_MAX/2,rowsum=0;
    if(sinks && blockIdx.y==0) {
        maximum=((const float *)sinks)[head0+column%8]+FATTN_KQ_MAX_OFFSET;
        rowsum=expf(-FATTN_KQ_MAX_OFFSET);
    }
    __syncthreads();
    const int stop=KV_max ? min(ne11,KV_max[sequence*gridDim.x+blockIdx.x]) : ne11;
    for(int start=blockIdx.y*kv_rows;start<stop;start+=gridDim.y*kv_rows) {
        bool live=false;
        for(int i=tid;i<queries*kv_rows;i+=nthreads) {
            const int q=query0+i/kv_rows,r=start+i%kv_rows;
            const half bias=q<int(ne01.z) && r<stop ? masks[q*(nb31/2)+r] : half(-INFINITY);
            biases[i]=bias;
            live |= __half_as_ushort(bias)!=0xfc00;
        }
        if(!__syncthreads_or(live))continue;
        for(int i=tid;i<kv_rows*256;i+=nthreads) {
            const int r=i/256,d=i%256;
            kv[r*stride+d]=start+r<stop ? kbase[(start+r)*(nb11/4)+d] : make_half2(0,0);
        }
        __syncthreads();
        accum scores[kv_rows/16];
#pragma unroll 2
        for(int d=vpart*64;d<(vpart+1)*64;d+=8) {
            input b;
            load_ldmatrix(b,qs+group*16*stride+d,stride);
#pragma unroll
            for(int r=0;r<kv_rows;r+=16) {
                input a;load_ldmatrix(a,kv+r*stride+d,stride);
                mma(scores[r/16],a,b);
            }
        }
#pragma unroll
        for(int r=0;r<kv_rows;r+=16) {
#pragma unroll
            for(int l=0;l<accum::ne;++l) {
                partials[(vpart*cols+column)*partial_stride+r+accum::get_j(l)]=scores[r/16].x[l];
            }
        }
        __syncthreads();
        float next=maximum;
#pragma unroll
        for(int r=0;r<kv_rows;r+=16) {
#pragma unroll
            for(int l=0;l<accum::ne;++l) {
                const int offset=column*partial_stride+r+accum::get_j(l);
                float sum=partials[offset];
#pragma unroll
                for(int part=1;part<4;++part)sum+=partials[part*cols*partial_stride+offset];
                scores[r/16].x[l]=sum;
                scores[r/16].x[l]+=__half2float(biases[(column/8)*kv_rows+r+accum::get_j(l)]);
                next=fmaxf(next,scores[r/16].x[l]+FATTN_KQ_MAX_OFFSET);
            }
        }
        next=fmaxf(next,__shfl_xor_sync(0xffffffff,next,16,32));
        float factor=expf(maximum-next);
        if(maximum-next<SOFTMAX_FTZ_THRESHOLD)factor=0;
        float sum=0;
#pragma unroll
        for(int r=0;r<kv_rows;r+=16) {
#pragma unroll
            for(int l=0;l<accum::ne;++l) {
                const float value=expf(scores[r/16].x[l]-next);
                scores[r/16].x[l]=value;sum+=value;
            }
        }
        sum+=__shfl_xor_sync(0xffffffff,sum,16,32);
        rowsum=rowsum*factor+sum;maximum=next;
#pragma unroll
        for(int d=0;d<8;++d) {
#pragma unroll
            for(int l=0;l<accum::ne;++l)output[d].x[l]*=factor;
        }
        input probabilities[kv_rows/16];
#pragma unroll
        for(int r=0;r<kv_rows/16;++r)probabilities[r]=get_half2(scores[r]);
        // The partial-sum barrier already retired every wave's K reads.
        if(!shared_kv) {
            for(int i=tid;i<kv_rows*256;i+=nthreads) {
                const int r=i/256,d=i%256;
                kv[r*stride+d]=start+r<stop ? vbase[(start+r)*(nb21/4)+d] : make_half2(0,0);
            }
            __syncthreads();
        }
#pragma unroll
        for(int d=0;d<8;++d) {
#pragma unroll
            for(int r=0;r<kv_rows/16;++r) {
                input a;
                load_ldmatrix_trans_swizzled<stride>(a,kv,16*r*stride+vpart*64+d*8);
                mma(output[d],a,probabilities[r]);
            }
        }
        __syncthreads();
    }
    float * meta=(float *)qs;
    if(vpart==0 && lane<16) {
        meta[2*column]=maximum;meta[2*column+1]=rowsum;
        const int q=query0+column/8,h=head0+column%8;
        if(q<int(ne01.z) && gridDim.y>1)dst_meta[((sequence*int(ne01.z)+q)*ne02+h)*gridDim.y+blockIdx.y]=make_float2(maximum,rowsum);
    }
    __syncthreads();
    // Reuse KV storage to transpose one 128-output slice into coalesced stores.
    float * stage=(float *)kv;
    for(int part=0;part<4;++part) {
        if(vpart==part) {
#pragma unroll
            for(int d=0;d<8;++d) {
#pragma unroll
                for(int l=0;l<accum::ne;++l)stage[column*128+d*16+accum::get_j(l)]=output[d].x[l];
            }
        }
        __syncthreads();
        for(int i=tid;i<cols*32;i+=nthreads) {
            const int c=i/32,d=(i%32)*4,q=query0+c/8,h=head0+c%8;
            if(q<int(ne01.z)) {
                float4 value=((float4 *)stage)[i];
                if(gridDim.y==1) {
                    const float inv=1.f/meta[2*c+1];
                    value.x*=inv;value.y*=inv;value.z*=inv;value.w*=inv;
                }
                const int64_t row=((sequence*int(ne01.z)+q)*ne02+h)*gridDim.y+blockIdx.y;
                *(float4 *)(dst+row*512+part*128+d)=value;
            }
        }
        __syncthreads();
    }
#else
    GGML_UNUSED_VARS(Q,K,V,mask,sinks,KV_max,dst,dst_meta,scale,max_bias,m0,m1,n_head_log2,logit_softcap,
                    ne00,ne01,ne02,ne03,nb01,nb02,nb03,ne10,ne11,ne12,ne13,nb11,nb12,nb13,
                    nb21,nb22,nb23,ne31,ne32,ne33,nb31,nb32,nb33);
    NO_DEVICE_CODE;
#endif
}

template<int queries,int rows>
static void ggml_hip_fattn_vsplit(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
    constexpr size_t shared=(queries*8+rows)*260*sizeof(half2)+queries*rows*sizeof(half)+4*(queries*8)*(rows+4)*sizeof(float);
    launch_fattn<512,queries,8>(ctx,dst,hip_fattn_wmma_vsplit<queries,rows>,queries*2,shared,rows,true,true,false,false,32,false,true);
}
#endif
