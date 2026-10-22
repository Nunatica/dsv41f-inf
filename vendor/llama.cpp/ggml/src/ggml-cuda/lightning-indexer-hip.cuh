#pragma once

#if defined(GGML_USE_HIP)
#include "mma.cuh"

static __device__ __forceinline__ bool hip_indexer_half_range(float v) {
    const float a=fabsf(v);
    return isfinite(v) && a<=65504.f && (a==0 || a>=0x1p-24f);
}

// One wave scores 16 keys against 32 heads. ReLU is applied only after
// accumulating the complete 128-element dot; head weights remain FP32.
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_hip_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    using namespace ggml_cuda_mma;
    static_assert(N_EMBD == 128 && N_HEAD == 32);
    constexpr int waves_per_query = K_VECS_PER_BLOCK/16;
    constexpr int queries_per_block = WARPS_PER_BLOCK/waves_per_query;
    static_assert(WARPS_PER_BLOCK%waves_per_query == 0);
    constexpr int stride = N_EMBD/2 + 4;
    const int lane = threadIdx.x;
    const int wave = threadIdx.y;
    const int tid = wave*32 + lane;
    const int query = blockIdx.y*queries_per_block+wave/waves_per_query;
    const int stream = blockIdx.z;
    const int first_key = blockIdx.x*K_VECS_PER_BLOCK;
    __shared__ half2 qs[queries_per_block*N_HEAD][stride];
    __shared__ half2 ks[K_VECS_PER_BLOCK][stride];
    __shared__ float ws[queries_per_block*N_HEAD];
    __shared__ int use_float;
    if (tid==0) use_float=0;
    __syncthreads();
    if (tid < queries_per_block*N_HEAD) {
        const int t=blockIdx.y*queries_per_block+tid/N_HEAD;
        ws[tid] = t<n_batch ? *(const float *) ((const char *) W + (tid%N_HEAD)*sizeof(float) + t*nbw1 + stream*nbw3) : 0;
    }
    for (int i=tid; i<queries_per_block*N_HEAD*(N_EMBD/4); i+=WARPS_PER_BLOCK*32) {
        const int h=i/(N_EMBD/4), c=i%(N_EMBD/4);
        const int t=blockIdx.y*queries_per_block+h/N_HEAD;
        const float4 v = t<n_batch ? *(const float4 *) ((const char *) Q+t*nbq2+stream*nbq3+(h%N_HEAD)*nbq1+c*sizeof(float4)) : make_float4(0,0,0,0);
        if (!hip_indexer_half_range(v.x) || !hip_indexer_half_range(v.y) ||
            !hip_indexer_half_range(v.z) || !hip_indexer_half_range(v.w)) atomicExch(&use_float,1);
        qs[h][2*c] = make_half2(v.x, v.y);
        qs[h][2*c+1] = make_half2(v.z, v.w);
    }
    for (int i=tid; i<K_VECS_PER_BLOCK*(N_EMBD/4); i+=WARPS_PER_BLOCK*32) {
        const int r=i/(N_EMBD/4), c=i%(N_EMBD/4);
        float4 v=make_float4(0,0,0,0);
        if (first_key+r < n_kv) {
            const void * row=K+(first_key+r)*nbk2+stream*nbk3;
            constexpr dequantize_V_t dequantize_k=get_dequantize_V<TYPE_K,float,4>();
            dequantize_k(row,&v,4*c);
        }
        if (!hip_indexer_half_range(v.x) || !hip_indexer_half_range(v.y) ||
            !hip_indexer_half_range(v.z) || !hip_indexer_half_range(v.w)) atomicExch(&use_float,1);
        ks[r][2*c] = make_half2(v.x, v.y);
        ks[r][2*c+1] = make_half2(v.z, v.w);
    }
    __syncthreads();
    // Retain FP32 input range for unusual tensors, without a host readback or
    // graph-dependent allocation. All waves take the same block-uniform branch.
    if (use_float) {
        if (query>=n_batch) return;
        for (int r=0; r<16; ++r) {
            const int key=first_key+(wave%waves_per_query)*16+r;
            if (key>=n_kv) break;
            float4 kv;
            constexpr dequantize_V_t dequantize_k=get_dequantize_V<TYPE_K,float,4>();
            dequantize_k(K+key*nbk2+stream*nbk3,&kv,lane*4);
            float score=0;
            for (int h=0; h<N_HEAD; ++h) {
                const float4 qv=*(const float4 *) ((const char *) Q+query*nbq2+stream*nbq3+h*nbq1+lane*sizeof(float4));
                float dot=0;
                ggml_cuda_mad(dot,qv.x,kv.x); ggml_cuda_mad(dot,qv.y,kv.y);
                ggml_cuda_mad(dot,qv.z,kv.z); ggml_cuda_mad(dot,qv.w,kv.w);
                dot=warp_reduce_sum(dot);
                score+=(dot>0 ? dot : 0)*ws[(wave/waves_per_query)*N_HEAD+h];
            }
            if (lane==0) {
                const half * mask=(const half *) ((const char *) M+query*nbm1+(stream%nem3)*nbm3);
                float * out=(float *) ((char *) dst+query*nb1+stream*nb3);
                out[key]=score+__half2float(mask[key]);
            }
        }
        return;
    }
    using input_tile=tile<16,8,half2,DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using output_tile=tile<16,16,float,DATA_LAYOUT_I_MAJOR>;
    float score=0;
#pragma unroll
    for (int h=0; h<N_HEAD; h+=16) {
        output_tile acc;
#pragma unroll
        for (int c=0; c<N_EMBD/2; c+=8) {
            input_tile a,b;
            load_generic(a,&ks[(wave%waves_per_query)*16][c],stride);
            load_generic(b,&qs[(wave/waves_per_query)*N_HEAD+h][c],stride);
            // RDNA3 native C keeps keys in lanes and heads in fragment elements.
            mma(acc,b,a);
        }
#pragma unroll
        for (int l=0; l<output_tile::ne; ++l) {
            const float dot=acc.x[l];
            score += (dot>0 ? dot : 0)*ws[(wave/waves_per_query)*N_HEAD+h+output_tile::get_j(l)];
        }
    }
    score += __shfl_xor_sync(0xffffffff,score,16,32);
    const int key=first_key+(wave%waves_per_query)*16+lane;
    if (lane<16 && key<n_kv && query<n_batch) {
        const half * mask=(const half *) ((const char *) M+query*nbm1+(stream%nem3)*nbm3);
        float * out=(float *) ((char *) dst+query*nb1+stream*nb3);
        out[key]=score+__half2float(mask[key]);
    }
#else
    GGML_UNUSED_VARS(Q,K,W,M,dst,n_stream,n_batch,n_kv,nb1,nb2,nb3,nbq1,nbq2,nbq3,
                    nbk1,nbk2,nbk3,nbw1,nbw2,nbw3,nbm1,nbm2,nbm3,nem3);
    NO_DEVICE_CODE;
#endif
}
#endif
