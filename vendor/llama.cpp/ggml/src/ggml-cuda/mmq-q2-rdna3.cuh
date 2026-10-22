#pragma once

// Cooperative Q2 -> signed INT8 LDS unpack. Keep each packed word's live range
// short rather than unrolling all rows, and load each scale entry only once.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_rdna3(
        const char * __restrict__ x, int * __restrict__ tile, int kbx0, int i_max, int stride) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    static_assert(type == GGML_TYPE_Q2_0, "Q2 loader only");
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);
    constexpr int threads = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8);
    constexpr int pitch = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int words_per_row = MMQ_ITER_K/16;
    constexpr int scales_per_row = MMQ_ITER_K/QK8_1;
    const int tid = threadIdx.y*32 + threadIdx.x;
    float * scales = reinterpret_cast<float *>(tile + 2*MMQ_TILE_NE_K);
#pragma unroll 1
    for (int index = tid; index < I*words_per_row; index += threads) {
        const int row = index/words_per_row, word = index%words_per_row;
        const int source_row = fallback ? min(row, i_max) : row;
        const auto * block = reinterpret_cast<const block_q2_0 *>(x) + kbx0 + source_row*stride + word/4;
        const auto * packed = reinterpret_cast<const uint16_t *>(block->qs) + 2*(word%4);
        const uint32_t bits = uint32_t(packed[0]) | uint32_t(packed[1])<<16;
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const uint32_t b = (bits >> (8*byte)) & 255;
            const uint32_t indices = (b & 3) | ((b & 12)<<6) | ((b & 48)<<12) | ((b & 192)<<18);
            tile[row*pitch + word*4 + byte] = __builtin_amdgcn_perm(0x020100ff, 0x020100ff, indices);
        }
    }
#pragma unroll 1
    for (int index = tid; index < I*scales_per_row; index += threads) {
        const int row = index/scales_per_row, entry = index%scales_per_row;
        const int source_row = fallback ? min(row, i_max) : row;
        const auto * block = reinterpret_cast<const block_q2_0 *>(x) + kbx0 + source_row*stride + entry/2;
        scales[row*pitch + entry] = __half2float(block->d);
    }
#else
    ggml_cuda_mmq_load_tiles_q2_0<type, J, fallback>(x, tile, kbx0, i_max, stride);
#endif
}

// Keep one K fragment live at a time. The generic dot's automatic unrolling
// otherwise overlaps multiple INT8 WMMA fragments and scale loads.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_dot_q2_rdna3(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, int k00) {
#if defined(AMD_WMMA_AVAILABLE) && defined(RDNA3)
    using namespace ggml_cuda_mma;
    using input = tile<16, 8, int, DATA_LAYOUT_I_MAJOR_MIRRORED>;
    using accum = tile<16, 16, int, DATA_LAYOUT_J_MAJOR>;
    constexpr int pitch = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    const int row0 = threadIdx.y*16;
    const float * dx = reinterpret_cast<const float *>(x + 2*MMQ_TILE_NE_K);
    const float * dy = reinterpret_cast<const float *>(y);
#pragma unroll 1
    for (int k=0; k<MMQ_TILE_NE_K; k+=QI8_0) {
        input a;
        load_ldmatrix(a, x + row0*pitch + k00+k, pitch);
#pragma unroll
        for (int j=0; j<J; j+=16) {
            input b;
            load_ldmatrix(b, y + 4 + j*MMQ_TILE_Y_K+k, MMQ_TILE_Y_K);
            accum c;
            mma(c, a, b);
            const float db = dy[(j+accum::get_j(0))*MMQ_TILE_Y_K+k/QI8_1];
#pragma unroll
            for (int l=0; l<accum::ne; ++l) {
                const float da = dx[(row0+accum::get_i(l))*pitch+(k00+k)/QI8_0];
                sum[(j/16)*accum::ne+l] += c.x[l]*da*db;
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif
}
