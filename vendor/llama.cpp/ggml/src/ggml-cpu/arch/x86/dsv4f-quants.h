#pragma once

// Wide CPU expert-miss dots. Keep these behind the ISA selector so numerical
// and timing comparisons can exercise the original ggml implementation too.
// Q8_K activation quantizers produce [-127, 127], allowing sign transfer in
// dsv4f_dot_signed64 without signed-byte overflow.

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VNNI__) && defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#define DSV4F_QUANT_AVX512 1

static inline __m512i dsv4f_pair256(__m256i lo, __m256i hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}
static inline __m512 dsv4f_scale_pair(float lo, float hi) {
    return _mm512_insertf32x8(_mm512_castps256_ps512(_mm256_set1_ps(lo)), _mm256_set1_ps(hi), 1);
}
static inline __m512i dsv4f_dot_signed64(__m512i weights, __m512i values) {
    const __m512i zero = _mm512_setzero_si512();
    values = _mm512_mask_sub_epi8(values, _mm512_cmplt_epi8_mask(weights, zero), zero, values);
    return _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(weights), values);
}
static inline __m512i dsv4f_scales16(__m512i scales, int group) {
    const __m512i index = _mm512_add_epi32(_mm512_setr_epi32(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3), _mm512_set1_epi32(group*4));
    return _mm512_permutexvar_epi32(index, scales);
}
// One 32-byte bit-plane array represents four consecutive groups of 32 values.
static inline __m512i dsv4f_planes64(const uint8_t * bytes, int group, int mask, int high_shift) {
    const __m512i packed = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)bytes));
    const __m512i shifts = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16(group*high_shift)), _mm256_set1_epi16(group*high_shift+high_shift/2),1);
    return _mm512_and_si512(_mm512_srlv_epi16(packed,shifts), _mm512_set1_epi8(mask));
}
static inline __m512i dsv4f_high_mask(const uint8_t * bytes, int group) {
    const __m512i h = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)bytes));
    const __m512i bits = dsv4f_pair256(_mm256_set1_epi8(1 << (group*2)),_mm256_set1_epi8(2 << (group*2)));
    return _mm512_and_si512(h,bits);
}
static float dsv4f_dot_q2_0_512(int n, const block_q2_0 * x, const block_q8_0 * y) {
    const __m512i mask = _mm512_set1_epi32(3), zero = _mm512_setzero_si512(), one = _mm512_set1_epi8(1);
    __m512 sum = _mm512_setzero_ps();
    for (int b=0;b<n/64;++b) {
        const __m512i bits = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)x[b].qs));
        __m512i w = _mm512_and_si512(bits,mask);
        w = _mm512_or_si512(w,_mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(bits,2),mask),8));
        w = _mm512_or_si512(w,_mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(bits,4),mask),16));
        w = _mm512_or_si512(w,_mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(bits,6),mask),24));
        const __m512i a = dsv4f_pair256(_mm256_loadu_si256((const __m256i *)y[2*b].qs),_mm256_loadu_si256((const __m256i *)y[2*b+1].qs));
        const __m512i dots = _mm512_sub_epi32(_mm512_dpbusd_epi32(zero,w,a),_mm512_dpbusd_epi32(zero,one,a));
        const float d = GGML_CPU_FP16_TO_FP32(x[b].d);
        sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots),dsv4f_scale_pair(d*GGML_CPU_FP16_TO_FP32(y[2*b].d),d*GGML_CPU_FP16_TO_FP32(y[2*b+1].d)),sum);
    }
    return _mm512_reduce_add_ps(sum);
}
static float dsv4f_dot_q4_0_512(int n, const block_q4_0 * x, const block_q8_0 * y) {
    const __m512i zero = _mm512_setzero_si512(), one = _mm512_set1_epi8(1);
    __m512 sum = _mm512_setzero_ps(); int b=0;
    for (;b+1<n/32;b+=2) {
        const __m512i w = dsv4f_pair256(bytes_from_nibbles_32(x[b].qs),bytes_from_nibbles_32(x[b+1].qs));
        const __m512i a = dsv4f_pair256(_mm256_loadu_si256((const __m256i *)y[b].qs),_mm256_loadu_si256((const __m256i *)y[b+1].qs));
        const __m512i dots = _mm512_sub_epi32(_mm512_dpbusd_epi32(zero,w,a),_mm512_slli_epi32(_mm512_dpbusd_epi32(zero,one,a),3));
        sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dots),dsv4f_scale_pair(GGML_CPU_FP16_TO_FP32(x[b].d)*GGML_CPU_FP16_TO_FP32(y[b].d),GGML_CPU_FP16_TO_FP32(x[b+1].d)*GGML_CPU_FP16_TO_FP32(y[b+1].d)),sum);
    }
    float tail=0;
    if(b<n/32){int dot=0;for(int j=0;j<16;++j){dot+=((x[b].qs[j]&15)-8)*y[b].qs[j];dot+=((x[b].qs[j]>>4)-8)*y[b].qs[j+16];}tail=dot*GGML_CPU_FP16_TO_FP32(x[b].d)*GGML_CPU_FP16_TO_FP32(y[b].d);}
    return _mm512_reduce_add_ps(sum)+tail;
}
static float dsv4f_dot_q3_K_512(int n, const block_q3_K * x, const block_q8_K * y) {
    __m512 sum = _mm512_setzero_ps();
    for (int b=0;b<n/256;++b) {
        uint32_t a[3];memcpy(a,x[b].scales,12);
        __m128i sc = _mm_set_epi32(((a[1]>>4)&0x0f0f0f0f)|(((a[2]>>6)&0x03030303)<<4),((a[0]>>4)&0x0f0f0f0f)|(((a[2]>>4)&0x03030303)<<4),(a[1]&0x0f0f0f0f)|(((a[2]>>2)&0x03030303)<<4),(a[0]&0x0f0f0f0f)|((a[2]&0x03030303)<<4));
        const __m512i scales = _mm512_cvtepi8_epi32(_mm_sub_epi8(sc,_mm_set1_epi8(32)));
        __m512i total = _mm512_setzero_si512();
        for(int g=0;g<4;++g){
            const __m512i lo = dsv4f_planes64(x[b].qs+(g/2)*32,g%2,3,4);
            const __mmask64 neg = _mm512_cmpeq_epi8_mask(dsv4f_high_mask(x[b].hmask,g),_mm512_setzero_si512());
            const __m512i w = _mm512_sub_epi8(lo,_mm512_maskz_mov_epi8(neg,_mm512_set1_epi8(4)));
            const __m512i dot = dsv4f_dot_signed64(w,_mm512_loadu_si512(y[b].qs+g*64));
            total = _mm512_add_epi32(total,_mm512_mullo_epi32(dot,dsv4f_scales16(scales,g)));
        }
        sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(total),_mm512_set1_ps(y[b].d*GGML_CPU_FP16_TO_FP32(x[b].d)),sum);
    }
    return _mm512_reduce_add_ps(sum);
}
// The first eight bytes contain low/high nibbles for scales 0..7/8..15.
// The last four bytes carry two more bits per scale, in groups of four.
// Exact 8+4-byte loads also work at the final block without overreading.
static inline __m512i dsv4f_q3_scales_512(const uint8_t * packed) {
    const __m128i low = _mm_loadl_epi64((const __m128i *) packed);
    const __m128i nibbles = _mm_and_si128(
        _mm_unpacklo_epi64(low, _mm_srli_epi16(low, 4)), _mm_set1_epi8(15));
    uint32_t high;
    memcpy(&high, packed + 8, sizeof(high));
    const __m128i upper = _mm_and_si128(
        _mm_srlv_epi32(_mm_set1_epi32(high), _mm_setr_epi32(0, 2, 4, 6)),
        _mm_set1_epi32(0x03030303));
    const __m128i scales = _mm_or_si128(nibbles, _mm_slli_epi16(upper, 4));
    return _mm512_cvtepi8_epi32(_mm_sub_epi8(scales, _mm_set1_epi8(32)));
}

static float dsv4f_dot_q3_K_offset_512(int n, const block_q3_K * x, const block_q8_K * y) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i four = _mm512_set1_epi8(4);
    __m512 sum = _mm512_setzero_ps();
    for (int b = 0; b < n / QK_K; ++b) {
        const __m512i scales = dsv4f_q3_scales_512(x[b].scales);
        __m512i total = zero;
        for (int g = 0; g < 4; ++g) {
            const __m512i low = dsv4f_planes64(x[b].qs + (g / 2) * 32, g % 2, 3, 4);
            const __mmask64 high = _mm512_cmpneq_epi8_mask(dsv4f_high_mask(x[b].hmask, g), zero);
            // q = low - 4 * !high, so q + 4 is unsigned in [0, 7].
            // This avoids transferring q's sign to the activation bytes.
            const __m512i weights = _mm512_or_si512(low, _mm512_maskz_mov_epi8(high, four));
            const __m512i dot = _mm512_dpbusd_epi32(zero, weights, _mm512_loadu_si512(y[b].qs + g * 64));
            total = _mm512_add_epi32(total, _mm512_mullo_epi32(dot, dsv4f_scales16(scales, g)));
        }
        // Undo +4 with the existing sum of each 16 activation bytes. Apply
        // each signed scale before the correction; scales differ per group.
        const __m512i sums = _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *) y[b].bsums));
        total = _mm512_sub_epi32(total, _mm512_slli_epi32(_mm512_mullo_epi32(scales, sums), 2));
        sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(total),
                            _mm512_set1_ps(y[b].d * GGML_CPU_FP16_TO_FP32(x[b].d)), sum);
    }
    return _mm512_reduce_add_ps(sum);
}

static inline void dsv4f_accum_q45_K(__m512 * sum,float * mins,const uint8_t * qs,const uint8_t * qh,const uint8_t * scales,float d,float dmin,const block_q8_K * y) {
    uint32_t aux[4];memcpy(aux,scales,12);
    aux[3]=((aux[2]>>4)&0x0f0f0f0f)|(((aux[1]>>6)&0x03030303)<<4);
    const uint32_t middle=aux[1]&0x3f3f3f3f;
    aux[1]=(aux[2]&0x0f0f0f0f)|(((aux[0]>>6)&0x03030303)<<4);
    aux[2]=middle;aux[0]&=0x3f3f3f3f;
    // Unpack eight 6-bit scales and eight minima, then apply each scale to
    // its 32 weights. Minima use the precomputed Q8_K sums instead of a dot.
    const __m512i allsc=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)aux));
    const __m256i min8=_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(aux+2)));
    const __m256i q8sums=_mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)y->bsums),_mm256_set1_epi16(1));
    const int minsum=_mm512_reduce_add_epi32(dsv4f_pair256(_mm256_mullo_epi32(min8,q8sums),_mm256_setzero_si256()));
    const __m512i groups=_mm512_setr_epi32(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1);
    __m512i total = _mm512_setzero_si512();
    for(int g=0;g<4;++g){
        const __m256i bits = _mm256_loadu_si256((const __m256i *)(qs+g*32));
        __m512i w = dsv4f_pair256(_mm256_and_si256(bits,_mm256_set1_epi8(15)),_mm256_and_si256(_mm256_srli_epi16(bits,4),_mm256_set1_epi8(15)));
        if(qh){const __mmask64 high=_mm512_cmpneq_epi8_mask(dsv4f_high_mask(qh,g),_mm512_setzero_si512());w=_mm512_or_si512(w,_mm512_maskz_mov_epi8(high,_mm512_set1_epi8(16)));}
        const __m512i sc = _mm512_permutexvar_epi32(_mm512_add_epi32(groups,_mm512_set1_epi32(2*g)),allsc);
        total = _mm512_add_epi32(total,_mm512_mullo_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(),w,_mm512_loadu_si512(y->qs+g*64)),sc));
    }
    *sum=_mm512_fmadd_ps(_mm512_cvtepi32_ps(total),_mm512_set1_ps(y->d*d),*sum);*mins+=y->d*dmin*minsum;
}
static float dsv4f_dot_q4_K_512(int n,const block_q4_K * x,const block_q8_K * y) {
    __m512 sum=_mm512_setzero_ps();float mins=0;for(int b=0;b<n/256;++b)dsv4f_accum_q45_K(&sum,&mins,x[b].qs,NULL,x[b].scales,GGML_CPU_FP16_TO_FP32(x[b].d),GGML_CPU_FP16_TO_FP32(x[b].dmin),y+b);return _mm512_reduce_add_ps(sum)-mins;
}
static float dsv4f_dot_q5_K_512(int n,const block_q5_K * x,const block_q8_K * y) {
    __m512 sum=_mm512_setzero_ps();float mins=0;for(int b=0;b<n/256;++b)dsv4f_accum_q45_K(&sum,&mins,x[b].qs,x[b].qh,x[b].scales,GGML_CPU_FP16_TO_FP32(x[b].d),GGML_CPU_FP16_TO_FP32(x[b].dmin),y+b);return _mm512_reduce_add_ps(sum)-mins;
}
static float dsv4f_dot_q2_K_512(int n,const block_q2_K * x,const block_q8_K * y) {
    __m512 sum=_mm512_setzero_ps();float mins=0;
    for(int b=0;b<n/256;++b){
        const __m512i rawsc=_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)x[b].scales));
        const __m512i sc=_mm512_and_si512(rawsc,_mm512_set1_epi32(15));
        const int minsum=_mm512_reduce_add_epi32(_mm512_mullo_epi32(_mm512_srli_epi32(rawsc,4),_mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)y[b].bsums))));
        __m512i total=_mm512_setzero_si512();
        for(int g=0;g<4;++g){const __m512i w=dsv4f_planes64(x[b].qs+(g/2)*32,g%2,3,4);total=_mm512_add_epi32(total,_mm512_mullo_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(),w,_mm512_loadu_si512(y[b].qs+g*64)),dsv4f_scales16(sc,g)));}
        sum=_mm512_fmadd_ps(_mm512_cvtepi32_ps(total),_mm512_set1_ps(y[b].d*GGML_CPU_FP16_TO_FP32(x[b].d)),sum);mins+=y[b].d*GGML_CPU_FP16_TO_FP32(x[b].dmin)*minsum;
    }
    return _mm512_reduce_add_ps(sum)-mins;
}
static float dsv4f_dot_q6_K_512(int n,const block_q6_K * x,const block_q8_K * y) {
    __m512 sum=_mm512_setzero_ps();
    for(int b=0;b<n/256;++b){
        const __m512i sc=_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)x[b].scales));__m512i total=_mm512_setzero_si512();
        for(int g=0;g<4;++g){
            __m512i low=_mm512_loadu_si512(x[b].ql+(g/2)*64);if(g%2)low=_mm512_srli_epi16(low,4);low=_mm512_and_si512(low,_mm512_set1_epi8(15));
            const __m512i high=_mm512_slli_epi16(dsv4f_planes64(x[b].qh+(g/2)*32,g%2,3,4),4);
            const __m512i w=_mm512_sub_epi8(_mm512_or_si512(low,high),_mm512_set1_epi8(32));
            total=_mm512_add_epi32(total,_mm512_mullo_epi32(dsv4f_dot_signed64(w,_mm512_loadu_si512(y[b].qs+g*64)),dsv4f_scales16(sc,g)));
        }
        sum=_mm512_fmadd_ps(_mm512_cvtepi32_ps(total),_mm512_set1_ps(y[b].d*GGML_CPU_FP16_TO_FP32(x[b].d)),sum);
    }
    return _mm512_reduce_add_ps(sum);
}
#endif
