#pragma once
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

// Small graph operations shared by CPU and CUDA/HIP. History is deliberately
// outside graph operands: uploading the authoritative host archive would defeat
// the bounded sliding window. The cache owns it for the lifetime of the graph.
constexpr int GGML_DSV41_HISTORY = 0x41524831;
constexpr int GGML_DSV41_MASK = 0x414d5331;
constexpr int GGML_DSV41_DIRECT_KV = 0x414b5631;
struct ggml_dsv41_history {
    ggml_tensor * host = nullptr;
    void * mapped = nullptr;
};
using ggml_dsv41_sync_fn = void (*)(ggml_backend_dev_t);
using ggml_dsv41_map_host_fn = void * (*)(ggml_backend_dev_t, void *);

// Match HIP sparse attention's finite-row packing before the FP16 tile. The
// gather's finite rows already follow physical cache order; masked gaps must
// move behind them, especially while the initial SWA window is not full.
inline void ggml_dsv41_pack_kv(ggml_context * ctx, ggml_tensor *& keys, ggml_tensor *& mask) {
    const int rows = keys->ne[1];
    auto * order = ggml_argsort(ctx, ggml_sub(ctx, ggml_arange(ctx, 0, rows, 1),
                               ggml_cast(ctx, mask, GGML_TYPE_F32)), GGML_SORT_ORDER_ASC);
    keys = ggml_get_rows(ctx, keys, order);
    mask = ggml_cast(ctx, ggml_reshape_2d(ctx, ggml_get_rows(ctx,
           ggml_reshape_2d(ctx, mask, 1, rows), order), rows, 1), GGML_TYPE_F16);
}
inline ggml_dsv41_history * ggml_dsv41_history_of(const ggml_tensor * t) {
    ggml_dsv41_history * p;
    std::memcpy(&p, t->op_params + 12, sizeof(p));
    return p;
}
inline void ggml_dsv41_cache_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    if (dst->op_params[9] == GGML_DSV41_DIRECT_KV) {
        const int nr=dst->op_params[10], ns=dst->op_params[11], rows=dst->op_params[12];
        const int dim=dst->src[0]->ne[0];
        auto * out=(ggml_fp16_t *)dst->data;
        std::fill(out,out+dim*rows,ggml_fp32_to_fp16(0.f));
        std::fill(out+dim*rows,out+(dim+1)*rows,ggml_fp32_to_fp16(-INFINITY));
        std::fill(out+(dim+1)*rows,out+(dim+2)*rows,ggml_fp32_to_fp16(-1.f));
        std::vector<float> values(dim);
        int next=0;
        for(int i=0;i<nr+ns;++i) {
            const int j=i<nr?0:3, index=i<nr?i:i-nr;
            const auto * mask=dst->src[j+2];
            const float m=mask->type==GGML_TYPE_F16?ggml_fp16_to_fp32(((ggml_fp16_t *)mask->data)[index]):((float *)mask->data)[index];
            if(!std::isfinite(m))continue;
            const int row=((int32_t *)dst->src[j+1]->data)[index];
            const auto * cache=dst->src[j];
            ggml_get_type_traits(cache->type)->to_float((const char *)cache->data+row*cache->nb[1],values.data(),dim);
            for(int k=0;k<dim;++k)out[next*dim+k]=ggml_fp32_to_fp16(values[k]);
            out[dim*rows+next]=ggml_fp32_to_fp16(0.f);
            out[(dim+1)*rows+next]=ggml_fp32_to_fp16(float(i));
            ++next;
        }
    } else if (dst->op_params[9] == GGML_DSV41_HISTORY) {
        auto * h = ggml_dsv41_history_of(dst);
        const auto * cache = dst->src[0];
        const auto * slots = (const int64_t *) dst->src[1]->data;
        const auto * pos = (const int64_t *) dst->src[2]->data;
        for (int64_t t = ith; t < dst->src[1]->ne[0]; t += nth) {
            GGML_ASSERT(pos[t] >= 0 && pos[t] < h->host->ne[1]);
            std::memcpy((char *) h->host->data + pos[t]*h->host->nb[1],
                        (const char *) cache->data + slots[t]*cache->nb[1], cache->nb[1]);
        }
    } else {
        const auto * pos = (const int64_t *) dst->src[0]->data;
        const int ratio = dst->op_params[10], window = dst->op_params[11];
        for (int64_t i = ith; i < ggml_nelements(dst); i += nth) {
            const int64_t row = i % dst->ne[0], p = pos[i / dst->ne[0]];
            const int64_t delta = (p % dst->ne[0] - row + dst->ne[0]) % dst->ne[0];
            const bool visible = ratio ? row < (p + 1)/ratio : delta < window && delta <= p;
            ((ggml_fp16_t *) dst->data)[i] = ggml_fp32_to_fp16(visible ? 0.f : -INFINITY);
        }
    }
}
inline ggml_tensor * ggml_dsv41_archive(ggml_context * ctx, ggml_dsv41_history * h,
        ggml_tensor * cache, ggml_tensor * slots, ggml_tensor * positions) {
    if (!h->host) return nullptr;
    ggml_tensor * args[] = {cache, slots, positions};
    auto * out = ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 3, ggml_dsv41_cache_cpu, 1, nullptr);
    out->op_params[9] = GGML_DSV41_HISTORY;
    std::memcpy(out->op_params + 12, &h, sizeof(h));
    return out;
}
inline ggml_tensor * ggml_dsv41_mask(ggml_context * ctx, ggml_tensor * positions,
        int rows, int ratio, int window) {
    ggml_tensor * args[] = {positions};
    auto * out = ggml_custom_4d(ctx, GGML_TYPE_F16, rows, positions->ne[0], 1, 1,
                              args, 1, ggml_dsv41_cache_cpu, 1, nullptr);
    out->op_params[9] = GGML_DSV41_MASK;
    out->op_params[10] = ratio;
    out->op_params[11] = window;
    return out;
}

// Single-query Q8 cache gather, stable finite-row packing, padding and FP16
// conversion. Masked KV values are zero; attention never consumes those rows.
inline void ggml_dsv41_direct_kv(ggml_context * ctx, ggml_tensor * raw,
        ggml_tensor * raw_ids, ggml_tensor * raw_mask, ggml_tensor * comp,
        ggml_tensor * selected, ggml_tensor * selected_mask,
        ggml_tensor *& keys, ggml_tensor *& mask) {
    const int nr=ggml_nelements(raw_ids), ns=comp?ggml_nelements(selected):0;
    const int dim=raw->ne[0], rows=GGML_PAD(nr+ns,256);
    GGML_ASSERT(raw->type==GGML_TYPE_Q8_0 && (!comp || comp->type==GGML_TYPE_Q8_0));
    GGML_ASSERT(nr>0 && rows<=1024 && raw_ids->type==GGML_TYPE_I32 && raw_mask->type==GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(raw_ids) && ggml_is_contiguous(raw_mask) && ggml_nelements(raw_mask)==nr);
    GGML_ASSERT(!comp || (comp->ne[0]==dim && ggml_nelements(selected_mask)==ns &&
                (selected_mask->type==GGML_TYPE_F16 || selected_mask->type==GGML_TYPE_F32)));
    GGML_ASSERT(!comp || (selected->type==GGML_TYPE_I32 && ggml_is_contiguous(selected) && ggml_is_contiguous(selected_mask)));
    ggml_tensor * args[]={raw,raw_ids,raw_mask,comp?comp:raw,comp?selected:raw_ids,comp?selected_mask:raw_mask};
    auto * out=ggml_custom_4d(ctx,GGML_TYPE_F16,(dim+2)*rows,1,1,1,args,6,ggml_dsv41_cache_cpu,1,nullptr);
    out->op_params[9]=GGML_DSV41_DIRECT_KV;
    out->op_params[10]=nr;out->op_params[11]=ns;out->op_params[12]=rows;
    keys=ggml_view_2d(ctx,out,dim,rows,dim*sizeof(ggml_fp16_t),0);
    mask=ggml_view_2d(ctx,out,rows,1,rows*sizeof(ggml_fp16_t),dim*rows*sizeof(ggml_fp16_t));
}
