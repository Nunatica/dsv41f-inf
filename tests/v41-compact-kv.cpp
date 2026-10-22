#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-dsv41-cache.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Compare production-size graph preparation to an independent host oracle and
// the original HIP sparse attention, without expert/cache/session differences.
static bool check(ggml_backend_t gpu, int comp_rows, bool sm_float=false) {
    constexpr int dim=512, window=4352, swa=128, top=512;
    const int selected=comp_rows?top:0, bound=swa+selected;
    const int full_rows=GGML_PAD(window+comp_rows,256), packed_rows=GGML_PAD(bound,256);
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});
    auto * raw=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,dim,window);
    auto * comp=comp_rows?ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,dim,comp_rows):nullptr;
    auto * ri=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,swa);
    auto * si=comp_rows?ggml_new_tensor_1d(ctx,GGML_TYPE_I32,selected):nullptr;
    auto * rm=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,swa,1);
    auto * sm=comp_rows?ggml_new_tensor_2d(ctx,sm_float?GGML_TYPE_F32:GGML_TYPE_F16,selected,1):nullptr;
    auto * fm=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,full_rows,1);
    auto * q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dim,1,64);
    auto * sinks=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,64);
    auto * full=ggml_cast(ctx,raw,GGML_TYPE_F32);
    auto * keys=ggml_get_rows(ctx,raw,ri);
    auto * mask=rm;
    if(comp_rows) {
        full=ggml_concat(ctx,full,ggml_cast(ctx,comp,GGML_TYPE_F32),1);
        keys=ggml_concat(ctx,keys,ggml_get_rows(ctx,comp,si),1);
        mask=ggml_concat(ctx,mask,ggml_cast(ctx,sm,GGML_TYPE_F16),0);
    }
    if(full_rows>window+comp_rows)full=ggml_pad(ctx,full,0,full_rows-window-comp_rows,0,0);
    full=ggml_cast(ctx,full,GGML_TYPE_F16);
    ggml_dsv41_pack_kv(ctx,keys,mask);
    ggml_tensor * direct_keys=nullptr,*direct_mask=nullptr;
    ggml_dsv41_direct_kv(ctx,raw,ri,rm,comp,si,sm,direct_keys,direct_mask);
    auto * direct=ggml_flash_attn_ext(ctx,q,direct_keys,direct_keys,direct_mask,1/std::sqrt(float(dim)),0,0);
    ggml_flash_attn_ext_add_sinks(direct,sinks);
    ggml_set_output(direct);ggml_set_output(direct_keys);ggml_set_output(direct_mask);
    auto * compact_keys=keys;
    auto * compact_mask=mask;
    // The graph and test use the same production packing operation. Padding
    // itself has a separate host mask input to avoid sharing a mask oracle.
    if(packed_rows>bound)keys=ggml_pad(ctx,keys,0,packed_rows-bound,0,0);
    keys=ggml_cast(ctx,keys,GGML_TYPE_F16);
    auto * pm=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,packed_rows,1);
    auto * old=ggml_flash_attn_ext(ctx,q,full,full,fm,1/std::sqrt(float(dim)),0,0);
    ggml_flash_attn_ext_set_n_kv_max(old,bound);
    ggml_flash_attn_ext_add_sinks(old,sinks);
    auto * out=ggml_flash_attn_ext(ctx,q,keys,keys,pm,1/std::sqrt(float(dim)),0,0);
    ggml_flash_attn_ext_add_sinks(out,sinks);
    ggml_set_output(old);ggml_set_output(out);
    ggml_set_output(compact_keys);ggml_set_output(compact_mask);
    auto * graph=ggml_new_graph(ctx);
    ggml_build_forward_expand(graph,old);ggml_build_forward_expand(graph,out);
    ggml_build_forward_expand(graph,compact_mask);
    ggml_build_forward_expand(graph,direct);
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);
    if(!buffer){ggml_free(ctx);return false;}
    auto upload=[&](ggml_tensor * t,float phase) {
        std::vector<float> f(ggml_nelements(t));
        for(size_t i=0;i<f.size();++i)f[i]=.4f*std::sin(i*.013f+phase);
        if(t->type==GGML_TYPE_F32)ggml_backend_tensor_set(t,f.data(),0,ggml_nbytes(t));
        else {
            std::vector<uint8_t> bytes(ggml_nbytes(t));
            ggml_quantize_chunk(t->type,f.data(),bytes.data(),0,ggml_nrows(t),t->ne[0],nullptr);
            ggml_backend_tensor_set(t,bytes.data(),0,bytes.size());
            ggml_get_type_traits(t->type)->to_float(bytes.data(),f.data(),f.size());
        }
        return f;
    };
    const auto raw_values=upload(raw,.3f);
    const auto comp_values=comp_rows?upload(comp,.7f):std::vector<float>();
    upload(q,.1f);upload(sinks,.9f);
    bool ok=true;float maximum=0;
    const auto inf=ggml_fp32_to_fp16(-INFINITY),zero=ggml_fp32_to_fp16(0.f);
    for(int pos:{-1,0,1,7,127,128,255,1029,4351,4352,4353,17777}) {
        const int count=std::min(pos+1,swa);
        std::vector<int32_t> raw_ids(swa,0),sel(selected);
        for(int i=0;i<count;++i)raw_ids[i]=(pos-count+1+i)%window;
        std::sort(raw_ids.begin(),raw_ids.begin()+count);
        for(int i=0;i<selected;++i)sel[i]=(i*37+pos*13+comp_rows)%comp_rows;
        std::sort(sel.begin(),sel.end());
        std::vector<ggml_fp16_t> raw_mask(swa,inf),sel_mask(selected,inf),full_mask(full_rows,inf),packed_mask(packed_rows,inf);
        std::vector<float> expected;
        auto append=[&](const std::vector<float> & values,int row) {
            expected.insert(expected.end(),values.begin()+size_t(row)*dim,values.begin()+size_t(row+1)*dim);
        };
        for(int i=0;i<count;++i){if(pos==255 && i%5==0)continue;raw_mask[i]=zero;full_mask[raw_ids[i]]=zero;append(raw_values,raw_ids[i]);}
        // Alternate compression ratios; leave genuinely invisible selected
        // rows present during the early window, as in the native indexer.
        const int visible=std::min(comp_rows,pos<0?0:(pos+1)/(1+pos%2));
        for(int i=0;i<selected;++i)if(sel[i]<visible && !(pos==1029 && i%7==0)){sel_mask[i]=zero;full_mask[window+sel[i]]=zero;append(comp_values,sel[i]);}
        const int finite=expected.size()/dim;
        std::fill(packed_mask.begin(),packed_mask.begin()+finite,zero);
        ggml_backend_tensor_set(ri,raw_ids.data(),0,ggml_nbytes(ri));
        ggml_backend_tensor_set(rm,raw_mask.data(),0,ggml_nbytes(rm));
        if(comp_rows){ggml_backend_tensor_set(si,sel.data(),0,ggml_nbytes(si));if(sm_float){std::vector<float> mf(selected);for(int i=0;i<selected;++i)mf[i]=ggml_fp16_to_fp32(sel_mask[i]);ggml_backend_tensor_set(sm,mf.data(),0,ggml_nbytes(sm));}
            else ggml_backend_tensor_set(sm,sel_mask.data(),0,ggml_nbytes(sm));}
        ggml_backend_tensor_set(fm,full_mask.data(),0,ggml_nbytes(fm));
        ggml_backend_tensor_set(pm,packed_mask.data(),0,ggml_nbytes(pm));
        ok &= ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS;
        std::vector<float> packed(ggml_nelements(compact_keys));
        std::vector<ggml_fp16_t> actual_mask(bound);
        ggml_backend_tensor_get(compact_keys,packed.data(),0,ggml_nbytes(compact_keys));
        ggml_backend_tensor_get(compact_mask,actual_mask.data(),0,ggml_nbytes(compact_mask));
        ok &= std::equal(expected.begin(),expected.end(),packed.begin());
        ok &= std::equal(actual_mask.begin(),actual_mask.end(),packed_mask.begin());
        std::vector<ggml_fp16_t> dk(ggml_nelements(direct_keys)),dm(packed_rows);
        ggml_backend_tensor_get(direct_keys,dk.data(),0,ggml_nbytes(direct_keys));
        ggml_backend_tensor_get(direct_mask,dm.data(),0,ggml_nbytes(direct_mask));
        ok &= dm==packed_mask;
        for(size_t i=0;i<dk.size();++i)ok &= dk[i]==ggml_fp32_to_fp16(i<expected.size()?expected[i]:0.f);
        std::vector<float> a(ggml_nelements(old)),b(a.size()),d(a.size());
        ggml_backend_tensor_get(old,a.data(),0,ggml_nbytes(old));
        ggml_backend_tensor_get(out,b.data(),0,ggml_nbytes(out));
        ggml_backend_tensor_get(direct,d.data(),0,ggml_nbytes(direct));
        ok &= !std::memcmp(b.data(),d.data(),b.size()*sizeof(float));
        for(size_t i=0;i<a.size();++i){ok &= std::isfinite(a[i]) && std::isfinite(b[i]);maximum=std::max(maximum,std::abs(a[i]-b[i]));}
    }
    ok &= maximum<1e-6f;
    std::printf("V4.1 compact KV raw=%d compressed=%d: original FP32 and direct FP16 rows/masks exact, direct attention bit-exact; attention max_error=%g %s\n",window,comp_rows,maximum,ok?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}
int main() {
    ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;
    if(std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu))),"ROCm")){ggml_backend_free(gpu);return 77;}
    const bool ok=check(gpu,0) & check(gpu,8960) & check(gpu,17920) & check(gpu,8960,true) & check(gpu,17920,true);
    ggml_backend_free(gpu);return ok?0:1;
}
