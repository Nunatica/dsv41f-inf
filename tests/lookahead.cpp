#include "moe.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Several Q3 descriptors must coexist: a one-layer test never reaches a
// future descriptor. Vary routes through ring wrap, resize and invalidation.
int main(int argc,char ** argv) {
    const bool zero=argc>1 && !std::strcmp(argv[1],"zero-cache");
    const bool early=std::getenv("DSV4F_EXPERT_LOOKAHEAD") && std::atoi(std::getenv("DSV4F_EXPERT_LOOKAHEAD"))>=3;
    const bool protect=std::getenv("DSV4F_EXPERT_LOOKAHEAD") && std::atoi(std::getenv("DSV4F_EXPERT_LOOKAHEAD"))==5;
    const int prediction_count=protect?16:8;
    ggml_type type=GGML_TYPE_Q3_K;
    if(argc>2 && !std::strcmp(argv[2],"q2"))type=GGML_TYPE_Q2_0;
    if(argc>2 && !std::strcmp(argv[2],"ptq1"))type=GGML_TYPE_PTQ1_0;
    setenv("DSV4F_GPU_SERVICE","1",1);setenv("DSV4F_GPU_WAIT","callback",0);
    setenv("DSV4F_EXPERT_CACHE_MIB",zero?"0":early?"4":"1",1);
    setenv("DSV4F_V41_CACHE_Q3_MIB",zero?"0":early?"4":"1",1);
    setenv("DSV4F_CACHE_POLICY","aged",1);setenv("DSV4F_CACHE_REPLACE_RATIO","1",1);
    ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;
    auto * cpu=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(cpu,4);
    constexpr int ne=512,nf=512,nex=12,nk=2,nl=3;
    auto * wc=ggml_init({2<<20,nullptr,true});
    struct tier {ggml_tensor * g,*u,*d;};std::vector<tier> tiers;
    for(int l=0;l<nl;++l) {
        auto * g=ggml_new_tensor_3d(wc,type,ne,nf,nex);
        tiers.push_back({g,ggml_dup_tensor(wc,g),ggml_new_tensor_3d(wc,type,nf,ne,nex)});
    }
    auto * wb=ggml_backend_alloc_ctx_tensors(wc,cpu);
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(352);std::normal_distribution<float> normal(0,.03f);
    for(auto & tier:tiers)for(auto * t:{tier.g,tier.u,tier.d}) {
        if(t->type==GGML_TYPE_PTQ1_0) {
            // PTQ has a decoder but no generic float quantizer. Generate packed
            // blocks as the existing V4.1 parity fixtures do.
            auto * bytes=(unsigned char *)t->data;const auto scale=ggml_fp32_to_fp16(.03f);
            for(size_t i=0;i<ggml_nbytes(t)/28;++i){for(int j=0;j<26;++j)bytes[i*28+j]=rng()%256;std::memcpy(bytes+i*28+26,&scale,2);}
        } else {
            std::vector<float> f(ggml_nelements(t));for(auto & v:f)v=normal(rng);
            ggml_quantize_chunk(t->type,f.data(),t->data,0,t->ne[1]*t->ne[2],t->ne[0],nullptr);
        }
    }
    struct model {ggml_context * ctx;ggml_cgraph * graph;ggml_tensor * x,*ids,*w,*out,*pred=nullptr,*map=nullptr;};
    model models[2]{};ggml_backend_buffer_t ref_buffer=nullptr;
    ggml_backend_t backends[]={gpu,cpu};
    auto * sched=ggml_backend_sched_new(backends,nullptr,2,2048,false,true);
    for(int mode=0;mode<2;++mode) {
        auto & m=models[mode];m.ctx=ggml_init({4<<20,nullptr,true});
        m.x=ggml_new_tensor_2d(m.ctx,GGML_TYPE_F32,ne,1);
        m.ids=ggml_new_tensor_2d(m.ctx,GGML_TYPE_I32,nk,1);
        m.w=ggml_new_tensor_3d(m.ctx,GGML_TYPE_F32,1,nk,1);
        m.out=ggml_scale(m.ctx,m.x,.001f);
        m.graph=ggml_new_graph_custom(m.ctx,2048,false);
        if(mode && early){
            m.pred=ggml_new_tensor_2d(m.ctx,GGML_TYPE_I32,prediction_count,1);
            m.map=ggml_new_tensor_2d(m.ctx,GGML_TYPE_I32,1,nex+4);
            ggml_set_input(m.pred);ggml_set_input(m.map);
        }
        for(int l=0;l<nl;++l) {
            auto & t=tiers[l];
            if(mode && early && l+1<nl){auto & next=tiers[l+1];
                ggml_build_forward_expand(m.graph,dsv4f_expert_prefetch(m.ctx,m.x,m.pred,m.map,next.g,next.u,next.d,l+1));
            }
            if(mode)m.out=dsv4f_moe_join(m.ctx,dsv4f_moe(m.ctx,m.x,m.ids,m.w,t.g,t.u,t.d,1.25f),m.out);
            else {
                auto * x3=ggml_reshape_3d(m.ctx,m.x,ne,1,1);
                auto * h=ggml_swiglu_clamp(m.ctx,ggml_mul_mat_id(m.ctx,t.g,x3,m.ids),ggml_mul_mat_id(m.ctx,t.u,x3,m.ids),1.25f);
                auto * v=ggml_mul(m.ctx,ggml_mul_mat_id(m.ctx,t.d,h,m.ids),m.w);
                for(int k=0;k<nk;++k)m.out=ggml_add(m.ctx,m.out,ggml_view_2d(m.ctx,v,ne,1,v->nb[2],k*v->nb[1]));
            }
            ggml_build_forward_expand(m.graph,m.out);
        }
        if(mode){for(auto * t:{m.x,m.ids,m.w})ggml_set_input(t);ggml_set_output(m.out);
            if(!ggml_backend_sched_alloc_graph(sched,m.graph))return 1;
        }else ref_buffer=ggml_backend_alloc_ctx_tensors(m.ctx,cpu);
    }
    bool ok=true;double worst=0;std::vector<float> x(ne),a(ne),b(ne);float w[nk]={.4f,.6f};
    for(int step=0;step<96;++step) {
        for(auto & v:x)v=normal(rng)*8;
        int32_t ids[nk]={int32_t((step/8)%nex),int32_t((step/8+3)%nex)};
        if(step%11==0)w[1]=0;else w[1]=.6f;
        for(int mode=0;mode<2;++mode) {
            auto & m=models[mode];ggml_backend_tensor_set(m.x,x.data(),0,ne*4);
            ggml_backend_tensor_set(m.ids,ids,0,nk*4);ggml_backend_tensor_set(m.w,w,0,nk*4);
            if(m.pred){
                // Reverse global-to-local mapping, with absent-format and invalid
                // predictions mixed with correct and deliberately wrong experts.
                int32_t mapping[nex+4];std::vector<int32_t> prediction(prediction_count);
                for(int i=0;i<nex+4;++i)mapping[i]=i>=2 && i<nex+2?nex+1-i:-1;
                for(int k=0;k<prediction_count;++k)prediction[k]=protect?(k+step)%(nex+4):nex+1-(step%3?((ids[0]+k)%nex):((ids[0]+k+5)%nex));
                if(!protect && step%7==0){prediction[6]=-1;prediction[7]=nex+3;}
                ggml_backend_tensor_set(m.pred,prediction.data(),0,prediction.size()*sizeof(int32_t));
                ggml_backend_tensor_set(m.map,mapping,0,sizeof(mapping));
            }
            ok&=(mode?ggml_backend_sched_graph_compute(sched,m.graph):ggml_backend_graph_compute(cpu,m.graph))==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(m.out,mode?b.data():a.data(),0,ne*4);
        }
        double error=0,norm=0;for(int i=0;i<ne;++i){error+=double(a[i]-b[i])*(a[i]-b[i]);norm+=double(a[i])*a[i];ok&=std::isfinite(b[i]);}
        const double rel=std::sqrt(error/std::max(norm,1e-30));worst=std::max(worst,rel);ok&=rel<.05;
        if(step==31 || step==63){ggml_dsv4f_experts.gpu_resize(false);ggml_dsv4f_experts.gpu_resize(true);}
    }
    // Drain before reading counters or freeing any source; reset and rerun
    // through source invalidation is exercised by the native graph fixture too.
    ggml_dsv4f_experts.gpu_resize(true);uint64_t routes,hits,admissions;
    ggml_dsv4f_experts.gpu_statistics(&routes,&hits,&admissions);
    ok&=routes>0 && (zero?(hits==0 && admissions==0):admissions>0);
    for(auto & t:tiers)ggml_dsv4f_experts.invalidate(t.g->data,ggml_nbytes(t.g));
    ggml_backend_sched_free(sched);ggml_backend_buffer_free(ref_buffer);
    for(auto & m:models)ggml_free(m.ctx);
    ggml_backend_buffer_free(wb);ggml_free(wc);ggml_backend_free(gpu);ggml_backend_free(cpu);
    std::printf("lookahead format=%s early=%d zero=%d replays=96 worst_rmse=%g %s\n",ggml_type_name(type),early,zero,worst,ok?"PASS":"FAIL");return ok?0:1;
}
