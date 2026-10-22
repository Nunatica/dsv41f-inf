#include "moe.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

static ggml_dsv4f_cpu_fn original_cpu;
static std::atomic<int> delays{0};
static void delayed_cpu(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,const float * x,const int32_t * ids,const float * w,int nk,uint64_t hits,float limit,float * out) {
    if(delays.fetch_sub(1)>0)std::this_thread::sleep_for(std::chrono::milliseconds(2));
    original_cpu(g,u,d,x,ids,w,nk,hits,limit,out);
}

static bool check(ggml_backend_t gpu,ggml_backend_t cpu,int nt) {
    constexpr int ne=256,nf=256,nex=8,nk=3;
    auto * wc=ggml_init({2<<20,nullptr,true});
    struct tier {ggml_tensor * g,*u,*d;};
    std::vector<tier> tiers;
    for(auto type:{GGML_TYPE_Q3_K,GGML_TYPE_Q2_0,GGML_TYPE_PTQ1_0}) {
        auto * g=ggml_new_tensor_3d(wc,type,ne,nf,nex);
        tiers.push_back({g,ggml_dup_tensor(wc,g),ggml_new_tensor_3d(wc,type,nf,ne,nex)});
    }
    auto * wb=ggml_backend_alloc_ctx_tensors(wc,cpu);
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(73401);std::normal_distribution<float> normal(0,.03f);
    for(auto & tier:tiers)for(auto * t:{tier.g,tier.u,tier.d}) {
        if(t->type==GGML_TYPE_PTQ1_0) {
            auto * p=(uint8_t*)t->data;const auto scale=ggml_fp32_to_fp16(.03f);
            for(size_t i=0;i<ggml_nbytes(t)/28;++i){for(int j=0;j<26;++j)p[i*28+j]=rng()%256;std::memcpy(p+i*28+26,&scale,2);}
        } else {
            std::vector<float> f(ggml_nelements(t));for(auto & x:f)x=normal(rng);
            ggml_quantize_chunk(t->type,f.data(),t->data,0,t->ne[1]*t->ne[2],t->ne[0],nullptr);
        }
    }
    struct model {
        ggml_context * ctx;ggml_backend_sched_t sched;ggml_cgraph * graph;
        ggml_tensor * x,*ids,*prob,*out;
    };
    std::array<model,3> models{};
    ggml_backend_t backends[]={gpu,cpu};
    for(int mode=0;mode<3;++mode) {
        auto & m=models[mode];m.ctx=ggml_init({4<<20,nullptr,true});
        m.graph=ggml_new_graph_custom(m.ctx,1024,false);
        m.x=ggml_new_tensor_2d(m.ctx,GGML_TYPE_F32,ne,nt);
        m.ids=ggml_new_tensor_2d(m.ctx,GGML_TYPE_I32,nk,nt);
        m.prob=ggml_new_tensor_3d(m.ctx,GGML_TYPE_F32,1,nk,nt);
        m.out=ggml_scale(m.ctx,m.x,.001f);
        std::vector<ggml_tensor*> pending;
        for(auto & t:tiers) {
            auto * routed=dsv4f_moe(m.ctx,m.x,m.ids,m.prob,t.g,t.u,t.d,1.25f);
            if(mode){if(mode==2)ggml_build_forward_expand(m.graph,dsv4f_moe_prepare(m.ctx,routed));else dsv4f_moe_submit(m.graph,routed);pending.push_back(routed);}
            else m.out=&t==&tiers.front()?dsv4f_moe_join(m.ctx,routed,m.out):ggml_add(m.ctx,m.out,routed);
        }
        if(mode==2)for(auto * routed:pending)dsv4f_moe_submit(m.graph,routed);
        for(auto * routed:pending)m.out=dsv4f_moe_join(m.ctx,routed,m.out);
        ggml_build_forward_expand(m.graph,m.out);
        if(mode==2) {
            int prepared=0;
            for(int i=0;i<ggml_graph_n_nodes(m.graph);++i) {
                auto * n=ggml_graph_nodes(m.graph)[i];if(n->op!=GGML_OP_CUSTOM)continue;
                if(n->op_params[9]==GGML_DSV4F_PREPARE_TAG)++prepared;
                if(n->op_params[9]==GGML_DSV4F_MOE_TAG)GGML_ASSERT(prepared==int(tiers.size()));
            }
        }
        for(auto * input:{m.x,m.ids,m.prob})ggml_set_input(input);
        ggml_set_output(m.out);
        m.sched=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);
        if(!ggml_backend_sched_alloc_graph(m.sched,m.graph))return false;
    }
    original_cpu=ggml_dsv4f_cpu_experts;
    GGML_ASSERT(original_cpu && original_cpu!=delayed_cpu);
    ggml_dsv4f_cpu_experts=delayed_cpu;
    bool ok=true;int exact=0,route_exact=0;
    std::vector<float> input(ne*nt),prob(nk*nt),expected(ne*nt),actual(ne*nt);
    std::vector<int32_t> ids(nk*nt);
    for(int step=-12;step<32;++step) {
        for(auto & v:input)v=normal(rng)*8;
        for(int i=0;i<nk*nt;++i){ids[i]=(step+16+i*3)%nex;prob[i]=(i+step+16)%5==0?0.f:(i%3+1)/6.f;}
        for(int mode=0;mode<3;++mode) {
            if(step==0 || step==9 || step==17)delays=8;
            auto & m=models[mode];
            ggml_backend_tensor_set(m.x,input.data(),0,input.size()*4);
            ggml_backend_tensor_set(m.ids,ids.data(),0,ids.size()*4);
            ggml_backend_tensor_set(m.prob,prob.data(),0,prob.size()*4);
            ok&=ggml_backend_sched_graph_compute(m.sched,m.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(m.out,mode?actual.data():expected.data(),0,actual.size()*4);
            ggml_dsv4f_experts.gpu_resize(true); // Drain admission before comparing stable cache paths.
            if(step>=0 && mode) {
                const bool same=std::memcmp(expected.data(),actual.data(),actual.size()*4)==0;
                if(mode==1)exact+=same;else route_exact+=same;
                ok&=same;for(float v:actual)ok&=std::isfinite(v);
            }
        }
    }
    std::printf("V4.1 all-format deferred join nt=%d exact=%d/32 route_first=%d/32 %s\n",nt,exact,route_exact,ok?"PASS":"FAIL");
    ggml_dsv4f_cpu_experts=original_cpu;
    for(auto & m:models){ggml_backend_sched_free(m.sched);ggml_free(m.ctx);}
    for(auto & t:tiers)ggml_dsv4f_experts.invalidate(t.g->data,ggml_nbytes(t.g));
    ggml_backend_buffer_free(wb);ggml_free(wc);return ok;
}
int main(int argc,char**) {
    setenv("DSV4F_GPU_SERVICE","1",1);setenv("DSV4F_GPU_WAIT","callback",1);
    setenv("DSV4F_EXPERT_CACHE_MIB",argc>1?"0":"16",1);
    setenv("DSV4F_VRAM_HEADROOM_MIB","256",1);setenv("DSV4F_VRAM_GUARD_MIB","256",1);
    ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;auto * cpu=ggml_backend_cpu_init();
    bool ok=true;for(int nt:{1,2,4})ok&=check(gpu,cpu,nt);
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
