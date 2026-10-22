#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static bool check(ggml_backend_t gpu,ggml_backend_t cpu,int ne,int nt,bool comb,bool strided) {
    struct state {ggml_context * ctx;ggml_backend_sched_t sched;ggml_cgraph * graph;ggml_tensor * r,*x,*packed,*post,*pre,*late;};
    state modes[2];
    const int nr=ne+(strided?16:0),nx=ne+(strided?8:0);
    const float scale=nt==2?-.75f:1.f;
    for(int k=0;k<2;++k) {
        auto & s=modes[k];s.ctx=ggml_init({2<<20,nullptr,true});
        ggml_backend_t backends[]={gpu,cpu};ggml_backend_buffer_type_t bufts[]={ggml_backend_get_default_buffer_type(gpu),ggml_backend_get_default_buffer_type(cpu)};
        s.sched=ggml_backend_sched_new(backends,bufts,2,256,false,true);
        s.r=ggml_new_tensor_3d(s.ctx,GGML_TYPE_F32,nr,4,nt);
        s.x=ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,nx,nt);
        s.packed=ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,24,nt);
        for(auto * t:{s.r,s.x,s.packed}) {ggml_set_input(t);ggml_backend_sched_set_tensor_backend(s.sched,t,gpu);}
        auto * r=ggml_view_3d(s.ctx,s.r,ne,4,nt,nr*4,nr*16,0);
        auto * x=ggml_view_2d(s.ctx,s.x,ne,nt,nx*4,0);
        auto * post=ggml_view_2d(s.ctx,s.packed,4,nt,24*4,4*4);
        auto * pre=ggml_view_2d(s.ctx,s.packed,4,nt,24*4,0);
        auto * c=ggml_view_3d(s.ctx,s.packed,4,4,nt,4*4,24*4,8*4);
        if(strided)c=ggml_permute(s.ctx,c,1,0,2,3);
        s.post=ggml_dsv4_hc_post(s.ctx,x,r,post,comb?c:nullptr);
        // A no-op view prevents the control's post/pre fusion without adding
        // a GPU kernel. The candidate has the direct graph edge used by V4.1.
        s.pre=ggml_dsv4_hc_pre(s.ctx,k?s.post:ggml_view_3d(s.ctx,s.post,ne,4,nt,s.post->nb[1],s.post->nb[2],0),pre);
        std::memcpy(s.pre->op_params,&scale,sizeof(scale));
        s.late=ggml_scale(s.ctx,s.post,.25f);
        ggml_set_output(s.pre);ggml_set_output(s.post);ggml_set_output(s.late);
        s.graph=ggml_new_graph_custom(s.ctx,256,false);
        ggml_build_forward_expand(s.graph,s.pre);ggml_build_forward_expand(s.graph,s.late);
        if(!ggml_backend_sched_alloc_graph(s.sched,s.graph))std::abort();
    }
    std::vector<float> r(nr*4*nt),x(nx*nt),packed(24*nt),expected(4*ne*nt),actual(expected.size());
    bool ok=true;int exact=0;
    for(int step=0;step<16;++step) {
        const float a=step==0?0.f:step%3==0?1e-7f:step%3==1?1.f:100.f;
        for(size_t i=0;i<r.size();++i)r[i]=a*std::sin(.013f*i+.37f*step);
        for(size_t i=0;i<x.size();++i)x[i]=a*std::cos(.017f*i+.13f*step);
        for(size_t i=0;i<packed.size();++i)packed[i]=.4f*std::cos(.11f*i+step*.3f);
        for(auto & s:modes) {
            ggml_backend_tensor_set(s.r,r.data(),0,r.size()*4);ggml_backend_tensor_set(s.x,x.data(),0,x.size()*4);ggml_backend_tensor_set(s.packed,packed.data(),0,packed.size()*4);
            ok &= ggml_backend_sched_graph_compute(s.sched,s.graph)==GGML_STATUS_SUCCESS;
        }
        bool same=true;
        for(int output=0;output<3;++output) {
            auto * ref=output==0?modes[0].pre:output==1?modes[0].post:modes[0].late;
            auto * got=output==0?modes[1].pre:output==1?modes[1].post:modes[1].late;
            const size_t bytes=ggml_nelements(ref)*4;
            ggml_backend_tensor_get(ref,expected.data(),0,bytes);ggml_backend_tensor_get(got,actual.data(),0,bytes);
            same &= !std::memcmp(actual.data(),expected.data(),bytes);
            for(size_t i=0;i<bytes/4;++i)ok &= std::isfinite(actual[i]);
        }
        exact+=same;ok &= same;
    }
    std::printf("HC post/pre ne=%d nt=%d comb=%d stride=%d post+pre+late exact=%d/16 %s\n",ne,nt,comb,strided,exact,ok?"PASS":"FAIL");
    for(auto & s:modes){ggml_backend_sched_free(s.sched);ggml_free(s.ctx);}return ok;
}
int main() {
    setenv("DSV4F_V41_HC_POST_PRE_FUSED","1",1);
    ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    if(!gpu || !cpu)return 77;
    bool ok=true;for(int ne:{256,5120})for(int nt:{1,2,4,7})for(bool comb:{false,true})for(bool stride:{false,true})ok &= check(gpu,cpu,ne,nt,comb,stride);
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
