#include "moe.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static bool check(ggml_backend_t gpu,ggml_backend_t cpu,int nk,int nt,bool strided,bool constant) {
    constexpr int ne=512;
    struct state {ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_cgraph * graph;
                  ggml_tensor * mapping_storage,*mapping,*ids_storage,*prob_storage,*ids,*prob,*lid,*weight;};
    state states[3]{};
    const int id_pitch=nk+(strided?5:0),prob_pitch=strided?2*nk+3:nk;
    for(int mode=0;mode<3;++mode) {
        auto & s=states[mode];s.ctx=ggml_init({2<<20,nullptr,true});
        s.mapping_storage=ggml_new_tensor_2d(s.ctx,GGML_TYPE_I32,strided?2:1,ne);
        s.mapping=ggml_view_2d(s.ctx,s.mapping_storage,1,ne,(strided?2:1)*4,0);
        s.ids_storage=ggml_new_tensor_2d(s.ctx,GGML_TYPE_I32,id_pitch,nt);
        s.prob_storage=ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,prob_pitch,nt);
        s.ids=ggml_view_2d(s.ctx,s.ids_storage,nk,nt,id_pitch*4,0);
        s.prob=ggml_view_3d(s.ctx,s.prob_storage,1,nk,nt,(strided?2:1)*4,prob_pitch*4,0);
        if(mode)dsv4f_route_plan(s.ctx,s.mapping,s.ids,s.prob,&s.lid,&s.weight);
        else {
            auto * local=ggml_get_rows(s.ctx,ggml_repeat_4d(s.ctx,ggml_cast(s.ctx,s.mapping,GGML_TYPE_F32),1,ne,nt,1),s.ids);
            s.weight=ggml_mul(s.ctx,s.prob,ggml_step(s.ctx,ggml_scale_bias(s.ctx,local,1.f,.5f)));
            s.lid=ggml_reshape_2d(s.ctx,ggml_cast(s.ctx,ggml_clamp(s.ctx,local,0.f,INFINITY),GGML_TYPE_I32),nk,nt);
        }
        s.graph=ggml_new_graph_custom(s.ctx,1024,false);
        ggml_build_forward_expand(s.graph,s.lid);ggml_build_forward_expand(s.graph,s.weight);
        if(!constant)ggml_set_input(s.mapping_storage);
        ggml_set_input(s.ids_storage);ggml_set_input(s.prob_storage);
        ggml_set_output(s.lid);ggml_set_output(s.weight);
        s.buffer=ggml_backend_alloc_ctx_tensors(s.ctx,mode==2?cpu:gpu);
        if(!s.buffer)return false;
    }
    std::mt19937 rng(9361+nk+nt);
    std::vector<int32_t> mapping(ne),ids(id_pitch*nt),reference_ids(nk*nt),actual_ids(nk*nt);
    std::vector<int32_t> mapping_storage(ne*(strided?2:1),-99);
    std::vector<float> prob(prob_pitch*nt),reference_prob(nk*nt),actual_prob(nk*nt);
    bool ok=true;
    const float values[]={0.f,-0.f,.125f,-.25f,1.f,.03f};
    for(int rep=0;rep<12;++rep) {
        for(int i=0;i<ne;++i)mapping[i]=i%3?int(rng()%512):-1-int(rng()%17);
        mapping[0]=-1;mapping[1]=0;mapping[2]=511;
        for(int i=0;i<ne;++i)mapping_storage[i*(strided?2:1)]=mapping[i];
        for(int t=0;t<nt;++t)for(int k=0;k<nk;++k)ids[t*id_pitch+k]=k<3?k:int(rng()%ne);
        for(size_t i=0;i<prob.size();++i)prob[i]=values[(i+rep)%6];
        for(int mode=0;mode<3;++mode) {
            auto & s=states[mode];
            ggml_backend_tensor_set(s.mapping_storage,mapping_storage.data(),0,mapping_storage.size()*4);
            ggml_backend_tensor_set(s.ids_storage,ids.data(),0,ids.size()*4);
            ggml_backend_tensor_set(s.prob_storage,prob.data(),0,prob.size()*4);
            ok&=ggml_backend_graph_compute(mode==2?cpu:gpu,s.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(s.lid,actual_ids.data(),0,actual_ids.size()*4);
            ggml_backend_tensor_get(s.weight,actual_prob.data(),0,actual_prob.size()*4);
            if(!mode){reference_ids=actual_ids;reference_prob=actual_prob;}
            else {
                ok&=std::memcmp(reference_ids.data(),actual_ids.data(),actual_ids.size()*4)==0;
                ok&=std::memcmp(reference_prob.data(),actual_prob.data(),actual_prob.size()*4)==0;
            }
            for(int t=0;t<nt;++t)for(int k=0;k<nk;++k) {
                const int local=mapping[ids[t*id_pitch+k]],i=t*nk+k;
                const float expected=prob[t*prob_pitch+k*(strided?2:1)]*(local>=0?1.f:0.f);
                ok&=actual_ids[i]==std::max(local,0);
                ok&=std::memcmp(&actual_prob[i],&expected,4)==0;
            }
        }
    }
    double times[2]{};
    for(int mode=0;mode<2;++mode) {
        std::vector<double> elapsed;
        for(int rep=-3;rep<25;++rep) {
            const auto start=std::chrono::steady_clock::now();
            ok&=ggml_backend_graph_compute(gpu,states[mode].graph)==GGML_STATUS_SUCCESS;
            ggml_backend_synchronize(gpu);
            if(rep>=0)elapsed.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
        }
        std::sort(elapsed.begin(),elapsed.end());times[mode]=elapsed[12];
    }
    std::printf("route plan nk=%d nt=%d strided=%d constant=%d bitwise parity %s old_us=%.3f fused_us=%.3f\n",
                nk,nt,int(strided),int(constant),ok?"PASS":"FAIL",times[0],times[1]);
    for(auto & s:states){ggml_backend_buffer_free(s.buffer);ggml_free(s.ctx);}
    return ok;
}
static bool check_router(ggml_backend_t gpu,ggml_backend_t cpu,int ne,int nk,int nt,bool norm,bool strided) {
    struct state {ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_cgraph * graph;
                  ggml_tensor * storage,*x,*bias,*ids,*weights;};
    state states[3]{};const int pitch=ne+(strided?7:0);constexpr float scale=1.7f;
    for(int mode=0;mode<3;++mode) {
        auto & a=states[mode];a.ctx=ggml_init({2<<20,nullptr,true});
        a.storage=ggml_new_tensor_2d(a.ctx,GGML_TYPE_F32,pitch,nt);
        a.x=ggml_view_2d(a.ctx,a.storage,ne,nt,pitch*4,0);
        a.bias=ggml_new_tensor_1d(a.ctx,GGML_TYPE_F32,ne);
        if(mode)dsv4f_router(a.ctx,a.x,a.bias,nk,norm,scale,&a.ids,&a.weights);
        else {
            auto * scores=ggml_sqrt(a.ctx,ggml_softplus(a.ctx,a.x));
            a.ids=ggml_cont(a.ctx,ggml_top_k(a.ctx,ggml_add(a.ctx,scores,a.bias),nk));
            a.weights=ggml_reshape_3d(a.ctx,ggml_get_rows(a.ctx,ggml_reshape_3d(a.ctx,scores,1,ne,nt),a.ids),1,nk,nt);
            if(norm) {auto * w=ggml_reshape_2d(a.ctx,a.weights,nk,nt);a.weights=ggml_reshape_3d(a.ctx,ggml_div(a.ctx,w,ggml_scale_bias(a.ctx,ggml_sum_rows(a.ctx,w),1.f,1e-20f)),1,nk,nt);}
            a.weights=ggml_scale(a.ctx,a.weights,scale);
        }
        ggml_set_input(a.storage);ggml_set_input(a.bias);ggml_set_output(a.ids);ggml_set_output(a.weights);
        a.graph=ggml_new_graph_custom(a.ctx,1024,false);ggml_build_forward_expand(a.graph,a.weights);ggml_build_forward_expand(a.graph,a.ids);
        a.buffer=ggml_backend_alloc_ctx_tensors(a.ctx,mode==2?cpu:gpu);if(!a.buffer)return false;
    }
    std::mt19937 rng(173+ne+nk+nt);std::uniform_real_distribution<float> dist(-25.f,40.f);
    std::vector<float> x(pitch*nt),bias(ne),ref(nk*nt),actual(nk*nt);std::vector<int32_t> ri(nk*nt),ai(nk*nt);
    bool ok=true;double max_error=0;
    for(int rep=0;rep<12;++rep) {
        for(auto & v:x)v=rep==0?0.f:rep==1?-100.f:rep==2?1000.f:dist(rng);
        for(auto & v:bias)v=rep<3?0.f:dist(rng)*.1f;
        for(int mode=0;mode<3;++mode) {
            auto & a=states[mode];ggml_backend_tensor_set(a.storage,x.data(),0,x.size()*4);ggml_backend_tensor_set(a.bias,bias.data(),0,bias.size()*4);
            ok&=ggml_backend_graph_compute(mode==2?cpu:gpu,a.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(a.ids,ai.data(),0,ai.size()*4);ggml_backend_tensor_get(a.weights,actual.data(),0,actual.size()*4);
            if(!mode){ri=ai;ref=actual;}
            else {
                ok&=ri==ai;
                for(size_t k=0;k<ref.size();++k){max_error=std::max(max_error,double(std::abs(ref[k]-actual[k])));ok&=std::isfinite(actual[k]) && std::abs(ref[k]-actual[k])<=2e-6f*std::max(1.f,std::abs(ref[k]));}
                if(mode==1 && std::memcmp(ref.data(),actual.data(),ref.size()*4)!=0) {
                    std::fprintf(stderr,"router GPU bitwise mismatch ne=%d nk=%d nt=%d norm=%d strided=%d rep=%d max_error=%g\n",ne,nk,nt,int(norm),int(strided),rep,max_error);ok=false;
                }
            }
        }
    }
    double times[2]{};
    if(ne==384 && nk==6 && nt==1 && norm && !strided)for(int mode=0;mode<2;++mode) {
        std::vector<double> samples;
        for(int rep=-5;rep<51;++rep){auto start=std::chrono::steady_clock::now();ggml_backend_graph_compute(gpu,states[mode].graph);ggml_backend_synchronize(gpu);if(rep>=0)samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());}
        std::sort(samples.begin(),samples.end());times[mode]=samples[samples.size()/2];
    }
    std::printf("router ne=%d nk=%d nt=%d norm=%d strided=%d GPU-bitwise CPU-tolerance %s max_error=%g old_us=%.3f fused_us=%.3f\n",ne,nk,nt,int(norm),int(strided),ok?"PASS":"FAIL",max_error,times[0],times[1]);
    for(auto & a:states){ggml_backend_buffer_free(a.buffer);ggml_free(a.ctx);}return ok;
}
int main() {
    setenv("GGML_CUDA_MOE_CACHE","0",1);ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 77;
    auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);if(!cpu)return 1;
    bool ok=true;
    for(int nk:{1,8,63})for(int nt:{1,2,4})for(bool strided:{false,true})for(bool constant:{false,true})
        ok&=check(gpu,cpu,nk,nt,strided,constant);
    for(int ne:{6,32,383,384,512})for(int nk:{1,6,32})if(nk<=ne)for(int nt:{1,2,4})for(bool norm:{false,true})for(bool strided:{false,true})
        ok&=check_router(gpu,cpu,ne,nk,nt,norm,strided);
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
