#include "moe.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <vector>
#include <random>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <algorithm>
#include <fstream>

static bool zero_cache=false;
static bool bench=false;
static bool prefill_bench=false;
static const char * dump_dir=nullptr;
static std::vector<std::pair<ggml_context *,ggml_backend_buffer_t>> held;
static bool check(ggml_backend_t gpu,ggml_backend_t cpu,ggml_type type,int ne,int nf,int nt,bool keep=false) {
    const int nex=3,nk=3;ggml_context * wc=ggml_init({1<<20,nullptr,true});
    auto * g=ggml_new_tensor_3d(wc,type,ne,nf,nex),*u=ggml_dup_tensor(wc,g),*d=ggml_new_tensor_3d(wc,type,nf,ne,nex);
    auto * wb=ggml_backend_alloc_ctx_tensors_from_buft(wc,ggml_backend_dev_host_buffer_type(ggml_backend_get_device(gpu)));ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::mt19937 rng(4101);std::normal_distribution<float> normal(0,.03f);
    for(auto * t:{g,u,d}) {
        if(type==GGML_TYPE_PTQ1_0){auto * b=(unsigned char*)t->data;const auto scale=ggml_fp32_to_fp16(.03f);for(size_t i=0;i<ggml_nbytes(t)/28;++i){for(int j=0;j<26;++j)b[i*28+j]=rng()%256;std::memcpy(b+i*28+26,&scale,2);}}
        else {std::vector<float> values(ggml_nelements(t));for(float & v:values)v=normal(rng);ggml_quantize_chunk(type,values.data(),t->data,0,t->ne[1]*t->ne[2],t->ne[0],nullptr);}
    }
    ggml_backend_t backends[]={gpu,cpu};auto * sched=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);
    ggml_context * gc[2];ggml_backend_buffer_t cb=nullptr;ggml_tensor * x[2],*ids[2],*w[2],*out[2];ggml_cgraph * graph[2];
    for(int k=0;k<2;++k){gc[k]=ggml_init({4<<20,nullptr,true});x[k]=ggml_new_tensor_2d(gc[k],GGML_TYPE_F32,ne,nt);ids[k]=ggml_new_tensor_2d(gc[k],GGML_TYPE_I32,nk,nt);w[k]=ggml_new_tensor_3d(gc[k],GGML_TYPE_F32,1,nk,nt);
        if(!k){auto * xx=ggml_reshape_3d(gc[k],x[k],ne,1,nt);auto * gate=ggml_mul_mat_id(gc[k],g,xx,ids[k]),*up=ggml_mul_mat_id(gc[k],u,xx,ids[k]);auto * h=ggml_swiglu_clamp(gc[k],gate,up,1.25f);auto * v=ggml_mul(gc[k],ggml_mul_mat_id(gc[k],d,h,ids[k]),w[k]);out[k]=nullptr;for(int j=0;j<nk;++j){auto * q=ggml_view_2d(gc[k],v,ne,nt,v->nb[2],j*v->nb[1]);out[k]=out[k]?ggml_add(gc[k],out[k],q):q;}cb=ggml_backend_alloc_ctx_tensors(gc[k],cpu);}
        else out[k]=dsv4f_moe(gc[k],x[k],ids[k],w[k],g,u,d,1.25f);
        graph[k]=ggml_new_graph_custom(gc[k],1024,false);ggml_build_forward_expand(graph[k],out[k]);ggml_set_input(x[k]);ggml_set_input(ids[k]);ggml_set_input(w[k]);ggml_set_output(out[k]);
    }
    if(!ggml_dsv4f_experts.gpu_supported(out[1]) || !ggml_backend_sched_alloc_graph(sched,graph[1]) || ggml_backend_sched_get_tensor_backend(sched,out[1])!=gpu){std::fprintf(stderr,"GPU expert path unavailable for %s\n",ggml_type_name(type));return false;}
    std::vector<float> a(ne*nt),b(ne*nt),in(ne*nt),prob(nk*nt);std::vector<int32_t> selected(nk*nt);for(auto & v:in)v=normal(rng)*8;for(int i=0;i<nk*nt;++i){selected[i]=i%3;prob[i]=i%5==0?0.f:(i%3+1)/6.f;}
    bool ok=true;double worst=0;
    for(int step=0;step<(prefill_bench?1:4);++step){for(int k=0;k<2;++k){ggml_backend_tensor_set(x[k],in.data(),0,in.size()*4);ggml_backend_tensor_set(ids[k],selected.data(),0,selected.size()*4);ggml_backend_tensor_set(w[k],prob.data(),0,prob.size()*4);ok&=(k?ggml_backend_sched_graph_compute(sched,graph[k]):ggml_backend_graph_compute(cpu,graph[k]))==GGML_STATUS_SUCCESS;ggml_backend_tensor_get(out[k],k?b.data():a.data(),0,a.size()*4);}
        double err=0,norm=0;for(size_t i=0;i<a.size();++i){err+=(double)(a[i]-b[i])*(a[i]-b[i]);norm+=(double)a[i]*a[i];ok&=std::isfinite(b[i]);}double rel=sqrt(err/std::max(norm,1e-30));worst=std::max(worst,rel);ok&=rel<.04;
        if(nt==1)ggml_dsv4f_experts.gpu_resize(true);
    }
    if (prefill_bench) {
        const auto expected = b;
        std::vector<double> times;
        for (int i=-3; i<15; ++i) {
            const auto start = std::chrono::steady_clock::now();
            ok &= ggml_backend_sched_graph_compute(sched,graph[1]) == GGML_STATUS_SUCCESS;
            ggml_backend_sched_synchronize(sched);
            if (i>=0) times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
        }
        ggml_backend_tensor_get(out[1], b.data(), 0, b.size()*sizeof(float));
        ok &= std::memcmp(expected.data(), b.data(), b.size()*sizeof(float)) == 0;
        std::sort(times.begin(),times.end());
        std::printf("[dsv4f-prefill-bench] format=%s tokens=%d median_ms=%.6f p95_ms=%.6f stable=%d\n",
            ggml_type_name(type),nt,times[times.size()/2],times[times.size()*95/100],int(ok));
        if (dump_dir) {
            const std::string path=std::string(dump_dir)+"/prefill-"+ggml_type_name(type)+"-t"+std::to_string(nt)+".f32";
            std::ofstream file(path,std::ios::binary);
            file.write(reinterpret_cast<const char *>(b.data()),b.size()*sizeof(float));
            ok &= file.good();
        }
    }
    if(bench) {
        // Fixed input, no admissions: isolate host handoff from changing expert
        // routes/cache hits. A separate idle series forces the service to sleep.
        for(int i=0;i<8;++i){ok&=ggml_backend_sched_graph_compute(sched,graph[1])==GGML_STATUS_SUCCESS;ggml_backend_tensor_get(out[1],b.data(),0,b.size()*sizeof(float));ggml_dsv4f_experts.gpu_resize(true);}
        const auto expected=b;
        for(bool idle:{false,true}) {
            constexpr int repeats=200;
            double wall_us=0;std::vector<double> samples;
            for(int i=0;i<repeats;++i) {
                if(idle)std::this_thread::sleep_for(std::chrono::milliseconds(10));
                const auto start=std::chrono::steady_clock::now();
                ok&=ggml_backend_sched_graph_compute(sched,graph[1])==GGML_STATUS_SUCCESS;
                ggml_backend_tensor_get(out[1],b.data(),0,b.size()*sizeof(float));
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
                wall_us+=us;samples.push_back(us);
                ok&=std::memcmp(expected.data(),b.data(),b.size()*sizeof(float))==0;
            }
            std::sort(samples.begin(),samples.end());
            std::printf("[dsv4f-service-bench] format=%s idle=%d n=%d mean_us=%.3f median_us=%.3f p95_us=%.3f bit_exact=%d\n",ggml_type_name(type),int(idle),repeats,wall_us/repeats,samples[repeats/2],samples[repeats*95/100],int(ok));
        }
    }
    if(nt==32) {
        auto * dc=ggml_init({2<<20,nullptr,true});auto * dx=ggml_new_tensor_2d(dc,GGML_TYPE_F32,ne,1),*di=ggml_new_tensor_2d(dc,GGML_TYPE_I32,nk,1),*dw=ggml_new_tensor_3d(dc,GGML_TYPE_F32,1,nk,1);
        auto * result=dsv4f_moe(dc,dx,di,dw,g,u,d,1.25f);auto * dg=ggml_new_graph(dc);ggml_build_forward_expand(dg,result);auto * ds=ggml_backend_sched_new(backends,nullptr,2,1024,false,true);
        ggml_set_input(dx);ggml_set_input(di);ggml_set_input(dw);ggml_set_output(result);ok&=ggml_backend_sched_alloc_graph(ds,dg);
        ggml_backend_tensor_set(dx,in.data(),0,ne*4);ggml_backend_tensor_set(di,selected.data(),0,nk*4);ggml_backend_tensor_set(dw,prob.data(),0,nk*4);
        for(int j=0;j<8;++j)ok&=ggml_backend_sched_graph_compute(ds,dg)==GGML_STATUS_SUCCESS;
        ggml_dsv4f_experts.gpu_resize(false);const size_t before=ggml_dsv4f_experts.cache_bytes();ggml_dsv4f_experts.gpu_resize(true);const size_t after=ggml_dsv4f_experts.cache_bytes();
        const bool phase_ok=before==0 && (zero_cache?after==0:after>before && after<=128ULL*1024*1024);ok&=phase_ok;
        std::printf("[dsv4f-v41-test] phase workspace cache prefill=%zu decode=%zu ceiling=%zu %s\n",before,after,zero_cache?size_t(0):size_t(128ULL*1024*1024),phase_ok?"PASS":"FAIL");ggml_backend_sched_free(ds);ggml_free(dc);
    }
    std::printf("[dsv4f-v41-test] %s D=%d FF=%d tokens=%d CPU-reference relative_rmse=%g %s\n",ggml_type_name(type),ne,nf,nt,worst,ok?"PASS":"FAIL");
    if(!keep)ggml_dsv4f_experts.invalidate(g->data,ggml_nbytes(g));ggml_backend_sched_free(sched);ggml_backend_buffer_free(cb);for(auto * c:gc)ggml_free(c);if(keep)held.push_back({wc,wb});else{ggml_backend_buffer_free(wb);ggml_free(wc);}return ok;
}
int main(int argc,char ** argv){prefill_bench=argc>1 && !std::strcmp(argv[1],"--prefill-bench");dump_dir=argc>2?argv[2]:nullptr;bench=argc>1 && (!std::strcmp(argv[1],"--bench") || !std::strcmp(argv[1],"--bench-cache"));zero_cache=argc>1 && std::strcmp(argv[1],"--bench-cache");setenv("DSV4F_GPU_SERVICE","1",1);setenv("DSV4F_GPU_WAIT","callback",1);setenv("DSV4F_EXPERT_CACHE_MIB",zero_cache?"0":bench?"192":"128",1);setenv("DSV4F_VRAM_HEADROOM_MIB","256",1);setenv("DSV4F_VRAM_GUARD_MIB","256",1);setenv("DSV4F_PREFILL_CACHE","0",1);ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);auto * cpu=ggml_backend_cpu_init();if(!gpu)return 77;ggml_backend_cpu_set_n_threads(cpu,12);bool ok=true;if(prefill_bench){for(auto t:{GGML_TYPE_Q2_0,GGML_TYPE_PTQ1_0})for(int nt:{32,128,512})ok&=check(gpu,cpu,t,5120,2304,nt);ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;}if(!bench)for(auto t:{GGML_TYPE_Q3_K,GGML_TYPE_Q2_0,GGML_TYPE_PTQ1_0})for(int nt:{1,32})ok&=check(gpu,cpu,t,5120,2304,nt);size_t previous=0;for(auto t:{GGML_TYPE_Q3_K,GGML_TYPE_Q2_0,GGML_TYPE_PTQ1_0}){ok&=check(gpu,cpu,t,5120,2304,1,true);size_t used=ggml_dsv4f_experts.cache_bytes();ok&=zero_cache?used==0:used>previous&&used<=(bench?192ULL:128ULL)*1024*1024;previous=used;std::printf("[dsv4f-v41-test] aggregate cache %s bytes=%zu ceiling=%zu\n",ggml_type_name(t),used,zero_cache?size_t(0):size_t((bench?192ULL:128ULL)*1024*1024));}
    for(auto & pair:held){auto * t=ggml_get_first_tensor(pair.first);ggml_dsv4f_experts.invalidate(t->data,ggml_nbytes(t));ggml_backend_buffer_free(pair.second);ggml_free(pair.first);}ok&=ggml_dsv4f_experts.cache_bytes()==0;ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;}
