#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Isolated D512 attention timing, including its ordinary scratch/combine work.
int main(int argc,char ** argv) {
    const int queries=argc>1?std::atoi(argv[1]):128;
    const int rows=argc>2?std::atoi(argv[2]):16384;
    const int repeats=argc>3?std::atoi(argv[3]):20;
    const bool dense=argc>4 && !std::strcmp(argv[4],"dense");
    const bool prefix=argc>4 && !std::strcmp(argv[4],"prefix");
    const ggml_type type=argc>5 && !std::strcmp(argv[5],"q8_0")?GGML_TYPE_Q8_0:GGML_TYPE_F16;
    if(queries<1 || rows<2048 || rows%256 || repeats<1)return 2;
    constexpr int dim=512,heads=64;
    ggml_backend_load_all();auto gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dim,queries,heads);
    auto k=ggml_new_tensor_3d(ctx,type,dim,rows,1);
    auto mask=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,rows,queries);
    auto sinks=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,heads);
    auto out=ggml_flash_attn_ext(ctx,q,k,k,mask,1/std::sqrt(float(dim)),0,0);
    ggml_flash_attn_ext_add_sinks(out,sinks);ggml_prec_set_acc(out,GGML_PREC_F32);
    ggml_flash_attn_ext_set_n_kv_max(out,dense?0:640);
    ggml_set_output(out);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);if(!buffer)return 3;
    uint32_t random=731;
    auto upload=[&](ggml_tensor * t) {
        std::vector<float> values(ggml_nelements(t));
        for(auto & v:values){random=1664525*random+1013904223;v=(float(random>>8)/16777216.f-.5f)*.8f;}
        if(t->type==GGML_TYPE_F32)ggml_backend_tensor_set(t,values.data(),0,ggml_nbytes(t));
        else {
            std::vector<unsigned char> data(ggml_nbytes(t));
            ggml_quantize_chunk(t->type,values.data(),data.data(),0,ggml_nrows(t),t->ne[0],nullptr);
            ggml_backend_tensor_set(t,data.data(),0,data.size());
        }
    };
    upload(q);upload(k);upload(sinks);
    std::vector<ggml_fp16_t> masks(ggml_nelements(mask),ggml_fp32_to_fp16(dense?0:-INFINITY));
    if(!dense)for(int t=0;t<queries;++t) {
        for(int i=0;i<128;++i)masks[size_t(t)*rows+i]=ggml_fp32_to_fp16(0);
        const int visible=prefix?std::min(rows-128,16384-128):rows-128;
        for(int i=0;i<512;++i){random=1664525*random+1013904223;int r=128+(random>>8)%visible;masks[size_t(t)*rows+r]=ggml_fp32_to_fp16(-.125f*(i%3));}
    }
    ggml_backend_tensor_set(mask,masks.data(),0,ggml_nbytes(mask));
    std::vector<double> ms;
    for(int i=-5;i<repeats;++i) {
        const auto start=std::chrono::steady_clock::now();
        if(ggml_backend_graph_compute(gpu,graph)!=GGML_STATUS_SUCCESS)return 4;
        ggml_backend_synchronize(gpu);
        if(i>=0)ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::vector<float> result(ggml_nelements(out));ggml_backend_tensor_get(out,result.data(),0,ggml_nbytes(out));
    bool ok=true;double norm=0;for(float v:result){ok &= std::isfinite(v);norm+=double(v)*v;}
    std::sort(ms.begin(),ms.end());
    std::printf("attention q=%d kv=%d pattern=%s type=%s repeats=%d median_ms=%.6f p95_ms=%.6f norm2=%.9g %s\n",
        queries,rows,dense?"dense":prefix?"prefix":"sparse",ggml_type_name(type),repeats,ms[ms.size()/2],ms[std::min(ms.size()-1,ms.size()*95/100)],norm,ok?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(gpu);return ok?0:1;
}
