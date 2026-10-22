#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

static bool check(ggml_backend_t gpu, ggml_type type, int heads, int queries, int rows,
                  bool strided, float amplitude=1.f) {
    constexpr int dim=128, streams=2;
    auto ctx=ggml_init({4*1024*1024,nullptr,true});
    auto qs=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,heads+strided,queries+strided,streams);
    auto ks=ggml_new_tensor_4d(ctx,type,dim,strided?2:1,rows+strided,streams);
    auto ws=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,heads+4*strided,queries+strided,1,streams);
    auto ms=ggml_new_tensor_4d(ctx,GGML_TYPE_F16,rows+8*strided,queries+strided,1,strided?streams:1);
    auto q=ggml_view_4d(ctx,qs,dim,heads,queries,streams,qs->nb[1],qs->nb[2],qs->nb[3],0);
    auto k=ggml_view_4d(ctx,ks,dim,1,rows,streams,ks->nb[1],ks->nb[2],ks->nb[3],0);
    auto w=ggml_view_4d(ctx,ws,heads,queries,1,streams,ws->nb[1],ws->nb[2],ws->nb[3],0);
    auto m=ggml_view_4d(ctx,ms,rows,queries,1,ms->ne[3],ms->nb[1],ms->nb[2],ms->nb[3],0);
    auto out=ggml_lightning_indexer(ctx,q,k,w,m);
    const int count=std::min(8,rows);
    auto ids=ggml_cont(ctx,ggml_top_k(ctx,out,count));
    ggml_set_output(out);ggml_set_output(ids);
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,ids);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,gpu);
    if(!buffer)return false;
    std::mt19937 rng(4097);
    std::uniform_real_distribution<float> random(-1,1);
    bool ok=true;double error2=0,norm2=0,max_scaled=0;int overlap=0,total=0;
    for(int pass=0;pass<3;++pass) {
        std::vector<float> qv(ggml_nelements(qs)),kv(ggml_nelements(ks)),wv(ggml_nelements(ws));
        for(auto & v:qv)v=random(rng)*amplitude;
        for(auto & v:kv)v=random(rng)*.5f;
        for(auto & v:wv)v=random(rng)/std::sqrt(float(dim*heads));
        // Cancellation and mixed-sign head weights catch moving ReLU/weights
        // inside the K reduction, and output-layout mistakes.
        std::vector<unsigned char> packed(ggml_nbytes(ks));
        if(type==GGML_TYPE_F32)std::memcpy(packed.data(),kv.data(),packed.size());
        else {
            ggml_quantize_chunk(type,kv.data(),packed.data(),0,ggml_nrows(ks),dim,nullptr);
            ggml_get_type_traits(type)->to_float(packed.data(),kv.data(),kv.size());
        }
        std::vector<ggml_fp16_t> mask(ggml_nelements(ms),ggml_fp32_to_fp16(0));
        for(int s=0;s<ms->ne[3];++s)for(int t=0;t<queries;++t)for(int r=0;r<rows;++r) {
            float v=(r+t+pass)%19==0 || (pass==2 && t==0) ? -INFINITY : -.125f*((r+t)%3);
            if(amplitude!=1.f && std::isfinite(v))v=0;
            mask[(s*ms->nb[3]+t*ms->nb[1])/2+r]=ggml_fp32_to_fp16(v);
        }
        ggml_backend_tensor_set(qs,qv.data(),0,ggml_nbytes(qs));
        ggml_backend_tensor_set(ks,packed.data(),0,packed.size());
        ggml_backend_tensor_set(ws,wv.data(),0,ggml_nbytes(ws));
        ggml_backend_tensor_set(ms,mask.data(),0,ggml_nbytes(ms));
        ok &= ggml_backend_graph_compute(gpu,graph)==GGML_STATUS_SUCCESS;
        std::vector<float> actual(ggml_nelements(out));std::vector<int32_t> selected(ggml_nelements(ids));
        ggml_backend_tensor_get(out,actual.data(),0,ggml_nbytes(out));
        ggml_backend_tensor_get(ids,selected.data(),0,ggml_nbytes(ids));
        for(int s=0;s<streams;++s)for(int t=0;t<queries;++t) {
            std::vector<double> ref(rows);double max_error=0;
            for(int r=0;r<rows;++r) {
                double score=0;
                for(int h=0;h<heads;++h) {
                    double dot=0;
                    for(int d=0;d<dim;++d)dot+=double(qv[(s*q->nb[3]+t*q->nb[2]+h*q->nb[1])/4+d])*kv[(s*k->nb[3]+r*k->nb[2])/ggml_type_size(type)*ggml_blck_size(type)+d];
                    score+=std::max(dot,0.)*wv[(s*w->nb[3]+t*w->nb[1])/4+h];
                }
                const float bias=ggml_fp16_to_fp32(mask[((s%ms->ne[3])*m->nb[3]+t*m->nb[1])/2+r]);
                ref[r]=score+bias;
                const float got=actual[(s*queries+t)*rows+r];
                if(std::isinf(ref[r])){ok &= got==-INFINITY;continue;}
                ok &= std::isfinite(got);
                double err=std::abs(got-ref[r]);max_error=std::max(max_error,err);
                max_scaled=std::max(max_scaled,err/std::max(1.,double(amplitude)));
                error2+=err*err;norm2+=ref[r]*ref[r];
            }
            if(pass==2 && t==0)continue;
            std::vector<int> order(rows);std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](int a,int b){return ref[a]>ref[b];});
            const double boundary=ref[order[count-1]];
            std::vector<bool> used(rows,false);
            for(int i=0;i<count;++i) {
                const int id=selected[(s*queries+t)*count+i];
                if(id<0 || id>=rows){ok=false;continue;}
                ok &= !used[id];used[id]=true;
                ok &= ref[id]>=boundary-2*max_error-1e-7*std::max(1.,double(amplitude));
                overlap+=std::find(order.begin(),order.begin()+count,id)!=order.begin()+count;++total;
            }
        }
    }
    double nmse=error2/std::max(norm2,1e-300);
    ok &= nmse<1e-6 && max_scaled<.003;
    std::printf("indexer %s heads=%d q=%d kv=%d strided=%d amplitude=%g nmse=%g scaled_max=%g topk_overlap=%d/%d %s\n",
        ggml_type_name(type),heads,queries,rows,int(strided),amplitude,nmse,max_scaled,overlap,total,ok?"PASS":"FAIL");
    ggml_backend_buffer_free(buffer);ggml_free(ctx);return ok;
}

int main() {
    ggml_backend_load_all();auto gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    if(!gpu)return 77;
    bool ok=true;
    for(auto type:{GGML_TYPE_F16,GGML_TYPE_Q8_0}) {
        ok &= check(gpu,type,32,1,17,false);
        ok &= check(gpu,type,32,3,65,true);
        ok &= check(gpu,type,32,33,289,true);
        ok &= check(gpu,type,32,2,33,false,1e6f);
        ok &= check(gpu,type,32,2,33,false,1e-9f);
    }
    // Unsupported type/head counts retain the original dispatch.
    ok &= check(gpu,GGML_TYPE_F32,32,3,65,true);
    ok &= check(gpu,GGML_TYPE_Q8_0,4,3,65,true);
    ggml_backend_free(gpu);return ok?0:1;
}
