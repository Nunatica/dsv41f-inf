#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
static bool check(ggml_backend_t gpu,int k,int m,int nt,int channels,ggml_type type) {
    struct state {ggml_context * ctx;ggml_backend_buffer_t buffer;ggml_cgraph * graph;ggml_tensor * w,*x,*out;};
    state s[3];
    for(auto & a:s){a.ctx=ggml_init({2<<20,nullptr,true});a.w=ggml_new_tensor_3d(a.ctx,type,k,m,channels);a.x=ggml_new_tensor_3d(a.ctx,GGML_TYPE_F32,k,nt,channels);a.out=ggml_mul_mat(a.ctx,a.w,a.x);ggml_mul_mat_set_prec(a.out,GGML_PREC_F32);ggml_set_output(a.out);a.graph=ggml_new_graph(a.ctx);ggml_build_forward_expand(a.graph,a.out);a.buffer=ggml_backend_alloc_ctx_tensors(a.ctx,gpu);if(!a.buffer)return false;}
    std::mt19937 rng(5742+k+m+nt+channels);std::normal_distribution<float> d(0,.03f);
    std::vector<float> w(ggml_nelements(s[0].w)),x(ggml_nelements(s[0].x)),out[3];for(auto & v:out)v.resize(ggml_nelements(s[0].out));
    std::vector<ggml_fp16_t> hw(w.size());for(size_t i=0;i<w.size();++i){w[i]=d(rng);hw[i]=ggml_fp32_to_fp16(w[i]);if(type==GGML_TYPE_F16)w[i]=ggml_fp16_to_fp32(hw[i]);}
    for(auto & a:s)ggml_backend_tensor_set(a.w,type==GGML_TYPE_F16?(const void*)hw.data():(const void*)w.data(),0,ggml_nbytes(a.w));
    bool ok=true;double worst=0,oracle_worst=0;
    for(int rep=0;rep<16;++rep){float scales[]={0.f,.0001f,.01f,1.f,20.f,100.f};for(auto & v:x)v=d(rng)*scales[rep%6];
        for(int mode=0;mode<3;++mode){if(mode==2)unsetenv("DSV4F_V41_HC_MAT_SPLIT");else setenv("DSV4F_V41_HC_MAT_SPLIT",mode?"1":"0",1);ggml_backend_tensor_set(s[mode].x,x.data(),0,x.size()*4);ok&=ggml_backend_graph_compute(gpu,s[mode].graph)==GGML_STATUS_SUCCESS;ggml_backend_tensor_get(s[mode].out,out[mode].data(),0,out[mode].size()*4);}
        ok&=!std::memcmp(out[1].data(),out[2].data(),out[1].size()*4);
        double err=0,norm=0,oe=0,on=0;
        for(int c=0;c<channels;++c)for(int t=0;t<nt;++t)for(int r=0;r<m;++r){int idx=(c*nt+t)*m+r;double ref=0;for(int i=0;i<k;++i)ref+=double(w[(c*m+r)*k+i])*x[(c*nt+t)*k+i];double e=out[1][idx]-out[0][idx];err+=e*e;norm+=double(out[0][idx])*out[0][idx];double od=out[1][idx]-ref;oe+=od*od;on+=ref*ref;ok&=std::isfinite(out[0][idx])&&std::isfinite(out[1][idx]);}
        worst=std::max(worst,std::sqrt(err/std::max(norm,1e-30)));oracle_worst=std::max(oracle_worst,std::sqrt(oe/std::max(on,1e-30)));
        if(k!=20480||m!=24||nt!=1||channels!=1||type!=GGML_TYPE_F16)ok&=!std::memcmp(out[0].data(),out[1].data(),out[0].size()*4);
    }
    ok&=worst<2e-6&&oracle_worst<2e-6;
    std::printf("HC mat split K=%d M=%d tokens=%d channels=%d type=%s relative=%g oracle=%g %s\n",k,m,nt,channels,ggml_type_name(type),worst,oracle_worst,ok?"PASS":"FAIL");
    for(auto & a:s){ggml_backend_buffer_free(a.buffer);ggml_free(a.ctx);}return ok;
}
int main(){ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);if(!gpu)return 77;bool ok=check(gpu,20480,24,1,1,GGML_TYPE_F16);ok&=check(gpu,20480,24,2,1,GGML_TYPE_F16);ok&=check(gpu,20480,24,1,2,GGML_TYPE_F16);ok&=check(gpu,1024,24,1,1,GGML_TYPE_F16);ok&=check(gpu,20480,23,1,1,GGML_TYPE_F16);ok&=check(gpu,20480,24,1,1,GGML_TYPE_F32);ggml_backend_free(gpu);return ok?0:1;}
