#include "ggml.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static bool check(ggml_backend_t gpu,ggml_backend_t cpu,int ne,int nt,int mode,bool i64,bool retain,ggml_type type=GGML_TYPE_Q8_0) {
    struct state {ggml_context * ctx;ggml_backend_sched_t sched;ggml_cgraph * graph;ggml_tensor * x,*ids,*cache,*fq,*out;};
    state states[2];constexpr int cap=7;
    for(int k=0;k<2;++k) {
        auto & s=states[k];s.ctx=ggml_init({2<<20,nullptr,true});
        ggml_backend_t backends[]={gpu,cpu};ggml_backend_buffer_type_t bufts[]={ggml_backend_get_default_buffer_type(gpu),ggml_backend_get_default_buffer_type(cpu)};
        s.sched=ggml_backend_sched_new(backends,bufts,2,256,false,true);
        s.x=ggml_new_tensor_2d(s.ctx,GGML_TYPE_F32,ne,nt);
        s.ids=ggml_new_tensor_1d(s.ctx,i64?GGML_TYPE_I64:GGML_TYPE_I32,nt+2);
        s.cache=ggml_new_tensor_2d(s.ctx,type,ne,cap);
        for(auto * t:{s.x,s.ids,s.cache}) {ggml_set_input(t);ggml_backend_sched_set_tensor_backend(s.sched,t,gpu);}
        s.fq=ggml_dsv41_fake_quant(s.ctx,s.x,mode);
        // Keep the original F32 output in the control, without adding work.
        // The other graph may elide it only when it has no other consumer.
        if(!k || retain)ggml_set_output(s.fq);
        auto * ids=ggml_view_1d(s.ctx,s.ids,nt,i64?8:4);
        s.out=ggml_set_rows(s.ctx,s.cache,s.fq,ids);ggml_set_output(s.out);
        s.graph=ggml_new_graph_custom(s.ctx,256,false);ggml_build_forward_expand(s.graph,s.out);
        if(!ggml_backend_sched_alloc_graph(s.sched,s.graph))std::abort();
    }
    const size_t bytes=ggml_nbytes(states[0].cache);
    std::vector<unsigned char> initial(bytes),expected(bytes),actual(bytes);
    for(size_t i=0;i<bytes;++i)initial[i]=static_cast<unsigned char>(i*37+11);
    for(auto & s:states)ggml_backend_tensor_set(s.cache,initial.data(),0,bytes);
    std::vector<float> input(ne*nt),fq_ref(input.size()),fq_actual(input.size());
    std::vector<int64_t> ids64(nt+2);std::vector<int32_t> ids32(nt+2);
    const float amplitudes[]={0.f,1e-20f,1e-7f,.01f,1.f,1000.f,1e6f,1e20f};
    bool ok=true;int exact=0;
    for(int step=0;step<16;++step) {
        for(size_t i=0;i<input.size();++i) {
            float v=amplitudes[step%8]*(std::sin(.13f*i+.37f*step)+.3f*std::cos(.23f*i));
            if(step==8)v=(i&1)?-0.f:0.f;
            if(step==9)v=(i&1)?-1.f:1.f;
            if(step==10)v=std::nextafter(float(int(i%32)-16)/16.f,(i&1)?INFINITY:-INFINITY);
            input[i]=v;
        }
        for(int t=0;t<nt;++t)ids64[t+1]=ids32[t+1]=(step+3*t)%cap;
        for(auto & s:states) {
            ggml_backend_tensor_set(s.x,input.data(),0,input.size()*4);
            ggml_backend_tensor_set(s.ids,i64?(const void*)ids64.data():(const void*)ids32.data(),0,(nt+2)*(i64?8:4));
            ok &= ggml_backend_sched_graph_compute(s.sched,s.graph)==GGML_STATUS_SUCCESS;
        }
        ggml_backend_tensor_get(states[0].out,expected.data(),0,bytes);ggml_backend_tensor_get(states[1].out,actual.data(),0,bytes);
        bool same=!std::memcmp(expected.data(),actual.data(),bytes);
        // All non-selected cache rows, including previously updated rows,
        // must remain unchanged across graph replay and wraparound.
        const size_t rowbytes=bytes/cap;
        for(int row=0;row<cap;++row) {
            bool selected=false;for(int t=0;t<nt;++t)selected|=row==ids64[t+1];
            if(!selected) same &= !std::memcmp(actual.data()+row*rowbytes,initial.data()+row*rowbytes,rowbytes);
        }
        initial=actual;
        if(retain) {
            ggml_backend_tensor_get(states[0].fq,fq_ref.data(),0,fq_ref.size()*4);ggml_backend_tensor_get(states[1].fq,fq_actual.data(),0,fq_actual.size()*4);
            same &= !std::memcmp(fq_ref.data(),fq_actual.data(),fq_ref.size()*4);
        }
        exact+=same;ok &= same;
    }
    std::printf("FQ/store ne=%d nt=%d mode=%d i64=%d retain=%d type=%s full_cache_exact=%d/16 %s\n",ne,nt,mode,i64,retain,ggml_type_name(type),exact,ok?"PASS":"FAIL");
    for(auto & s:states){ggml_backend_sched_free(s.sched);ggml_free(s.ctx);}return ok;
}
int main() {
    ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    if(!gpu || !cpu)return 77;
    bool ok=true;
    for(int ne:{32,128,512,8192})for(int nt:{1,2,4})for(int mode:{0,1,2})for(bool i64:{false,true})ok &= check(gpu,cpu,ne,nt,mode,i64,false);
    for(int mode:{0,1,2}) {
        ok &= check(gpu,cpu,512,1,mode,true,true);
        ok &= check(gpu,cpu,512,1,mode,true,false,GGML_TYPE_F16);
    }
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
