#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct fixture {
    ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    ggml_tensor * storage,*view,*packed,*scaled,*output;
    ggml_cgraph * graph;
    explicit fixture(ggml_backend_t backend) {
        ctx=ggml_init({2*1024*1024,nullptr,true});
        storage=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1024);
        view=ggml_view_2d(ctx,storage,128,2,128*4,0);
        packed=ggml_cont(ctx,view);
        scaled=ggml_scale(ctx,packed,.5f);
        output=ggml_silu(ctx,scaled);
        graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,output);
        buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    }
    ~fixture(){ggml_backend_buffer_free(buffer);ggml_free(ctx);}
    void configure(int phase) {
        const int cols=phase%2?64:128,rows=256/cols;
        const int stride=phase==3?64:128,offset=phase==2?512:phase==3?256:0;
        const float scale=phase==1?1.5f:phase==3?-.5f:.5f;
        view->view_offs=offset*4;view->data=(char *)storage->data+view->view_offs;
        for(auto * t:{view,packed,scaled,output}) {
            t->ne[0]=cols;t->ne[1]=rows;
            t->nb[1]=(t==view?stride:cols)*4;
            t->nb[2]=t->nb[1]*rows;t->nb[3]=t->nb[2];
        }
        std::memcpy(scaled->op_params,&scale,4);
    }
};

int main() {
    ggml_backend_load_all();
    auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
    auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
    if(!gpu || !cpu)return 77;
    bool ok=true;float maximum=0;
    {
        fixture reference(cpu),candidate(gpu);
        std::vector<float> input(1024),expected(256),actual(256);
        // Four stable calls per phase exercise direct warmup, initial capture,
        // replay, then required recapture after shape/stride/pointer/op changes.
        // Returning through the phases also checks reuse of the graph key.
        for(int iteration=0;iteration<32;++iteration) {
            const int phase=(iteration/4)%4;
            for(int i=0;i<1024;++i)input[i]=std::sin(.13f*i+.37f*iteration);
            for(auto * f:{&reference,&candidate}) {
                f->configure(phase);
                ggml_backend_tensor_set(f->storage,input.data(),0,input.size()*4);
            }
            ok &= ggml_backend_graph_compute(cpu,reference.graph)==GGML_STATUS_SUCCESS;
            ok &= ggml_backend_graph_compute(gpu,candidate.graph)==GGML_STATUS_SUCCESS;
            ggml_backend_tensor_get(reference.output,expected.data(),0,expected.size()*4);
            ggml_backend_tensor_get(candidate.output,actual.data(),0,actual.size()*4);
            float error=0;
            for(size_t i=0;i<actual.size();++i) {
                maximum=std::max(maximum,std::abs(actual[i]-expected[i]));
                error=std::max(error,std::abs(actual[i]-expected[i]));
                ok &= std::isfinite(actual[i]);
            }
            if(error>1e-5f)std::fprintf(stderr,"iteration=%d phase=%d error=%g first=%g expected=%g\n",iteration,phase,error,actual[0],expected[0]);
        }
        ok &= maximum<1e-5f;
    }
    std::printf("HIP graph warmup/capture/update with shape, stride, pointer and scale changes: max_error=%g %s\n",maximum,ok?"PASS":"FAIL");
    ggml_backend_free(gpu);ggml_backend_free(cpu);return ok?0:1;
}
