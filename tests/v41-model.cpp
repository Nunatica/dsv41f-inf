#include "llama.h"
#include "llama-model.h"
#include "models/deepseek41.h"
#include "ggml-cpu.h"
#include "engram.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstring>
int main(int argc,char ** argv) {
    if(argc<2)return 2;
    auto p=llama_model_default_params();p.no_alloc=true;p.load_mode=LLAMA_LOAD_MODE_NONE;p.n_gpu_layers=0;p.use_extra_bufts=false;
    auto * base=llama_model_load_from_file(argv[1],p);if(!base)return 1;
    auto * m=dynamic_cast<llama_model_deepseek41*>(base);if(!m)return 1;
    int fails=0;
    if(argc>2){dsv4f_engram reader(*m,argv[2]);std::vector<int32_t> history;for(int t=0;t<4;++t)for(int n=0;n<m->engram_ngram;++n)history.push_back(t==0?-1:t==1?0:100+t*7-n);FILE*f=fopen("/tmp/dsv4f-v41-engram.f32","wb");for(int slot=0;slot<(int)m->engram_layers.size();++slot){std::vector<float> out;reader.read(slot,history,4,out);fwrite(out.data(),4,out.size(),f);}fclose(f);}

    for(auto & layer:m->tiers){std::vector<int> seen(m->hparams.n_expert);for(auto & tier:layer){int n=0;for(size_t i=0;i<tier.local.size();++i)if(tier.local[i]>=0){if(tier.local[i]!=n++)++fails;++seen[i];}if(n!=tier.gate->ne[2])++fails;}for(int n:seen)if(n!=1)++fails;}
    // PTQ1_0 dot vs independent float dequantization, covering random packed bytes.
    const int n=5120;std::vector<unsigned char> q(n/128*28);for(size_t i=0;i<q.size();++i)q[i]=(i*71+3)%256;
    ggml_fp16_t scale=ggml_fp32_to_fp16(0.125f);for(int b=0;b<n/128;++b)std::memcpy(q.data()+28*b+26,&scale,2);
    std::vector<float>w(n),x(n),qx(n);for(int i=0;i<n;++i)x[i]=std::sin(i*0.17f);
    auto * t=ggml_get_type_traits(GGML_TYPE_PTQ1_0);t->to_float(q.data(),w.data(),n);
    auto * dot=ggml_get_type_traits_cpu(GGML_TYPE_PTQ1_0);std::vector<unsigned char>a(ggml_row_size(dot->vec_dot_type,n));ggml_get_type_traits_cpu(dot->vec_dot_type)->from_float(x.data(),a.data(),n);ggml_get_type_traits(dot->vec_dot_type)->to_float(a.data(),qx.data(),n);
    float got=0;dot->vec_dot(n,&got,0,q.data(),0,a.data(),0,1);double want=0;for(int i=0;i<n;++i)want+=(double)w[i]*qx[i];if(std::abs(got-want)>1e-4)++fails;
    std::printf("[dsv4f-v41-test] native loader, %zu mixed-format maps, PTQ dot: %s\n",m->tiers.size(),fails?"FAIL":"PASS");llama_model_free(base);return fails?1:0;
}
