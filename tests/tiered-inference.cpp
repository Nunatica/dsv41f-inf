#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static std::vector<float> run(const char * path,bool direct,bool gpu) {
    const char * test_budget=std::getenv("DSV4F_TEST_RAM_BUDGET_MIB");
    setenv("DSV4F_RAM_BUDGET_MIB",direct?(test_budget?test_budget:"131072"):"0",1);
    setenv("DSV4F_RAM_CACHE_MIB","64",1);
    auto mp=llama_model_default_params();mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.n_gpu_layers=gpu?999:0;mp.use_extra_bufts=false;
    const llama_model_tensor_buft_override overrides[]={{"\\.ffn_(gate|up|down)_exps",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    mp.tensor_buft_overrides=overrides;
    auto * model=llama_model_load_from_file(path,mp);if(!model)throw std::runtime_error("model load failed");
    auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=64;cp.n_ubatch=32;cp.n_threads=4;cp.n_threads_batch=4;
    cp.n_seq_max=1;cp.offload_kqv=gpu;cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * context=llama_init_from_model(model,cp);if(!context)throw std::runtime_error("context failed");
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));std::vector<float> logits;
    for(int pos=0;pos<36;) {
        const int count=pos==0?32:1;
        auto batch=llama_batch_init(count,0,1);batch.n_tokens=count;
        for(int i=0;i<count;++i){batch.token[i]=2+(pos+i)%30;batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1;}
        if(llama_decode(context,batch))throw std::runtime_error("decode failed");
        for(int i=0;i<count;++i){auto * row=llama_get_logits_ith(context,i);logits.insert(logits.end(),row,row+nv);}
        llama_batch_free(batch);pos+=count;
    }
    llama_free(context);llama_model_free(model);return logits;
}
int main(int argc,char ** argv) {
    if(argc<2)return 2;const bool gpu=argc>2 && !std::strcmp(argv[2],"--gpu");
    ggml_backend_load_all();if(gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))return 77;
    const auto resident=run(argv[1],false,gpu),direct=run(argv[1],true,gpu);
    if(resident.empty() || resident.size()!=direct.size())return 1;
    double error=0,norm=0,worst=0;
    for(size_t i=0;i<resident.size();++i){if(!std::isfinite(direct[i]))return 1;worst=std::max(worst,std::abs(double(resident[i])-direct[i]));error+=std::pow(double(resident[i])-direct[i],2);norm+=double(resident[i])*resident[i];}
    const double relative=std::sqrt(error/std::max(norm,1e-30));
    std::printf("[direct-parity] %s prefill/decode logits: relative_rmse=%g worst=%g\n",gpu?"GPU":"CPU",relative,worst);
    return relative<1e-6?0:1;
}
