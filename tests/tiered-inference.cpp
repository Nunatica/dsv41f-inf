#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml-backend-moe-cache.h"
#include "direct-weights.h"
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
    const bool phase_test=std::getenv("DSV4F_TEST_PHASE_WORKSPACE");
    const bool preloaded=phase_test && std::getenv("DSV4F_TEST_PREFILL_RAM");
    auto cp=llama_context_default_params();cp.n_ctx=phase_test?2048:128;cp.n_batch=phase_test?2048:64;cp.n_ubatch=phase_test?2048:32;cp.n_threads=4;cp.n_threads_batch=4;
    cp.n_seq_max=1;cp.offload_kqv=gpu;cp.type_k=GGML_TYPE_Q8_0;cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * context=llama_init_from_model(model,cp);if(!context)throw std::runtime_error("context failed");
    if(phase_test && direct && (llama_n_batch(context)!=1024 || llama_n_ubatch(context)!=1024))throw std::runtime_error("RAM prefill workspace was not capped at 1024");
    if(preloaded && direct) {
        setenv("DSV4F_RAM_PRELOAD","1",1);
        dsv4f_preload_experts();
    }
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));std::vector<float> logits;
    uint64_t first_reads=0;
    for(int pos=0;pos<(phase_test?66:36);) {
        const int count=pos==0 || (phase_test && pos==33)?32:1;
        auto batch=llama_batch_init(count,0,1);batch.n_tokens=count;
        for(int i=0;i<count;++i){batch.token[i]=2+(pos+i)%30;batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1;}
        if(llama_decode(context,batch))throw std::runtime_error("decode failed");
        if(phase_test && direct) {
            const char * phase=std::getenv("DSV4F_PHASE_WORKSPACE");
            const bool switching=phase && std::atoi(phase)!=0;
            const uint32_t expected=switching && count==1?32:1024;
            const auto & actual=context->get_cparams();
            if(actual.n_batch!=expected || actual.n_ubatch!=expected)throw std::runtime_error("incorrect prefill/decode workspace size");
        }
        for(int i=0;i<count;++i){auto * row=llama_get_logits_ith(context,i);logits.insert(logits.end(),row,row+nv);}
        if(phase_test && direct && gpu && count==32) {
            ggml_dsv4f_host_weights_stats stats{};ggml_dsv4f_host_weights.statistics(&stats);
            if((!preloaded && stats.resident_bytes) || stats.transient_bytes)throw std::runtime_error("prefill GPU cache retained duplicate RAM experts");
            if(preloaded && !stats.resident_bytes)throw std::runtime_error("prefill discarded RAM-covered experts");
            if(pos==0)first_reads=stats.reads;
            else if(stats.reads!=first_reads)throw std::runtime_error("warm prefill reread GPU-resident fixture experts");
        }
        llama_batch_free(batch);pos+=count;
    }
    llama_free(context);llama_model_free(model);return logits;
}
int main(int argc,char ** argv) {
    if(argc<2)return 2;const bool gpu=argc>2 && !std::strcmp(argv[2],"--gpu");
    ggml_backend_load_all();if(gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))return 77;
    const auto resident=run(argv[1],false,gpu),direct=run(argv[1],true,gpu);
    if(resident.empty() || resident.size()!=direct.size())return 1;
    double error=0,norm=0,worst=0,prompt_error=0,prompt_norm=0;
    const bool phase_test=std::getenv("DSV4F_TEST_PHASE_WORKSPACE");
    const size_t first_prompt=resident.size()*32/(phase_test?66:36);
    for(size_t i=0;i<resident.size();++i){if(!std::isfinite(direct[i]))return 1;worst=std::max(worst,std::abs(double(resident[i])-direct[i]));const double e=std::pow(double(resident[i])-direct[i],2),n=double(resident[i])*resident[i];error+=e;norm+=n;if(i<first_prompt){prompt_error+=e;prompt_norm+=n;}}
    const double relative=std::sqrt(error/std::max(norm,1e-30));
    const double prompt_relative=std::sqrt(prompt_error/std::max(prompt_norm,1e-30));
    std::printf("[direct-parity] %s prefill/decode logits: relative_rmse=%g first_prefill_rmse=%g worst=%g\n",gpu?"GPU":"CPU",relative,prompt_relative,worst);
    // The nonzero-routing fixture can move decode work from CPU to GPU.
    // Permit its Q8 activation-quantization noise after the first decode,
    // while keeping the first cached prefill comparison strict.
    return relative<(phase_test?1e-3:1e-6) && prompt_relative<1e-6?0:1;
}
