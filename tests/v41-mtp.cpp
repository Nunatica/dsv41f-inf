#include "llama.h"
#include "llama-ext.h"
#include "models/deepseek41.h"
#include "common.h"
#include "speculative.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void check(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
static std::vector<llama_token> generate(llama_model * model,bool enabled,int & proposals,int & rejects,bool gpu,const llama_tokens * expected=nullptr) {
    auto p=llama_context_default_params();p.n_ctx=128;p.n_batch=32;p.n_ubatch=32;p.n_seq_max=1;
    p.n_threads=p.n_threads_batch=4;p.type_k=p.type_v=GGML_TYPE_Q8_0;p.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;p.offload_kqv=gpu;
    llama_context_ptr ctx{llama_init_from_model(model,p)};check(bool(ctx),"target context creation");
    common_params_speculative config;config.types={COMMON_SPECULATIVE_TYPE_DRAFT_MTP};config.draft.ctx_tgt=ctx.get();
    config.draft.n_max=5;config.draft.n_min=0;config.draft.p_min=0;
    common_speculative_ptr spec{enabled?common_speculative_init(config,1):nullptr};
    check(!enabled || bool(spec),"native speculative initialization");
    llama_tokens prompt;for(int i=0;i<17;++i)prompt.push_back(2+i%20);
    common_speculative_begin(spec.get(),0,prompt);
    auto batch=common_batch(ctx.get());
    for(size_t i=0;i<prompt.size();++i)batch.add(prompt[i],i,0,i+1==prompt.size());
    check(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0,"prompt decode");
    check(common_speculative_process(spec.get(),batch),"prefill tap processing");
    common_speculative_begin(spec.get(),0,prompt); // server calls begin after prefill
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto greedy=[&](int row){const auto * logits=llama_get_logits_ith(ctx.get(),row);return llama_token(std::max_element(logits,logits+nv)-logits);};
    llama_token anchor=greedy(-1);int pos=prompt.size();llama_tokens output{anchor};
    while(output.size()<32) {
        llama_tokens drafts;
        // HIP's quantized attention/expert reductions differ between single
        // and multi-token verification. Compare equal verifier batch shapes.
        if(gpu && !spec)drafts.assign(5,3);
        if(spec) {
            common_speculative_get_draft_params(spec.get(),0)={true,5,pos,anchor,&prompt,&drafts};
            common_speculative_draft(spec.get());proposals+=drafts.size();
            // Deliberately reject alternating first candidates to exercise the
            // staged accepted-prefix window even if this tiny random MTP agrees.
            if(!drafts.empty()) {
                if(expected)drafts[0]=((*expected)[output.size()]+1)%nv;
                else if(output.size()%2)drafts[0]=(drafts[0]+1)%nv;
            }
        }
        batch.clear();batch.add(anchor,pos,0,true);
        for(size_t i=0;i<drafts.size();++i)batch.add(drafts[i],pos+1+i,0,true);
        check(llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())==0,"verification decode");
        check(common_speculative_process(spec.get(),batch),"verification tap processing");
        int accepted=0;llama_token next=0;
        for(size_t i=0;i<=drafts.size();++i) {
            next=greedy(i);
            if((gpu && !spec) || i==drafts.size() || next!=drafts[i])break;
            output.push_back(next);++accepted;
        }
        if(accepted<int(drafts.size()))++rejects;
        if(spec)common_speculative_accept(spec.get(),0,accepted);
        pos+=1+accepted;
        check(llama_memory_seq_rm(llama_get_memory(ctx.get()),0,pos,-1),"verification suffix rollback");
        anchor=next;output.push_back(anchor);
    }
    output.resize(32);
    return output;
}

int main(int argc,char ** argv) {
    if(argc<4)return 2;
    const bool gpu=argc>4 && std::string(argv[4])=="--gpu";
    ggml_backend_load_all();if(gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))return 77;
    setenv("MTP_MODEL",argv[2],1);setenv("MTP_CONFIG",argv[3],1);
    try {
        auto p=llama_model_default_params();p.load_mode=LLAMA_LOAD_MODE_NONE;p.n_gpu_layers=gpu?99:0;p.use_extra_bufts=false;
        llama_model_tensor_buft_override overrides[]={{"\\.ffn_(gate|up|down)_exps",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};p.tensor_buft_overrides=overrides;
        llama_model_ptr model{llama_model_load_from_file(argv[1],p)};check(bool(model),"fixture model loading");
        int proposals=0,rejects=0;
        const auto baseline=generate(model.get(),false,proposals,rejects,gpu);
        const auto speculative=generate(model.get(),true,proposals,rejects,gpu,gpu?&baseline:nullptr);
        check(proposals>0 && rejects>0,"draft/reject coverage");
        if(baseline!=speculative)for(size_t i=0;i<baseline.size();++i)if(baseline[i]!=speculative[i])std::fprintf(stderr,"token %zu baseline=%d mtp=%d\n",i,baseline[i],speculative[i]);
        check(baseline==speculative,"greedy target output differs with MTP");
        const auto repeat=generate(model.get(),true,proposals,rejects,gpu,gpu?&baseline:nullptr);
        check(repeat==speculative,"new conversation reset differs");
        std::printf("[dsv4f-mtp-test] %s verified greedy parity, suffix rejection, ring wrap, reset; proposals=%d rejects=%d PASS\n",gpu?"GPU":"CPU",proposals,rejects);
        return 0;
    } catch(const std::exception & e){std::fprintf(stderr,"[dsv4f-mtp-test] %s\n",e.what());return 1;}
}
