#include "deepseek41.h"
#include "llama-model-loader.h"
#include "gguf.h"
#include <type_traits>
#include "llama-kv-cache-dsv41.h"
#include "llama-cparams.h"
#include "engram.h"
#include "ggml-dsv4f-timeline.h"
#include "moe.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <set>
#include <stdexcept>
using json=nlohmann::json;
void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    auto u=[&](const char * n,auto & v){
        const std::string key=std::string("deepseek41.")+n;
        using T=std::decay_t<decltype(v)>;
        if constexpr(std::is_same_v<T,std::string>)ml.get_key(key,v);
        else {const int k=gguf_find_key(ml.metadata,key.c_str());if(k<0)throw std::runtime_error("missing "+key);double x=0;switch(gguf_get_kv_type(ml.metadata,k)){
            case GGUF_TYPE_UINT32:x=gguf_get_val_u32(ml.metadata,k);break;
            case GGUF_TYPE_INT32:x=gguf_get_val_i32(ml.metadata,k);break;
            case GGUF_TYPE_UINT64:x=gguf_get_val_u64(ml.metadata,k);break;
            case GGUF_TYPE_FLOAT32:x=gguf_get_val_f32(ml.metadata,k);break;
            case GGUF_TYPE_FLOAT64:x=gguf_get_val_f64(ml.metadata,k);break;
            case GGUF_TYPE_BOOL:x=gguf_get_val_bool(ml.metadata,k);break;
            default:throw std::runtime_error("non-numeric "+key);}
            if(!std::isfinite(x))throw std::runtime_error("non-finite "+key);v=(T)x;
        }};
    u("moe_intermediate_size",ff);u("hc_mult",hc);u("o_groups",og);u("o_lora_rank",ol);
    u("index_n_heads",idx_heads);u("index_head_dim",idx_dim);u("index_topk",idx_top);
    u("candidate_source_layer_id",candidate_layer);u("candidate_topk_blocks",candidate_blocks);u("candidate_block_size",candidate_size);
    u("rms_norm_eps",eps);u("hc_eps",hc_eps);u("swiglu_limit",clamp);u("routed_scaling_factor",scaling);
    u("norm_topk_prob",norm_topk);u("rope_theta",rope_base);u("compress_rope_theta",compress_base);
    u("rope_scaling.factor",rope_factor);u("rope_scaling.beta_fast",beta_fast);u("rope_scaling.beta_slow",beta_slow);float original=0;u("rope_scaling.original_max_position_embeddings",original);orig_ctx=(int)original;hparams.n_ctx_orig_yarn=orig_ctx;
    u("q_lora_rank",hparams.n_lora_q);u("sliding_window",hparams.n_swa);u("hc_sinkhorn_iters",hparams.dsv4_hc_sinkhorn_iters);
    ml.get_arr("deepseek41.compress_ratios",ratios);std::vector<int32_t> ks,is;
    ml.get_arr("deepseek41.kv_source_layer_ids",ks);ml.get_arr("deepseek41.index_source_layer_ids",is);
    if(ratios.size()!=hparams.n_layer())throw std::runtime_error("V4.1 compress_ratios length mismatch");
    for(auto r:ratios)if(r<0||r>2)throw std::runtime_error("unsupported V4.1 compression ratio");
    for(auto & list:{ks,is})for(int k:list)if(k<0||k>=(int)ratios.size())throw std::runtime_error("V4.1 source layer out of range");
    kv_sources.assign(hparams.n_layer(),-1);index_sources=kv_sources;
    // Source lists name the latest source at/before the layer (rather than one item per layer).
    for(size_t il=0;il<ratios.size();++il)if(ratios[il]) {
        for(int k:ks)if(k<=(int)il && ratios[k]==ratios[il])kv_sources[il]=k;
        for(int k:is)if(k<=(int)il && ratios[k]==ratios[il])index_sources[il]=k;
        if(kv_sources[il]<0 || index_sources[il]<0)throw std::runtime_error("invalid V4.1 shared KV/index topology");
    }
    if(ratios.size()!=hparams.n_layer())throw std::runtime_error("V4.1 compress_ratios length mismatch");
    for(auto r:ratios)if(r<0||r>2)throw std::runtime_error("unsupported V4.1 compression ratio");
    std::string cfg;u("config",cfg);auto root=json::parse(cfg);auto c=root.contains("text_config")?root.at("text_config"):root;
    dspark_block=c.value("dspark_block_size",0);dspark_noise=c.value("dspark_noise_token_id",0);
    dspark_experts=c.value("dspark_n_routed_experts",0);dspark_used=c.value("dspark_num_experts_per_tok",0);
    dspark_rank=c.value("dspark_markov_rank",0);dspark_layers=c.value("dspark_target_layer_ids",std::vector<int32_t>{});
    engram_heads=c.at("engram_n_heads");engram_dim=c.at("engram_head_dim");engram_ngram=c.at("engram_max_ngram_size");
    u("engram.pad_id",engram_pad);ml.get_arr("deepseek41.engram.layer_ids",engram_layers);
    ml.get_arr("deepseek41.engram.rows",engram_rows);ml.get_arr("deepseek41.engram.token_map",token_map);
    ml.get_arr("deepseek41.engram.primes",primes);ml.get_arr("deepseek41.engram.multipliers",multipliers);
    if(hc!=4 || engram_ngram<2 || engram_ngram>8 || engram_dim%32 || primes.size()!=engram_layers.size()*(engram_ngram-1)*engram_heads || multipliers.size()!=engram_layers.size()*engram_ngram || engram_rows.size()!=engram_layers.size())throw std::runtime_error("unsupported V4.1 Engram geometry");
    uint32_t v41_vocab=0;u("vocab_size",v41_vocab);
    if(engram_heads<=0 || engram_dim<=0 || token_map.size()!=v41_vocab || idx_top<=0 || candidate_size<=0 || candidate_blocks<=0 || ff<=0 || og<=0 || ol<=0 || idx_dim<=0 || idx_heads<=0 || hparams.n_head()*hparams.n_embd_head_k()%og)throw std::runtime_error("invalid V4.1 model geometry");
    for(size_t slot=0;slot<engram_layers.size();++slot){const int layer=engram_layers[slot];if(layer<0||layer>=int(hparams.n_layer())||std::count(engram_layers.begin(),engram_layers.end(),layer)!=1)throw std::runtime_error("invalid Engram layer");uint64_t rows=0;for(int c=0;c<(engram_ngram-1)*engram_heads;++c){auto prime=primes[slot*(engram_ngram-1)*engram_heads+c];if(!prime)throw std::runtime_error("invalid Engram prime");rows+=prime;}if(rows!=engram_rows[slot])throw std::runtime_error("Engram rows do not match hash partitions");}
    hparams.rope_freq_base_train=rope_base;hparams.n_head_kv_arr.fill(1);hparams.n_ff_exp_arr.fill(ff);hparams.n_ff_arr.fill(ff);
    hparams.f_norm_rms_eps=eps;hparams.dsv4_hc_mult=hc;hparams.dsv4_hc_eps=hc_eps;
    hparams.dsv4_o_group_count=og;hparams.dsv4_o_lora_rank=ol;hparams.n_expert_shared=1;
    hparams.n_embd_out_impl=hparams.n_embd;hparams.n_layer_nextn=0;
    hparams.expert_gating_func=LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS;hparams.expert_weights_scale=scaling;hparams.expert_weights_norm=norm_topk;
    hparams.swiglu_clamp_exp.fill(clamp);hparams.swiglu_clamp_shexp.fill(clamp);
    tiers.resize(hparams.n_layer());
    LLAMA_LOG_INFO("[dsv4f-v41] native llama.cpp graph: %u layers, %u experts, shared KV, Engram\n",hparams.n_layer(),hparams.n_expert);
}
ggml_tensor * llama_model_deepseek41::W(int l,const std::string & role) const {return weights.at(l<0?role:"blk."+std::to_string(l)+"."+role);}
void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    const LLM_TN tn(arch);
    for(auto & entry:ml.weights_map) {
        const std::string & name=entry.first;auto * t=entry.second.tensor;int il=-1;std::sscanf(name.c_str(),"blk.%d.",&il);
        const bool aux=name.rfind("mtp.",0)==0||name.rfind("vision.",0)==0||name.rfind("aligner.",0)==0||name.rfind("image_",0)==0||name.find("exp_probs_b_vl.bias")!=std::string::npos;
        llm_tensor kind=il<0?LLM_TENSOR_OUTPUT:LLM_TENSOR_ATTN_Q;
        if(name=="token_embd.weight")kind=LLM_TENSOR_TOKEN_EMBD;
        else if(name=="output_norm.weight")kind=LLM_TENSOR_OUTPUT_NORM;
        else if(il>=0 && t->ne[1]==1)kind=LLM_TENSOR_FFN_NORM;
        if(name.find("ffn_gate_exps")!=std::string::npos)kind=LLM_TENSOR_FFN_GATE_EXPS;
        if(name.find("ffn_up_exps")!=std::string::npos)kind=LLM_TENSOR_FFN_UP_EXPS;
        if(name.find("ffn_down_exps")!=std::string::npos)kind=LLM_TENSOR_FFN_DOWN_EXPS;
        if(name.find("ffn_exps_")!=std::string::npos)kind=LLM_TENSOR_FFN_GATE_TID2EID;
        if(aux){kind=LLM_TENSOR_TOKEN_EMBD;il=-1;}
        auto spec=tn(kind,il);spec.name_override=name;
        weights[name]=create_tensor(spec,{t->ne[0],t->ne[1],t->ne[2],t->ne[3]},aux?llama_model_loader::TENSOR_SKIP:0);
    }
    tok_embd=W(-1,"token_embd.weight");output=W(-1,"output.weight");output_norm=W(-1,"output_norm.weight");
    for(uint32_t l=0;l<hparams.n_layer();++l) {
        const std::string b="blk."+std::to_string(l)+".";
        std::vector<int32_t> map(hparams.n_expert);
        const auto * mw=ml.get_weight((b+"ffn_exps_gu_type").c_str());
        if(!mw){auto * g=W(l,"ffn_gate_exps.weight");tiers[l].push_back({(int)g->type,g,W(l,"ffn_up_exps.weight"),W(l,"ffn_down_exps.weight"),{}});auto & v=tiers[l].back().local;for(uint32_t i=0;i<hparams.n_expert;++i)v.push_back(i);continue;}
        ml.files.at(mw->idx)->seek(mw->offs,SEEK_SET);ml.files.at(mw->idx)->read_raw(map.data(),map.size()*4);
        auto * dw=ml.get_weight((b+"ffn_exps_down_type").c_str());std::vector<int32_t> dm(map.size());
        if(!dw)throw std::runtime_error("missing V4.1 down type map");ml.files.at(dw->idx)->seek(dw->offs,SEEK_SET);ml.files.at(dw->idx)->read_raw(dm.data(),dm.size()*4);
        if(dm!=map)throw std::runtime_error("V4.1 currently requires matching gate/up/down expert classes");
        std::set<int> formats(map.begin(),map.end());
        for(int format:formats){if(format!=GGML_TYPE_Q3_K&&format!=GGML_TYPE_Q2_0&&format!=GGML_TYPE_PTQ1_0)throw std::runtime_error("unsupported expert format");std::string suffix=ggml_type_name((ggml_type)format);for(char & ch:suffix)ch=std::tolower((unsigned char)ch);experts e{format,W(l,"ffn_gate_exps."+suffix+".weight"),W(l,"ffn_up_exps."+suffix+".weight"),W(l,"ffn_down_exps."+suffix+".weight"),{}};int index=0;for(int v:map)e.local.push_back(v==format?index++:-1);tiers[l].push_back(std::move(e));}
    }
    const char * static_maps=std::getenv("DSV4F_STATIC_EXPERT_MAPS");
    if(static_maps && std::strcmp(static_maps,"0") && std::strcmp(static_maps,"1"))throw std::runtime_error("DSV4F_STATIC_EXPERT_MAPS must be 0 or 1");
    if(static_maps && !std::strcmp(static_maps,"1")) {
        // Place constants alongside each layer's routing operations, rather
        // than its CPU-resident expert weights. No scheduler input copies are
        // then needed, even after graph/phase/context changes.
        std::map<ggml_backend_buffer_type_t,std::vector<experts *>> groups;
        for(uint32_t l=0;l<hparams.n_layer();++l)
            for(auto & e:tiers[l])groups[ggml_backend_dev_buffer_type(dev_layer(l))].push_back(&e);
        size_t count=0,bytes=0,allocated=0;
        for(auto & group:groups) {
            auto buft=group.first;
            ggml_context_ptr ctx{ggml_init({group.second.size()*ggml_tensor_overhead(),nullptr,true})};
            if(!ctx)throw std::runtime_error("cannot create expert mapping constants");
            for(auto * e:group.second) {
                e->mapping=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,1,hparams.n_expert);
                ggml_format_name(e->mapping,"v41_static_expert_ids_%zu",count++);
                bytes+=ggml_nbytes(e->mapping);
            }
            allocated+=ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(),buft);
            ggml_backend_buffer_ptr buffer{hparams.no_alloc?ggml_backend_buft_alloc_buffer(buft,0):ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),buft)};
            if(!buffer)throw std::runtime_error("cannot allocate expert mapping constants");
            for(auto * e:group.second) {
                if(hparams.no_alloc)e->mapping->buffer=buffer.get();
                else ggml_backend_tensor_set(e->mapping,e->local.data(),0,ggml_nbytes(e->mapping));
            }
            add_constant_buffer(std::move(ctx),std::move(buffer));
        }
        std::fprintf(stderr,"[dsv4f-v41] static expert maps: %zu tensors, %zu bytes, %zu allocated bytes, %zu devices%s\n",count,bytes,allocated,groups.size(),hparams.no_alloc?" (no-alloc)":"");
    }
    if(!hparams.no_alloc && !engram_layers.empty()){const char * path=std::getenv("DSV4F_ENGRAM_DIR");if(!path)throw std::runtime_error("set DSV4F_ENGRAM_DIR to the original FP8 shards");engram=std::make_shared<dsv4f_engram>(*this,path);}
}

namespace {
int v41_cap(const llama_kv_cache_context * k,const llama_ubatch & u,int maximum) {
    int n=k->get_n_kv();if(u.pos)for(uint32_t i=0;i<u.n_tokens;++i)n=std::max(n,u.pos[i]+1);
    return std::min(maximum,std::max(256,(n+255)/256*256));
}
struct v41_input : llm_graph_input_i {
    const llama_model_deepseek41 & m;
    const llama_kv_cache_context * kc;
    int cap,nt,maximum,window;
    ggml_tensor * raw_slots,*raw_positions,*output_slots,*identity_pre;
    ggml_tensor * compact_raw_ids=nullptr,*compact_raw_mask=nullptr;
    struct pool {ggml_tensor * slots,*pos,*pair,*last;};
    pool comp[3]{};
    std::vector<ggml_tensor *> engr;
    std::unique_ptr<dsv4f_engram_async> engram_async;
    std::vector<std::pair<ggml_tensor*,const std::vector<int32_t>*>> maps;
    const bool profile=[] {const char * v=std::getenv("DSV4F_INPUT_PROFILE");return v && std::atoi(v)!=0;}();
    const bool engram_profile=[] {const char * v=std::getenv("DSV4F_ENGRAM_PROFILE");return v && std::atoi(v)!=0;}();
    v41_input(const llama_model_deepseek41 & model,const llama_kv_cache_context * k,int c,int n,int mx,int w):m(model),kc(k),cap(c),nt(n),maximum(mx),window(w){}
    bool can_reuse(const llm_graph_params & p) override {
        auto * k=dynamic_cast<const llama_kv_cache_context*>(p.mctx);
        if(!k || (int)p.ubatch.n_tokens!=nt || v41_cap(k,p.ubatch,maximum)!=cap)return false;kc=k;return true;
    }
    static void upload(ggml_tensor * t,const void * data,size_t bytes) { if(t->buffer)ggml_backend_tensor_set(t,data,0,bytes); }
    void set_input(const llama_ubatch * u) override {
        dsv4f_timeline::update_collection();
        dsv4f_timeline::scope span(nt==1?"v41 input/decode":"v41 input/prefill");
        const int64_t start=profile?ggml_time_us():0;
        if(!u->token || u->n_seqs_unq!=1)throw std::runtime_error("[dsv4f-v41] requires a single text sequence");
        auto * mem=dynamic_cast<llama_kv_cache_dsv41*>(kc->get_cache());
        std::vector<int64_t> slots(nt),positions(nt),outputs(nt);
        std::vector<float> pre((size_t)m.hc*nt,0.f);
        for(int t=0;t<nt;++t) {
            const int p=u->pos[t];
            if(p<0||p>=maximum || (t && p!=u->pos[0]+t))throw std::runtime_error("V4.1 requires consecutive positions within a batch");
            slots[t]=p%window;positions[t]=p;outputs[t]=t;pre[t*m.hc]=1.f;
        }
        mem->prepare(*u);
        upload(raw_slots,slots.data(),nt*8);upload(raw_positions,positions.data(),nt*8);
        upload(output_slots,outputs.data(),nt*8);upload(identity_pre,pre.data(),pre.size()*4);
        if(compact_raw_ids) {
            // Preserve the sparse gather's ascending physical ring order.
            const int count=std::min<int64_t>(u->pos[0]+1,compact_raw_ids->ne[0]);
            std::vector<int32_t> ids(compact_raw_ids->ne[0],0);
            std::vector<ggml_fp16_t> mask(ids.size(),ggml_fp32_to_fp16(-INFINITY));
            for(int i=0;i<count;++i)ids[i]=(u->pos[0]-count+1+i)%window;
            std::sort(ids.begin(),ids.begin()+count);
            std::fill(mask.begin(),mask.begin()+count,ggml_fp32_to_fp16(0.f));
            upload(compact_raw_ids,ids.data(),ids.size()*sizeof(int32_t));
            upload(compact_raw_mask,mask.data(),mask.size()*sizeof(ggml_fp16_t));
        }
        for(int r=1;r<=2;++r) {
            auto & c=comp[r];std::vector<int64_t> cs(nt);std::vector<int32_t> cp(nt),pair(2*nt);std::vector<float> last(nt);
            for(int t=0;t<nt;++t){int p=u->pos[t],nv=(p+1)/r;cs[t]=(p+1)%r?((maximum+r-1)/r):p/r;cp[t]=p/r*r;pair[2*t]=(p/2*2)%window;pair[2*t+1]=(p/2*2+1)%window;last[t]=(nv-1)/m.candidate_size;}
            upload(c.slots,cs.data(),nt*8);upload(c.pos,cp.data(),nt*4);upload(c.pair,pair.data(),2*nt*4);upload(c.last,last.data(),nt*4);
        }
        const int64_t map_start=profile?ggml_time_us():0;
        size_t map_bytes=0;
        for(auto & v:maps){const size_t bytes=v.second->size()*4;upload(v.first,v.second->data(),bytes);map_bytes+=bytes;}
        const int64_t map_end=profile?ggml_time_us():0;
        std::vector<llama_token> prev;kc->get_prev_tokens(*u,m.engram_ngram-1,prev);
        std::vector<int32_t> history((size_t)nt*m.engram_ngram);
        for(int t=0;t<nt;++t){history[t*m.engram_ngram]=u->token[t];for(int j=1;j<m.engram_ngram;++j)history[t*m.engram_ngram+j]=prev[t*(m.engram_ngram-1)+(m.engram_ngram-1-j)];}
        for(size_t s=0;s<engr.size();++s){if(engram_async && engram_async->enabled(s))continue;std::vector<float> rows;m.engram->read(s,history,nt,rows);const int64_t at=engram_profile?ggml_time_us():0;upload(engr[s],rows.data(),rows.size()*4);if(engram_profile)std::fprintf(stderr,"[dsv4f-engram-upload] slot=%zu tokens=%d bytes=%zu host_set_us=%lld\n",s,nt,rows.size()*4,(long long)(ggml_time_us()-at));}
        if(engram_async){
            std::shared_ptr<dsv4f_engram_prefetch::job> pending;
            bool all_async=true;for(size_t s=0;s<engr.size();++s)all_async&=engram_async->enabled(s);
            if(all_async && mem->engram_prefetch)pending=mem->engram_prefetch->begin(history,nt,u->pos[0],u->seq_id[0][0]);
            engram_async->start(history,std::move(pending));
        }
        // Explicit profiling must remain visible at llama-server's default
        // verbosity, which filters the library's INFO callback.
        if(profile)std::fprintf(stderr,"[dsv4f-input] tokens=%d maps=%zu map_bytes=%zu maps_us=%lld total_us=%lld\n",nt,maps.size(),map_bytes,(long long)(map_end-map_start),(long long)(ggml_time_us()-start));
    }
};
}
llama_model_deepseek41::graph::graph(const llama_model_deepseek41 & m,const llm_graph_params & p):llm_graph_context(p) {
    if(loras && !loras->empty())throw std::runtime_error("[dsv4f-v41] LoRA is not integrated");
    auto * kc=dynamic_cast<const llama_kv_cache_context*>(mctx);if(!kc)throw std::runtime_error("V4.1 memory context mismatch");
    auto * mem=dynamic_cast<llama_kv_cache_dsv41*>(kc->get_cache());if(!mem)throw std::runtime_error("V4.1 cache mismatch");
    const int D=n_embd,H=m.hc,N=n_tokens,DH=hparams.n_embd_head_k(),NH=hparams.n_head(),SWA=hparams.n_swa;
    const bool compact_decode=N==1 && DH==512 && [] {const char * v=std::getenv("DSV4F_V41_COMPACT_KV");return !v || std::strcmp(v,"0");}();
    const bool overlap=N<=4 && [] {const char * v=std::getenv("DSV4F_V41_MOE_OVERLAP");return !v || std::strcmp(v,"0");}();
    const bool pipeline=overlap && [] {const char * v=std::getenv("DSV4F_V41_MOE_PIPELINE");return !v || std::strcmp(v,"0");}();
    const int engram_overlap=[&] {const char * v=std::getenv(N<=4?"DSV4F_ENGRAM_OVERLAP":"DSV4F_ENGRAM_PREFILL_OVERLAP");return v?std::atoi(v):0;}();
    const bool route_fused=N<=4 && [] {const char * v=std::getenv("DSV4F_V41_ROUTE_FUSED");return !v || std::strcmp(v,"0");}();
    const bool route_publish_fused=route_fused && [] {const char * v=std::getenv("DSV4F_V41_ROUTE_PUBLISH_FUSED");return v && !std::strcmp(v,"1");}();
    const int cap=v41_cap(kc,ubatch,mem->capacity);
    auto own=std::make_unique<v41_input>(m,kc,cap,N,mem->capacity,mem->window_capacity);auto * in=own.get();
    auto input=[&](ggml_type ty,int a,int b,const char * name){auto * t=ggml_new_tensor_2d(ctx0,ty,a,b);ggml_set_input(t);ggml_set_name(t,name);return t;};
    in->raw_slots=input(GGML_TYPE_I64,N,1,"v41_window_slots");
    in->raw_positions=input(GGML_TYPE_I64,N,1,"v41_absolute_positions");
    in->output_slots=input(GGML_TYPE_I64,N,1,"v41_output_slots");
    in->identity_pre=input(GGML_TYPE_F32,H,N,"v41_identity_pre");
    for(int r=1;r<=2;++r){auto & c=in->comp[r];c.slots=input(GGML_TYPE_I64,N,1,"v41_comp_slots");c.pos=input(GGML_TYPE_I32,N,1,"v41_comp_pos");c.pair=input(GGML_TYPE_I32,2*N,1,"v41_pool_rows");c.last=input(GGML_TYPE_F32,1,N,"v41_last_block");}
    for(size_t s=0;s<m.engram_layers.size();++s)in->engr.push_back(input(GGML_TYPE_F32,(m.engram_ngram-1)*m.engram_heads*m.engram_dim,N,"v41_engram_rows"));
    // A full-size empty leaf is allocated for the entire graph, even though
    // FILL never reads it. Materialize constants from a scalar instead so the
    // allocator can reuse each layer's mask storage after its last consumer.
    auto full=[&](int a,int b,int c,float value) {
        auto * scalar=ggml_fill(ctx0,ggml_new_tensor_1d(ctx0,GGML_TYPE_F32,1),value);
        return ggml_repeat_4d(ctx0,scalar,a,b,c,1);
    };
    auto * emb=build_inp_embd(m.tok_embd);auto * pos=build_inp_pos();auto * out_ids=build_inp_out_ids();
    auto norm=[&](ggml_tensor * t,ggml_tensor * w){return ggml_mul(ctx0,ggml_rms_norm(ctx0,t,m.eps),w);};
    auto flat=[&](ggml_tensor * t,int a,int b){return ggml_cont_2d(ctx0,t,a,b);};
    auto fq=[&](ggml_tensor * t,int mode){return ggml_dsv41_fake_quant(ctx0,ggml_cont(ctx0,t),mode);};
    auto rope=[&](ggml_tensor * t,ggml_tensor * ps,int r,bool back=false){float scale=r?1.f/m.rope_factor:1.f;float att=r?1.f/(1.f+0.1f*logf(1.f/scale)):1.f;auto * o=back?ggml_rope_ext_back(ctx0,t,ps,nullptr,hparams.n_rot(),GGML_ROPE_TYPE_NORMAL,r?m.orig_ctx:0,r?m.compress_base:m.rope_base,scale,r?1.f:0.f,att,m.beta_fast,m.beta_slow):ggml_rope_ext(ctx0,t,ps,nullptr,hparams.n_rot(),GGML_ROPE_TYPE_NORMAL,r?m.orig_ctx:0,r?m.compress_base:m.rope_base,scale,r?1.f:0.f,att,m.beta_fast,m.beta_slow);return ggml_rope_set_offset(o,t->ne[0]-hparams.n_rot());};
    auto mixes=[&](ggml_tensor * x,int l,const char * site,ggml_tensor ** pre,ggml_tensor ** post,ggml_tensor ** comb){std::string b=std::string("hc_")+site;auto * f=ggml_mul_mat(ctx0,m.W(l,b+"_fn.weight"),ggml_rms_norm(ctx0,ggml_reshape_2d(ctx0,x,D*H,N),m.eps));ggml_mul_mat_set_prec(f,GGML_PREC_F32);auto * v=ggml_dsv4_hc_mix(ctx0,f,m.W(l,b+"_scale.weight"),m.W(l,b+"_base.weight"),m.hc_eps,hparams.dsv4_hc_sinkhorn_iters);*pre=ggml_view_2d(ctx0,v,H,N,v->nb[1],0);*post=ggml_view_2d(ctx0,v,H,N,v->nb[1],H*4);*comb=ggml_view_3d(ctx0,v,H,H,N,H*4,v->nb[1],2*H*4);};
    ggml_tensor * x=ggml_repeat_4d(ctx0,ggml_reshape_3d(ctx0,emb,D,1,N),D,H,N,1),*previous=in->identity_pre;
    // Query tiles bound indexer scores, sorting and attention masks. Dense
    // projections and expert prefill still run at the requested full ubatch.
    constexpr int query_tile=128;
    std::map<std::pair<int,int>,std::pair<ggml_tensor*,ggml_tensor*>> selected;
    std::map<int,ggml_tensor*> candidates;
    std::map<int,ggml_tensor*> comps,indices;
    ggml_tensor * scratch=nullptr;
    for(int l=0;l<n_layer;++l) {
        if((size_t)l<cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[l]) {
            // DSpark is trained on the uniform HC mean ENTERING each tapped layer.
            auto * mean=ggml_mean(ctx0,ggml_cont(ctx0,ggml_permute(ctx0,x,1,0,2,3)));
            res->t_layer_inp[l]=ggml_reshape_2d(ctx0,mean,D,N);
            cb(res->t_layer_inp[l],"layer_inp",l);ggml_build_forward_expand(gf,res->t_layer_inp[l]);
        }
        const int r=m.ratios[l];auto & cache=mem->layers41[l];
        // Match the existing HIP sparse path's packed KV layout and tile size.
        // CPU and dense attention have different FP16 reduction partitions;
        // leave those paths intact rather than changing their rounding.
        auto * kv_dev=cache.raw->buffer?ggml_backend_buft_get_device(ggml_backend_buffer_get_type(cache.raw->buffer)):nullptr;
        const bool route_first=pipeline && kv_dev && !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm") &&
            [] {const char * v=std::getenv("DSV4F_V41_ROUTE_FIRST");return !v || !std::strcmp(v,"1");}();

        const int dense_nc=r?((cap+r-1)/r+7)/8*8:0;
        const int sparse_rows=GGML_PAD(SWA+std::min(dense_nc,m.idx_top),256);
        const char * sparse_env=std::getenv("GGML_HIP_SPARSE_FA");
        const bool compact_kv=compact_decode && kv_dev &&
            !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm") &&
            (!sparse_env || std::strcmp(sparse_env,"0")) &&
            GGML_PAD(int(mem->window_capacity)+dense_nc,256)>=std::max(2048,2*sparse_rows);
        auto * direct_comp=r?mem->layers41.at(m.kv_sources[l]).comp:nullptr;
        // A cross-device compressed cache must keep the existing gather on its
        // owner; moving the whole cache to this custom operation would regress.
        const bool direct_kv=compact_kv && cache.raw->type==GGML_TYPE_Q8_0 &&
            (!r || (direct_comp && direct_comp->type==GGML_TYPE_Q8_0 && direct_comp->buffer &&
             ggml_backend_buft_get_device(ggml_backend_buffer_get_type(direct_comp->buffer))==kv_dev)) &&
            SWA+std::min(dense_nc,m.idx_top)<=1024 && [] {const char * v=std::getenv("DSV4F_V41_DIRECT_KV");return !v || std::strcmp(v,"0");}();
        if(compact_kv && !in->compact_raw_ids) {
            const int rows=std::min(SWA,int(mem->window_capacity));
            in->compact_raw_ids=input(GGML_TYPE_I32,rows,1,"v41_compact_raw_ids");
            in->compact_raw_mask=input(GGML_TYPE_F16,rows,1,"v41_compact_raw_mask");
        }
        auto eit=std::find(m.engram_layers.begin(),m.engram_layers.end(),l);
        if(eit!=m.engram_layers.end() && engram_overlap && (engram_overlap==1 || eit!=m.engram_layers.begin()) && kv_dev &&
           !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm")) {
            if(!in->engram_async)in->engram_async=std::make_unique<dsv4f_engram_async>(*m.engram,N,
                (m.engram_ngram-1)*m.engram_heads*m.engram_dim,m.engram_layers.size());
            const int slot=eit-m.engram_layers.begin();
            in->engr[slot]=in->engram_async->tensor(ctx0,x,slot,kv_dev);
        }
        if(eit!=m.engram_layers.end()) {
            auto * kv=ggml_mul_mat(ctx0,m.W(l,"engram_kv.weight"),in->engr[eit-m.engram_layers.begin()]);
            auto * key=ggml_view_3d(ctx0,kv,D,H,N,D*4,kv->nb[1],0);
            auto * val=ggml_view_3d(ctx0,kv,D,1,N,D*4,kv->nb[1],D*H*4);
            // Native UD files store the product of the two RMSNorm scales.
            auto * scales=m.weights.count("blk."+std::to_string(l)+".engram_gate.weight")?
                m.W(l,"engram_gate.weight"):ggml_mul(ctx0,m.W(l,"engram_q_norm.weight"),m.W(l,"engram_k_norm.weight"));
            auto * dot=ggml_scale(ctx0,ggml_sum_rows(ctx0,ggml_mul(ctx0,ggml_mul(ctx0,ggml_rms_norm(ctx0,x,m.eps),scales),ggml_rms_norm(ctx0,ggml_cont(ctx0,key),m.eps))),1.f/sqrtf(D));
            auto * gate=ggml_sigmoid(ctx0,ggml_mul(ctx0,ggml_sgn(ctx0,dot),ggml_sqrt(ctx0,ggml_clamp(ctx0,ggml_abs(ctx0,dot),1e-6f,INFINITY))));
            x=ggml_add(ctx0,x,ggml_mul(ctx0,ggml_repeat(ctx0,ggml_cont(ctx0,val),x),gate));
        }
        ggml_tensor * ap,*ao,*ac;mixes(x,l,"attn",&ap,&ao,&ac);
        auto * xn=norm(ggml_dsv4_hc_pre(ctx0,x,previous),m.W(l,"attn_norm.weight"));
        auto * qr=norm(ggml_mul_mat(ctx0,m.W(l,"attn_q_a.weight"),xn),m.W(l,"attn_q_a_norm.weight"));
        auto * q=rope(ggml_reshape_3d(ctx0,ggml_mul_mat(ctx0,m.W(l,"attn_q_b.weight"),qr),DH,NH,N),pos,r);
        auto * kv=fq(flat(rope(ggml_reshape_3d(ctx0,norm(ggml_mul_mat(ctx0,m.W(l,"attn_kv.weight"),xn),m.W(l,"attn_kv_a_norm.weight")),DH,1,N),pos,r),DH,N),0);
        auto * raw=ggml_set_rows(ctx0,cache.raw,kv,in->raw_slots);ggml_build_forward_expand(gf,raw);
        auto archive=[&](ggml_dsv41_history & history,ggml_tensor * t){auto * op=ggml_dsv41_archive(ctx0,&history,t,in->raw_slots,in->raw_positions);if(op)ggml_build_forward_expand(gf,op);};
        archive(cache.raw_history,raw);
        ggml_tensor * keys=direct_kv?nullptr:compact_kv?ggml_get_rows(ctx0,raw,in->compact_raw_ids):ggml_cast(ctx0,raw,GGML_TYPE_F32);
        ggml_tensor * compressed=nullptr;
        int nc=0,K=0;
        if(r) {
            if(m.kv_sources[l]==l) {
                auto * latent=ggml_mul_mat(ctx0,m.W(l,"attn_compressor_kv.weight"),xn);
                if(r==2){auto * score=ggml_mul_mat(ctx0,m.W(l,"attn_compressor_gate.weight"),xn);ggml_mul_mat_set_prec(latent,GGML_PREC_F32);ggml_mul_mat_set_prec(score,GGML_PREC_F32);auto * a=ggml_set_rows(ctx0,cache.state,latent,in->raw_slots);auto * b=ggml_set_rows(ctx0,cache.score,score,in->raw_slots);ggml_build_forward_expand(gf,a);ggml_build_forward_expand(gf,b);archive(cache.state_history,a);archive(cache.score_history,b);auto * av=ggml_cont(ctx0,ggml_permute(ctx0,ggml_reshape_3d(ctx0,ggml_get_rows(ctx0,a,in->comp[r].pair),DH,2,N),1,0,2,3));auto * bv=ggml_cont(ctx0,ggml_permute(ctx0,ggml_reshape_3d(ctx0,ggml_get_rows(ctx0,b,in->comp[r].pair),DH,2,N),1,0,2,3));latent=ggml_reshape_2d(ctx0,ggml_sum_rows(ctx0,ggml_mul(ctx0,av,ggml_soft_max(ctx0,bv))),DH,N);}
                latent=norm(latent,m.W(l,"attn_compressor_norm.weight"));
                auto * ik=norm(ggml_mul_mat(ctx0,m.W(l,"indexer.attn_k.weight"),latent),m.W(l,"indexer.k_norm.weight"));
                ik=fq(flat(rope(ggml_reshape_3d(ctx0,ik,m.idx_dim,1,N),in->comp[r].pos,r),m.idx_dim,N),2);
                indices[l]=ggml_set_rows(ctx0,cache.idx,ik,in->comp[r].slots);comps[l]=ggml_set_rows(ctx0,cache.comp,fq(flat(rope(ggml_reshape_3d(ctx0,latent,DH,1,N),in->comp[r].pos,r),DH,N),1),in->comp[r].slots);ggml_build_forward_expand(gf,indices[l]);ggml_build_forward_expand(gf,comps[l]);
            }
            compressed=comps.at(m.kv_sources[l]);
            nc=((cap+r-1)/r+7)/8*8;K=std::min(nc,m.idx_top);
            if(!compact_kv) {
                auto * k=ggml_cast(ctx0,ggml_view_2d(ctx0,compressed,DH,nc,compressed->nb[1],0),GGML_TYPE_F32);
                keys=ggml_concat(ctx0,keys,k,1);
            }
        }
        int M=keys?keys->ne[1]:0,pad=(256-M%256)%256;
        ggml_tensor * k=nullptr;
        if(!compact_kv) {
            if(pad){keys=ggml_pad(ctx0,keys,0,pad,0,0);M+=pad;}
            k=ggml_cast(ctx0,ggml_reshape_4d(ctx0,keys,DH,M,1,1),GGML_TYPE_F16);
        }
        ggml_tensor * result=nullptr;
        if(N>query_tile)result=ggml_reshape_2d(ctx0,full(DH*NH,N,1,0.f),DH*NH,N);
        for(int first=0;first<N;first+=query_tile) {
            const int T=std::min(query_tile,N-first);
            auto * positions=ggml_view_1d(ctx0,in->raw_positions,T,first*sizeof(int64_t));
            ggml_tensor * direct_sel=nullptr,* direct_sm=nullptr;
            auto * mask=compact_kv?in->compact_raw_mask:ggml_dsv41_mask(ctx0,positions,mem->window_capacity,0,SWA);
            if(r) {
                auto * visible=ggml_dsv41_mask(ctx0,positions,nc,r,0);
                ggml_tensor * sel,*sm;
                if(m.index_sources[l]==l) {
                    auto * qt=ggml_view_2d(ctx0,qr,qr->ne[0],T,qr->nb[1],first*qr->nb[1]);
                    auto * xt=ggml_view_2d(ctx0,xn,D,T,xn->nb[1],first*xn->nb[1]);
                    auto * pt=ggml_view_1d(ctx0,pos,T,first*sizeof(int32_t));
                    auto * iq=fq(rope(ggml_reshape_3d(ctx0,ggml_mul_mat(ctx0,m.W(l,"indexer.attn_q_b.weight"),qt),m.idx_dim,m.idx_heads,T),pt,r),2);
                    auto * iw=ggml_scale(ctx0,ggml_mul_mat(ctx0,m.W(l,"indexer.proj.weight"),xt),1.f/sqrtf(m.idx_dim*m.idx_heads));
                    auto * index=indices.at(m.kv_sources[l]);
                    auto * ik=ggml_view_3d(ctx0,index,m.idx_dim,1,nc,index->nb[1],index->nb[1],0);
                    auto * score=ggml_reshape_2d(ctx0,ggml_lightning_indexer(ctx0,iq,ik,iw,visible),nc,T);
                    if(r==1 && l>m.candidate_layer && candidates.count(first)) {
                        const int nb=nc/m.candidate_size;auto * cb=candidates.at(first);
                        auto * mask0=full(m.candidate_size,nb,T,-INFINITY);
                        mask0=ggml_set_rows(ctx0,mask0,full(m.candidate_size,cb->ne[0],T,0.f),cb);
                        score=ggml_add(ctx0,score,ggml_view_2d(ctx0,mask0,nc,T,mask0->nb[2],0));
                    }
                    if(l==m.candidate_layer) {
                        const int nb=nc/m.candidate_size;
                        auto * bm=ggml_pool_2d(ctx0,score,GGML_OP_POOL_MAX,m.candidate_size,1,m.candidate_size,1,0,0);
                        auto * ramp=ggml_arange(ctx0,0,nb,1);
                        auto * last=ggml_view_2d(ctx0,in->comp[r].last,1,T,sizeof(float),first*sizeof(float));
                        auto * eq=ggml_step(ctx0,ggml_scale_bias(ctx0,ggml_abs(ctx0,ggml_sub(ctx0,ggml_repeat(ctx0,ramp,bm),last)),-1.f,0.5f));
                        bm=ggml_add(ctx0,bm,ggml_scale(ctx0,eq,1e30f));
                        candidates[first]=ggml_cont(ctx0,ggml_top_k(ctx0,bm,std::min(nb,m.candidate_blocks)));
                    }
                    sel=ggml_cont(ctx0,ggml_top_k(ctx0,score,K));
                    sel=ggml_reshape_2d(ctx0,ggml_get_rows(ctx0,ggml_reshape_3d(ctx0,sel,1,K,T),ggml_argsort(ctx0,ggml_cast(ctx0,sel,GGML_TYPE_F32),GGML_SORT_ORDER_ASC)),K,T);
                    sm=ggml_reshape_2d(ctx0,ggml_get_rows(ctx0,ggml_reshape_3d(ctx0,visible,1,nc,T),sel),K,T);
                    selected[{l,first}]={sel,sm};
                } else {
                    const auto & pair=selected.at({m.index_sources[l],first});sel=pair.first;sm=pair.second;
                }
                if(direct_kv) {direct_sel=sel;direct_sm=sm;}
                else if(compact_kv) {
                    keys=ggml_concat(ctx0,keys,ggml_get_rows(ctx0,compressed,sel),1);
                    mask=ggml_concat(ctx0,mask,ggml_cast(ctx0,sm,GGML_TYPE_F16),0);
                } else {
                    auto * sparse=ggml_set_rows(ctx0,full(1,nc,T,-INFINITY),ggml_reshape_3d(ctx0,ggml_cont(ctx0,sm),1,K,T),sel);
                    mask=ggml_concat(ctx0,mask,ggml_cast(ctx0,ggml_reshape_2d(ctx0,sparse,nc,T),GGML_TYPE_F16),0);
                }
            }
            if(direct_kv) {
                ggml_dsv41_direct_kv(ctx0,raw,in->compact_raw_ids,in->compact_raw_mask,compressed,direct_sel,direct_sm,keys,mask);
                M=keys->ne[1];pad=0;
                k=ggml_reshape_4d(ctx0,keys,DH,M,1,1);
            } else if(compact_kv) {
                ggml_dsv41_pack_kv(ctx0,keys,mask);
                M=keys->ne[1];pad=(256-M%256)%256;
                if(pad){keys=ggml_pad(ctx0,keys,0,pad,0,0);M+=pad;}
                k=ggml_cast(ctx0,ggml_reshape_4d(ctx0,keys,DH,M,1,1),GGML_TYPE_F16);
            }
            if(pad)mask=ggml_concat(ctx0,mask,ggml_cast(ctx0,ggml_reshape_2d(ctx0,full(pad,T,1,-INFINITY),pad,T),GGML_TYPE_F16),0);
            auto * qt=ggml_view_3d(ctx0,q,DH,NH,T,q->nb[1],q->nb[2],first*q->nb[2]);
            auto * q4=ggml_permute(ctx0,ggml_reshape_4d(ctx0,qt,DH,NH,T,1),0,2,1,3);
            auto * tile=ggml_flash_attn_ext(ctx0,q4,k,k,ggml_reshape_4d(ctx0,mask,M,T,1,1),1.f/sqrtf(DH),0.f,0.f);
            ggml_flash_attn_ext_set_n_kv_max(tile,compact_kv?0:SWA+K);ggml_flash_attn_ext_add_sinks(tile,m.W(l,"attn_sinks.weight"));
            if(N<=query_tile)result=ggml_reshape_2d(ctx0,tile,DH*NH,N);
            else result=ggml_set_rows(ctx0,result,ggml_reshape_2d(ctx0,tile,DH*NH,T),ggml_view_1d(ctx0,in->output_slots,T,first*sizeof(int64_t)));
        }
        auto * o=ggml_reshape_3d(ctx0,result,DH,NH,N);
        o=rope(ggml_reshape_3d(ctx0,o,DH,NH,N),pos,r,true);o=ggml_permute(ctx0,ggml_reshape_3d(ctx0,o,NH*DH/m.og,m.og,N),0,2,1,3);
        auto * wa=ggml_reshape_3d(ctx0,m.W(l,"attn_output_a.weight"),NH*DH/m.og,m.ol,m.og);auto * oa=ggml_permute(ctx0,ggml_mul_mat(ctx0,wa,o),0,2,1,3);
        auto * att=ggml_mul_mat(ctx0,m.W(l,"attn_output_b.weight"),flat(oa,m.ol*m.og,N));
        x=ggml_dsv4_hc_post(ctx0,att,x,ao,ac);
        ggml_tensor * fp,*fo,*fc;mixes(x,l,"ffn",&fp,&fo,&fc);auto * fn=norm(ggml_dsv4_hc_pre(ctx0,x,ap),m.W(l,"ffn_norm.weight"));
        // OpenBMB LES predictor: current post-attention normalized state times
        // the NEXT layer's router, expanded before the current routed FFN.
        // Raw logits reproduce its ranking rule; native applies DeepSeek's
        // sqrt-softplus + selection bias. Neither changes the actual router.
        const char * lookahead=std::getenv("DSV4F_EXPERT_LOOKAHEAD");
        if(N==1 && l+1<n_layer && kv_dev && lookahead &&
           (!std::strcmp(lookahead,"3") || !std::strcmp(lookahead,"4") || !std::strcmp(lookahead,"5")) &&
           !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm")) {
            int depth=8;if(const char * value=std::getenv("DSV4F_EXPERT_LOOKAHEAD_DEPTH"))depth=std::atoi(value);
            GGML_ASSERT(depth>=1 && depth<=16);depth=std::min(depth,int(hparams.n_expert));
            auto * logits=ggml_mul_mat(ctx0,m.W(l+1,"ffn_gate_inp.weight"),fn);ggml_mul_mat_set_prec(logits,GGML_PREC_F32);
            ggml_set_name(logits,"v41_early_router_logits");
            ggml_tensor * predicted=nullptr;
            const char * score=std::getenv("DSV4F_EXPERT_LOOKAHEAD_SCORE");
            if(score && !std::strcmp(score,"native")) {
                ggml_tensor * unused;dsv4f_router(ctx0,logits,m.W(l+1,"exp_probs_b.bias"),depth,false,1.f,&predicted,&unused);
            } else predicted=ggml_cont(ctx0,ggml_top_k(ctx0,logits,depth));
            for(auto & e:m.tiers[l+1]) {
                auto * mapping=e.mapping;
                if(!mapping){mapping=input(GGML_TYPE_I32,1,hparams.n_expert,"v41_early_expert_ids");in->maps.push_back({mapping,&e.local});}
                ggml_build_forward_expand(gf,dsv4f_expert_prefetch(ctx0,fn,predicted,mapping,e.gate,e.up,e.down,l+1));
            }
        }
        auto * router=ggml_mul_mat(ctx0,m.W(l,"ffn_gate_inp.weight"),fn);ggml_mul_mat_set_prec(router,GGML_PREC_F32);
        ggml_tensor * ids,*probs;
        const char * router_fused=std::getenv("DSV4F_V41_ROUTER_FUSED");
        if(N<=4 && hparams.n_expert<=512 && hparams.n_expert_used(l)<=32 && kv_dev &&
           !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm") &&
           (!router_fused || std::strcmp(router_fused,"0"))) {
            dsv4f_router(ctx0,router,m.W(l,"exp_probs_b.bias"),hparams.n_expert_used(l),m.norm_topk,m.scaling,&ids,&probs);
        } else {
            auto * scores=ggml_sqrt(ctx0,ggml_softplus(ctx0,router));ids=ggml_cont(ctx0,ggml_top_k(ctx0,ggml_add(ctx0,scores,m.W(l,"exp_probs_b.bias")),hparams.n_expert_used(l)));
            probs=ggml_reshape_3d(ctx0,ggml_get_rows(ctx0,ggml_reshape_3d(ctx0,scores,1,hparams.n_expert,N),ids),1,hparams.n_expert_used(l),N);if(m.norm_topk){auto * pw=ggml_reshape_2d(ctx0,probs,hparams.n_expert_used(l),N);probs=ggml_reshape_3d(ctx0,ggml_div(ctx0,pw,ggml_scale_bias(ctx0,ggml_sum_rows(ctx0,pw),1.f,1e-20f)),1,hparams.n_expert_used(l),N);}probs=ggml_scale(ctx0,probs,m.scaling);
        }
        auto * gt=ggml_mul_mat(ctx0,m.W(l,"ffn_gate_shexp.weight"),fn);
        auto * up=ggml_mul_mat(ctx0,m.W(l,"ffn_up_shexp.weight"),fn);
        const char * shared_fused=std::getenv("DSV4F_V41_SHARED_FUSED");
        const bool fuse_shared=N==1 && kv_dev &&
            !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm") &&
            m.W(l,"ffn_gate_shexp.weight")->type==GGML_TYPE_Q8_0 &&
            m.W(l,"ffn_up_shexp.weight")->type==GGML_TYPE_Q8_0 &&
            (!shared_fused || std::strcmp(shared_fused,"0"));
        auto * hidden=fuse_shared?ggml_swiglu_clamp(ctx0,gt,up,m.clamp):
            ggml_swiglu_split(ctx0,ggml_clamp(ctx0,gt,-INFINITY,m.clamp),ggml_clamp(ctx0,up,-m.clamp,m.clamp));
        ggml_tensor * sum=ggml_mul_mat(ctx0,m.W(l,"ffn_down_shexp.weight"),hidden);
        std::vector<ggml_tensor *> pending;
        for(auto & e:m.tiers[l]){auto * mapping=e.mapping;if(!mapping){mapping=input(GGML_TYPE_I32,1,hparams.n_expert,"v41_local_expert_ids");in->maps.push_back({mapping,&e.local});}
            ggml_tensor * weight,*lid;
            const bool publish_mapped=route_publish_fused && route_first;
            if(publish_mapped) {lid=ids;weight=probs;}
            else if(route_fused && kv_dev && !std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(kv_dev)),"ROCm")) {
                dsv4f_route_plan(ctx0,mapping,ids,probs,&lid,&weight);
            } else {
                auto * local=ggml_get_rows(ctx0,ggml_repeat_4d(ctx0,ggml_cast(ctx0,mapping,GGML_TYPE_F32),1,hparams.n_expert,N,1),ids);
                weight=ggml_mul(ctx0,probs,ggml_step(ctx0,ggml_scale_bias(ctx0,local,1.f,0.5f)));
                lid=ggml_reshape_2d(ctx0,ggml_cast(ctx0,ggml_clamp(ctx0,local,0.f,INFINITY),GGML_TYPE_I32),hparams.n_expert_used(l),N);
            }
            auto * act=fn;
            if(N>4 && N<32){act=ggml_pad(ctx0,fn,0,32-N,0,0);lid=ggml_cast(ctx0,ggml_pad(ctx0,ggml_cast(ctx0,lid,GGML_TYPE_F32),0,32-N,0,0),GGML_TYPE_I32);weight=ggml_pad(ctx0,weight,0,0,32-N,0);}
            auto * routed=dsv4f_moe(ctx0,ggml_cont(ctx0,act),lid,weight,e.gate,e.up,e.down,m.clamp,&scratch);if(N>4&&N<32)routed=ggml_view_2d(ctx0,routed,D,N,routed->nb[1],0);
            if(pipeline) {if(route_first)ggml_build_forward_expand(gf,dsv4f_moe_prepare(ctx0,routed,publish_mapped?mapping:nullptr));else dsv4f_moe_submit(gf,routed);pending.push_back(routed);}
            else sum=overlap && &e==&m.tiers[l].front()?dsv4f_moe_join(ctx0,routed,sum):ggml_add(ctx0,sum,routed);
        }
        if(route_first)for(auto * routed:pending)dsv4f_moe_submit(gf,routed);
        // All formats' GPU work and shared experts run before waiting for CPU
        // answers. Retain the preceding shared + tier0 + tier1 + tier2 sum order.
        for(auto * routed:pending)sum=dsv4f_moe_join(ctx0,routed,sum);
        x=ggml_dsv4_hc_post(ctx0,sum,x,fo,fc);previous=fp;cb(x,"layer_out",l);
    }
    auto * h=norm(ggml_dsv4_hc_pre(ctx0,x,previous),m.output_norm);h=ggml_get_rows(ctx0,h,out_ids);res->t_embd=h;res->t_logits=ggml_mul_mat(ctx0,m.output,h);cb(res->t_logits,"result_output",-1);ggml_build_forward_expand(gf,res->t_logits);
    res->add_input(std::move(own));
}
std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & p) const {return std::make_unique<graph>(*this,p);}
