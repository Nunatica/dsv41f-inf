#include "gguf-compat.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int key(gguf_context * ctx,const std::string & name) {
    const int id=gguf_find_key(ctx,name.c_str());
    if(id<0)throw std::runtime_error("native V4.1 metadata missing "+name);
    return id;
}
uint64_t number(gguf_context * ctx,const std::string & name) {
    const int id=key(ctx,name);
    switch(gguf_get_kv_type(ctx,id)) {
        case GGUF_TYPE_UINT32:return gguf_get_val_u32(ctx,id);
        case GGUF_TYPE_UINT64:return gguf_get_val_u64(ctx,id);
        default:throw std::runtime_error("native V4.1 metadata must be unsigned: "+name);
    }
}
void alias(gguf_context * ctx,const char * dest,const char * source,size_t count=SIZE_MAX) {
    const std::string from=std::string("deepseek41.")+source,to=std::string("deepseek41.")+dest;
    const int id=key(ctx,from);
    if(gguf_find_key(ctx,to.c_str())>=0)throw std::runtime_error("native V4.1 alias conflicts: "+to);
    switch(gguf_get_kv_type(ctx,id)) {
        case GGUF_TYPE_UINT32:gguf_set_val_u32(ctx,to.c_str(),gguf_get_val_u32(ctx,id));break;
        case GGUF_TYPE_INT32:gguf_set_val_i32(ctx,to.c_str(),gguf_get_val_i32(ctx,id));break;
        case GGUF_TYPE_FLOAT32:gguf_set_val_f32(ctx,to.c_str(),gguf_get_val_f32(ctx,id));break;
        case GGUF_TYPE_BOOL:gguf_set_val_bool(ctx,to.c_str(),gguf_get_val_bool(ctx,id));break;
        case GGUF_TYPE_ARRAY: {
            const auto type=gguf_get_arr_type(ctx,id);
            if(type!=GGUF_TYPE_UINT32 && type!=GGUF_TYPE_INT32 && type!=GGUF_TYPE_UINT64)
                throw std::runtime_error("unsupported native V4.1 array: "+from);
            const size_t n=count==SIZE_MAX?gguf_get_arr_n(ctx,id):count;
            if(n>gguf_get_arr_n(ctx,id))throw std::runtime_error("short native V4.1 array: "+from);
            const size_t bytes=n*(type==GGUF_TYPE_UINT64?8:4);
            std::vector<unsigned char> data(bytes);std::memcpy(data.data(),gguf_get_arr_data(ctx,id),bytes);
            gguf_set_arr_data(ctx,to.c_str(),type,data.data(),n);break;
        }
        default:throw std::runtime_error("unsupported native V4.1 alias type: "+from);
    }
}
}
void dsv4f_native_metadata(gguf_context * ctx) {
    const int arch=key(ctx,"general.architecture");
    if(std::strcmp(gguf_get_val_str(ctx,arch),"deepseek41"))throw std::runtime_error("Q4_K_XL filename requires deepseek41 architecture");
    const auto layers=number(ctx,"deepseek41.block_count");
    if(!layers || layers>512)throw std::runtime_error("invalid native V4.1 layer count");
    const std::pair<const char *,const char *> names[]={
        {"hidden_size","embedding_length"},{"max_position_embeddings","context_length"},
        {"num_hidden_layers","block_count"},{"num_attention_heads","attention.head_count"},
        {"num_key_value_heads","attention.head_count_kv"},{"head_dim","attention.key_length"},
        {"qk_rope_head_dim","rope.dimension_count"},{"q_lora_rank","attention.q_lora_rank"},
        {"o_lora_rank","attention.output_lora_rank"},{"o_groups","attention.output_group_count"},
        {"n_routed_experts","expert_count"},{"num_experts_per_tok","expert_used_count"},
        {"moe_intermediate_size","expert_feed_forward_length"},{"n_shared_experts","expert_shared_count"},
        {"sliding_window","attention.sliding_window"},{"index_n_heads","attention.indexer.head_count"},
        {"index_head_dim","attention.indexer.key_length"},{"index_topk","attention.indexer.top_k"},
        {"candidate_source_layer_id","attention.indexer.candidate_source_layer"},
        {"candidate_topk_blocks","attention.indexer.candidate_top_k_blocks"},
        {"candidate_block_size","attention.indexer.candidate_block_size"},
        {"hc_mult","hyper_connection.count"},{"hc_sinkhorn_iters","hyper_connection.sinkhorn_iterations"},
        {"hc_eps","hyper_connection.epsilon"},{"rms_norm_eps","attention.layer_norm_rms_epsilon"},
        {"rope_theta","rope.freq_base"},{"compress_rope_theta","attention.compress_rope_freq_base"},
        {"routed_scaling_factor","expert_weights_scale"},{"norm_topk_prob","expert_weights_norm"},
        {"rope_scaling.factor","rope.scaling.factor"},{"rope_scaling.beta_fast","rope.scaling.yarn_beta_fast"},
        {"rope_scaling.beta_slow","rope.scaling.yarn_beta_slow"},
        {"rope_scaling.original_max_position_embeddings","rope.scaling.original_context_length"},
        {"kv_source_layer_ids","attention.kv_source_layers"},{"index_source_layer_ids","attention.indexer.source_layers"},
        {"engram.layer_ids","ple.layers"},{"engram.pad_id","engram.pad_token_id"},
        {"engram.primes","ple.head_vocab_sizes"},{"engram.multipliers","ple.layer_multipliers"},
    };
    for(const auto & n:names)alias(ctx,n.first,n.second);
    // Native files include next-N ratios; the target graph has block_count layers.
    alias(ctx,"compress_ratios","attention.compress_ratios",layers);
    const int clamp=key(ctx,"deepseek41.swiglu_clamp_exp"),shared=key(ctx,"deepseek41.swiglu_clamp_shexp");
    if(gguf_get_arr_type(ctx,clamp)!=GGUF_TYPE_FLOAT32 || gguf_get_arr_type(ctx,shared)!=GGUF_TYPE_FLOAT32 ||
       gguf_get_arr_n(ctx,clamp)!=layers || gguf_get_arr_n(ctx,shared)!=layers)throw std::runtime_error("invalid native V4.1 clamp arrays");
    const auto * values=(const float*)gguf_get_arr_data(ctx,clamp),*s=(const float*)gguf_get_arr_data(ctx,shared);
    const float limit=values[0];
    for(size_t i=0;i<layers;++i)if(values[i]!=limit || s[i]!=limit)throw std::runtime_error("per-layer native V4.1 clamps are not supported");
    gguf_set_val_f32(ctx,"deepseek41.swiglu_limit",limit);
    const int tokens=key(ctx,"tokenizer.ggml.tokens");
    const auto vocab=gguf_get_arr_n(ctx,tokens);
    if(vocab>UINT32_MAX)throw std::runtime_error("native V4.1 vocabulary too large");
    gguf_set_val_u32(ctx,"deepseek41.vocab_size",vocab);
    const auto heads=number(ctx,"deepseek41.ple.heads_per_ngram"),dim=number(ctx,"deepseek41.embedding_length_per_layer_input"),ngram=number(ctx,"deepseek41.ple.ngram_size");
    if(!heads || ngram<2 || ngram>8 || !dim || dim%32)throw std::runtime_error("invalid native Engram geometry");
    const int ids=key(ctx,"deepseek41.engram.layer_ids"),primes=key(ctx,"deepseek41.engram.primes");
    const size_t slots=gguf_get_arr_n(ctx,ids),per=(ngram-1)*heads;
    if(gguf_get_arr_n(ctx,primes)!=slots*per)throw std::runtime_error("native Engram partition count mismatch");
    const auto type=gguf_get_arr_type(ctx,primes);const void * data=gguf_get_arr_data(ctx,primes);
    std::vector<uint32_t> rows(slots),partition_sizes(slots*per);
    for(size_t i=0;i<slots;++i)for(size_t j=0;j<per;++j) {
        const size_t at=i*per+j;uint64_t p=0;
        if(type==GGUF_TYPE_UINT64)p=((const uint64_t*)data)[at];
        else if(type==GGUF_TYPE_UINT32)p=((const uint32_t*)data)[at];
        else throw std::runtime_error("unsupported native Engram prime type");
        if(!p || p>UINT32_MAX || rows[i]>UINT32_MAX-p)throw std::runtime_error("invalid native Engram prime");
        partition_sizes[at]=p;rows[i]+=p;
    }
    gguf_set_arr_data(ctx,"deepseek41.engram.primes",GGUF_TYPE_UINT32,partition_sizes.data(),partition_sizes.size());
    gguf_set_arr_data(ctx,"deepseek41.engram.rows",GGUF_TYPE_UINT32,rows.data(),rows.size());
    nlohmann::json config={{"engram_n_heads",heads},{"engram_head_dim",dim},{"engram_max_ngram_size",ngram}};
    // Keep the existing optional DSpark sidecar usable with either model format.
    const std::pair<const char *,const char *> dspark[]={
        {"dspark_block_size","block_size"},{"dspark_noise_token_id","noise_token_id"},
        {"dspark_n_routed_experts","expert_count"},{"dspark_num_experts_per_tok","expert_used_count"},
        {"dspark_markov_rank","markov_rank"},
    };
    for(const auto & n:dspark)config[n.first]=number(ctx,std::string("deepseek41.dspark.")+n.second);
    const int target=key(ctx,"deepseek41.dspark.target_layers");
    if(gguf_get_kv_type(ctx,target)!=GGUF_TYPE_ARRAY ||
       (gguf_get_arr_type(ctx,target)!=GGUF_TYPE_UINT32 && gguf_get_arr_type(ctx,target)!=GGUF_TYPE_INT32))
        throw std::runtime_error("invalid native DSpark target layers");
    const auto * targets=static_cast<const uint32_t*>(gguf_get_arr_data(ctx,target));
    std::vector<uint32_t> target_layers(targets,targets+gguf_get_arr_n(ctx,target));
    for(auto l:target_layers)if(l>=layers)throw std::runtime_error("native DSpark target layer out of range");
    config["dspark_target_layer_ids"]=target_layers;
    gguf_set_val_str(ctx,"deepseek41.config",config.dump().c_str());
}
