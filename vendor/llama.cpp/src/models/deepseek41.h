#pragma once
#include "models.h"
#include <map>
#include <memory>

class dsv4f_engram;
struct llama_model_deepseek41 : public llama_model_base {
    explicit llama_model_deepseek41(const llama_model_params & p) : llama_model_base(p) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & p) const override;
    std::map<std::string,ggml_tensor *> weights;
    struct experts { int type; ggml_tensor * gate,*up,*down; std::vector<int32_t> local; ggml_tensor * mapping = nullptr; };
    std::vector<std::vector<experts>> tiers;
    std::vector<int32_t> ratios, kv_sources, index_sources, engram_layers;
    std::vector<uint32_t> token_map;
    std::vector<uint32_t> primes, engram_rows;
    std::vector<uint64_t> multipliers;
    int hc=4, ff=2304, og=8, ol=1024, idx_heads=32, idx_dim=128, idx_top=512;
    int candidate_layer=20, candidate_blocks=2048, candidate_size=8;
    int engram_heads=8, engram_dim=256, engram_ngram=4, engram_pad=2;
    float eps=1e-20f, hc_eps=1e-6f, clamp=10, scaling=1, rope_base=10000, compress_base=1000000;
    float rope_factor=1, beta_fast=32, beta_slow=1; int orig_ctx=4096;
    bool norm_topk=true;
    int dspark_block=0, dspark_noise=0, dspark_experts=0, dspark_used=0, dspark_rank=0;
    std::vector<int32_t> dspark_layers;
    std::shared_ptr<dsv4f_engram> engram;
    ggml_tensor * W(int layer,const std::string & role) const;
    struct graph : public llm_graph_context {
        graph(const llama_model_deepseek41 & model,const llm_graph_params & p);
    };
};
