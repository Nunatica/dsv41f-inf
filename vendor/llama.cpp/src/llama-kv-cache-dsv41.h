#pragma once
#include "llama-kv-cache.h"
#include "ggml-dsv41-cache.h"
struct llama_model_deepseek41;
class dsv4f_engram_prefetch;
class llama_kv_cache_dsv41 : public llama_kv_cache {
    ggml_context_ptr ctx;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<ggml_context_ptr> contexts;
    std::map<ggml_backend_dev_t,ggml_dsv41_sync_fn> history_devices;
    void synchronize_history() const;
public:
    std::shared_ptr<dsv4f_engram_prefetch> engram_prefetch;
    struct layer {
        ggml_tensor * raw=nullptr,*comp=nullptr,*idx=nullptr,*state=nullptr,*score=nullptr;
        ggml_dsv41_history raw_history, state_history, score_history;
    };
    std::vector<layer> layers41;
    const llama_model_deepseek41 & model41;
    uint32_t capacity, window_capacity;
    int64_t next_position = 0;
    void prepare(const llama_ubatch & ubatch);
    llama_kv_cache_dsv41(const llama_model_deepseek41 & m,const llama_cparams & p,ggml_type type);
    bool get_can_shift() const override { return false; }
    void clear(bool data) override;
    std::map<ggml_backend_buffer_type_t,size_t> memory_breakdown() const override;
    void state_write(llama_io_write_i & io,llama_seq_id seq=-1,llama_state_seq_flags flags=0) const override;
    void state_read(llama_io_read_i & io,llama_seq_id seq=-1,llama_state_seq_flags flags=0) override;
};
