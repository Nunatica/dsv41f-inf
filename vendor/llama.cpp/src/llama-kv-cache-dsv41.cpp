#include "llama-kv-cache-dsv41.h"
#include "models/deepseek41.h"
#include "llama-cparams.h"
#include "llama-io.h"
#include "engram.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>
llama_kv_cache_dsv41::llama_kv_cache_dsv41(const llama_model_deepseek41 & m,const llama_cparams & p,ggml_type type):
    llama_kv_cache(m,m.hparams,type,type,false,p.offload_kqv,true,p.n_ctx,1,1,0,LLAMA_SWA_TYPE_NONE,nullptr,[](int32_t){return false;},nullptr,nullptr),model41(m),capacity(p.n_ctx) {
    if(p.n_seq_max!=1)throw std::runtime_error("[dsv4f-v41] currently requires --parallel 1");
    // A batch must not overwrite the preceding SWA window before its first
    // query reads it. Two extra rows also cover a ratio-2 pair at an odd edge.
    window_capacity=std::min(capacity, uint32_t(GGML_PAD(m.hparams.n_swa+p.n_ubatch+2,256)));
    layers41.resize(m.hparams.n_layer());
    size_t resident=0,host_bytes=0;
    for(size_t il=0;il<layers41.size();++il) {
        auto gc=ggml_context_ptr(ggml_init({32*ggml_tensor_overhead(),nullptr,true}));auto & l=layers41[il];
        auto tensor=[&](ggml_type t,int d,int n,const char * role){auto * x=ggml_new_tensor_2d(gc.get(),t,d,n);ggml_format_name(x,"cache_v41_%s_l%zu",role,il);return x;};
        const int dh=m.hparams.n_embd_head_k();l.raw=tensor(type,dh,window_capacity,"raw");
        if(m.ratios[il] && m.kv_sources[il]==(int)il) {
            const int nc=(capacity+m.ratios[il]-1)/m.ratios[il]+1;
            l.comp=tensor(type,dh,nc,"comp");l.idx=tensor(type,m.idx_dim,nc,"idx");
            if(m.ratios[il]==2){l.state=tensor(GGML_TYPE_F32,dh,window_capacity,"pool");l.score=tensor(GGML_TYPE_F32,dh,window_capacity,"score");}
        }
        auto buft=p.offload_kqv?ggml_backend_dev_buffer_type(m.dev_layer(il)):ggml_backend_cpu_buffer_type();
        auto buf=ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors_from_buft(gc.get(),buft));
        if(!buf)throw std::runtime_error("[dsv4f-v41] KV allocation failed");ggml_backend_buffer_clear(buf.get(),0);
        resident+=ggml_backend_buffer_get_size(buf.get());
        contexts.push_back(std::move(gc));buffers.push_back(std::move(buf));
        if(window_capacity<capacity) {
            auto hc=ggml_context_ptr(ggml_init({8*ggml_tensor_overhead(),nullptr,true}));
            for(auto pair:{std::make_pair(l.raw,&l.raw_history),std::make_pair(l.state,&l.state_history),std::make_pair(l.score,&l.score_history)}) {
                if(!pair.first)continue;
                pair.second->host=ggml_new_tensor_2d(hc.get(),pair.first->type,dh,capacity);
                host_bytes+=ggml_nbytes(pair.second->host);
            }
            if(!m.hparams.no_alloc) {
                auto * dev=m.dev_layer(il);
                const bool gpu=p.offload_kqv && ggml_backend_dev_type(dev)!=GGML_BACKEND_DEVICE_TYPE_CPU;
                auto hbuft=gpu?ggml_backend_dev_host_buffer_type(dev):ggml_backend_cpu_buffer_type();
                if(!hbuft)throw std::runtime_error("V4.1 rollback history requires pinned host buffers");
                auto hb=ggml_backend_buffer_ptr(ggml_backend_alloc_ctx_tensors_from_buft(hc.get(),hbuft));
                if(!hb)throw std::runtime_error("V4.1 host history allocation failed");
                ggml_backend_buffer_clear(hb.get(),0);
                auto map=gpu?(ggml_dsv41_map_host_fn)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_dsv41_map_host"):nullptr;
                if(gpu) {
                    auto sync=(ggml_dsv41_sync_fn)ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),"ggml_dsv41_sync");
                    if(!sync)throw std::runtime_error("V4.1 history synchronization is unavailable");
                    history_devices[dev]=sync;
                }
                if(gpu && !map)throw std::runtime_error("V4.1 bounded KV requires the CUDA/HIP history backend");
                for(auto * h:{&l.raw_history,&l.state_history,&l.score_history})if(h->host)h->mapped=gpu?map(dev,h->host->data):h->host->data;
                buffers.push_back(std::move(hb));
            }
            contexts.push_back(std::move(hc));
        }
    }
    std::fprintf(stderr,"[dsv4f-v41-kv] context=%u window=%u resident=%.2f MiB rollback_host=%.2f MiB%s\n",
                 capacity,window_capacity,resident/1048576.,host_bytes/1048576.,m.hparams.no_alloc?" (metadata)":"");
}
void llama_kv_cache_dsv41::synchronize_history() const {
    for(const auto & entry:history_devices)entry.second(entry.first);
}
void llama_kv_cache_dsv41::prepare(const llama_ubatch & u) {
    if(!u.n_tokens)return;
    const int64_t first=u.pos[0];
    if(window_capacity<capacity && first!=next_position) {
        // Normal forward execution never reads history back. On rewind, wait
        // for write-through kernels, then restore just the preceding window.
        synchronize_history();
        const int64_t begin=std::max<int64_t>(0,first-model41.hparams.n_swa);
        for(auto & l:layers41) {
            for(auto pair:{std::make_pair(l.raw,&l.raw_history),std::make_pair(l.state,&l.state_history),std::make_pair(l.score,&l.score_history)}) {
                if(!pair.first)continue;
                for(int64_t p=begin;p<first;) {
                    const int slot=p%window_capacity;
                    const int64_t count=std::min<int64_t>(first-p,window_capacity-slot);
                    ggml_backend_tensor_set(pair.first,(const char*)pair.second->host->data+p*pair.first->nb[1],slot*pair.first->nb[1],count*pair.first->nb[1]);
                    p+=count;
                }
            }
        }
    }
    next_position=u.pos[u.n_tokens-1]+1;
}
void llama_kv_cache_dsv41::clear(bool data){if(engram_prefetch)engram_prefetch->clear();if(data)synchronize_history();llama_kv_cache::clear(data);if(data)for(auto & b:buffers)ggml_backend_buffer_clear(b.get(),0);next_position=0;}
std::map<ggml_backend_buffer_type_t,size_t> llama_kv_cache_dsv41::memory_breakdown() const {auto result=llama_kv_cache::memory_breakdown();for(auto & b:buffers)result[ggml_backend_buffer_get_type(b.get())]+=ggml_backend_buffer_get_size(b.get());return result;}
// Serialize authoritative positional history, not the transient ring. This
// keeps sessions portable across different ubatch/window capacities. Older
// absolute-position layouts are rejected explicitly by the version marker.
void llama_kv_cache_dsv41::state_write(llama_io_write_i & io,llama_seq_id seq,llama_state_seq_flags flags) const {
    llama_kv_cache::state_write(io,seq,flags);
    const uint32_t layout[]={0x56343102,capacity};io.write(layout,sizeof(layout));
    for(auto & l:layers41) {
        for(auto * t:{l.comp,l.idx})if(t)io.write_tensor(t,0,ggml_nbytes(t));
        for(auto pair:{std::make_pair(l.raw,&l.raw_history),std::make_pair(l.state,&l.state_history),std::make_pair(l.score,&l.score_history)}) {
            auto * t=pair.second->host?pair.second->host:pair.first;
            if(t)io.write_tensor(t,0,ggml_nbytes(t));
        }
    }
}
void llama_kv_cache_dsv41::state_read(llama_io_read_i & io,llama_seq_id seq,llama_state_seq_flags flags) {
    if(engram_prefetch)engram_prefetch->clear();
    llama_kv_cache::state_read(io,seq,flags);
    uint32_t layout[2];io.read(layout,sizeof(layout));
    if(layout[0]!=0x56343102 || layout[1]!=capacity)throw std::runtime_error("incompatible V4.1 cache session layout");
    for(auto & l:layers41) {
        for(auto * t:{l.comp,l.idx})if(t)io.read_tensor(t,0,ggml_nbytes(t));
        for(auto pair:{std::make_pair(l.raw,&l.raw_history),std::make_pair(l.state,&l.state_history),std::make_pair(l.score,&l.score_history)}) {
            auto * t=pair.second->host?pair.second->host:pair.first;
            if(t)io.read_tensor(t,0,ggml_nbytes(t));
        }
    }
    // The first execution restores only its preceding window from history.
    next_position=-1;
}
