#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include "ggml-backend.h"
struct llama_model_deepseek41;
class dsv4f_engram {
    struct table { int fdw=-1,fds=-1; uint64_t ow=0,os=0; };
    std::vector<table> tables;
    const llama_model_deepseek41 & model;
    struct state;
    std::unique_ptr<state> impl;
public:
    struct options {
        bool lut=false, dedup=false, profile=false, batch_io=true;
        size_t cache_bytes=0;
        std::string trace;
    };
    struct stats {
        uint64_t rows=0, reads=0, bytes=0, hits=0, duplicates=0;
        uint64_t hash_ns=0, io_ns=0, fp8_ns=0, copy_ns=0, total_ns=0;
    };
    static options from_env();
    explicit dsv4f_engram(const llama_model_deepseek41 & m,const std::string & dir);
    dsv4f_engram(const llama_model_deepseek41 & m,const std::string & dir,const options & opts);
    ~dsv4f_engram();
    void read(int slot,const std::vector<int32_t> & history,int nt,std::vector<float> & out,stats * result=nullptr,const std::atomic<bool> * cancel=nullptr) const;
    void clear_cache() const;
    size_t cache_bytes() const;
};

// Context-owned, one-batch lookahead. Hints never change KV state. A job is
// adopted only when its complete history, position, sequence and FP mode match.
class dsv4f_engram_prefetch {
    struct state;
    std::unique_ptr<state> impl;
public:
    struct job;
    dsv4f_engram_prefetch(const dsv4f_engram & reader,int ngram,int tables);
    ~dsv4f_engram_prefetch();
    void hint(const int32_t * tokens,int count,int seq,int batch);
    void clear();
    std::shared_ptr<job> begin(const std::vector<int32_t> & history,int nt,int pos,int seq);
    static void take(const std::shared_ptr<job> & pending,int slot,std::vector<float> & rows);
};

// One persistent reader per graph. Start after token history is known; the
// custom row op waits and uploads only when its layer's hidden state is ready.
class dsv4f_engram_async {
    struct state;
    std::unique_ptr<state> impl;
public:
    dsv4f_engram_async(const dsv4f_engram & reader,int nt,int width,int tables);
    ~dsv4f_engram_async();
    bool enabled(int slot) const;
    ggml_tensor * tensor(ggml_context * ctx,ggml_tensor * anchor,int slot,ggml_backend_dev_t device);
    // The caller must complete preceding graph consumers before reusing rows.
    void start(const std::vector<int32_t> & history,std::shared_ptr<dsv4f_engram_prefetch::job> pending=nullptr);
};
