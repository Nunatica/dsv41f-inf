#include "engram.h"
#include "engram-fp8.h"
#include "models/deepseek41.h"
#include "nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <limits>
#include <cstdlib>
#include <cerrno>
#include <algorithm>
using json=nlohmann::json;
namespace {
uint64_t now_ns() {return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
uint64_t mix(uint64_t x) {x^=x>>30;x*=UINT64_C(0xbf58476d1ce4e5b9);x^=x>>27;x*=UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
void read_exact(int fd,void * out,size_t n,uint64_t off,dsv4f_engram::stats * stats) {
    size_t done=0;while(done<n){ssize_t got=pread(fd,(char*)out+done,n-done,off+done);if(stats){++stats->reads;if(got>0)stats->bytes+=got;}if(got<0 && errno==EINTR)continue;if(got<=0)throw std::runtime_error("[dsv4f-engram] truncated or unreadable table");done+=got;}
}
void write_exact(int fd,const void * data,size_t n) {
    size_t done=0;while(done<n){ssize_t got=write(fd,(const char*)data+done,n-done);if(got<0 && errno==EINTR)continue;if(got<=0)throw std::runtime_error("[dsv4f-engram] trace write failed");done+=got;}
}
}
struct dsv4f_engram::state {
    options opts;
    struct entry {uint64_t row=UINT64_MAX,stamp=0;};
    struct cache {
        std::mutex mutex;
        std::vector<entry> tags;
        std::vector<uint8_t> data;
        size_t sets=0;
        uint64_t clock=0;
    };
    std::vector<std::unique_ptr<cache>> caches;
    std::mutex trace_mutex;
    int trace_fd=-1;
    ~state(){if(trace_fd>=0)close(trace_fd);}
};
dsv4f_engram::options dsv4f_engram::from_env() {
    options o;
    if(const char * v=std::getenv("DSV4F_ENGRAM_DECODE")) {
        if(!std::strcmp(v,"lut"))o.lut=true;
        else if(std::strcmp(v,"baseline"))throw std::runtime_error("DSV4F_ENGRAM_DECODE must be baseline or lut");
    }
    auto flag=[](const char * name){const char * v=std::getenv(name);if(!v)return false;if(std::strcmp(v,"0") && std::strcmp(v,"1"))throw std::runtime_error(std::string(name)+" must be 0 or 1");return !std::strcmp(v,"1");};
    o.dedup=flag("DSV4F_ENGRAM_DEDUP");o.profile=flag("DSV4F_ENGRAM_PROFILE");
    if(const char * v=std::getenv("DSV4F_ENGRAM_CACHE_MIB")) {
        if(!*v || std::strspn(v,"0123456789")!=std::strlen(v))throw std::runtime_error("DSV4F_ENGRAM_CACHE_MIB must be an integer in 0..4096");
        errno=0;char * end=nullptr;const auto n=std::strtoull(v,&end,10);
        if(errno || *end || n>4096)throw std::runtime_error("DSV4F_ENGRAM_CACHE_MIB must be an integer in 0..4096");
        o.cache_bytes=size_t(n)<<20;
    }
    if(const char * v=std::getenv("DSV4F_ENGRAM_TRACE"))o.trace=v;
    return o;
}
dsv4f_engram::dsv4f_engram(const llama_model_deepseek41 & m,const std::string & dir):dsv4f_engram(m,dir,from_env()){}
dsv4f_engram::dsv4f_engram(const llama_model_deepseek41 & m,const std::string & dir,const options & opts):model(m),impl(new state) {
    impl->opts=opts;
    tables.resize(m.engram_layers.size());
    try {
        for(auto & entry:std::filesystem::directory_iterator(dir)) {
            if(entry.path().extension()!=".safetensors")continue;
            std::ifstream f(entry.path(),std::ios::binary);uint64_t bytes=0;f.read((char*)&bytes,8);
            if(!f || bytes>64*1024*1024)throw std::runtime_error("invalid safetensors header");
            std::string header(bytes,'\0');f.read(header.data(),bytes);auto h=json::parse(header);
            const uint64_t size=std::filesystem::file_size(entry.path());
            for(size_t s=0;s<tables.size();++s)for(int k=0;k<2;++k) {
                std::string key="layers."+std::to_string(m.engram_layers[s])+".engram.embed."+(k?"scale":"weight");
                if(!h.contains(key))continue;auto & t=h.at(key);auto shape=t.at("shape").get<std::vector<int64_t>>();
                auto off=t.at("data_offsets").get<std::vector<uint64_t>>();const int width=k?m.engram_dim/32:m.engram_dim;
                if(t.at("dtype")!=(k?"F8_E8M0":"F8_E4M3") || shape!=std::vector<int64_t>{m.engram_rows[s],width} || off.size()!=2 || off[1]<off[0] || off[1]-off[0]!=(uint64_t)m.engram_rows[s]*width || off[1]>size-8-bytes)throw std::runtime_error("Engram tensor dtype/shape/extent mismatch: "+key);
                auto & tab=tables[s];int & fd=k?tab.fds:tab.fdw;uint64_t & at=k?tab.os:tab.ow;
                if(fd>=0)throw std::runtime_error("duplicate Engram tensor: "+key);
                fd=open(entry.path().c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("cannot open Engram shard");at=8+bytes+off[0];
            }
        }
        for(auto & t:tables)if(t.fdw<0 || t.fds<0)throw std::runtime_error("missing original FP8 Engram tables");
    }catch(...){for(auto & t:tables){if(t.fdw>=0)close(t.fdw);if(t.fds>=0)close(t.fds);}throw;}
    // The cap includes fixed-size tags and original FP8 + scale bytes,
    // shared across all tables. No unbounded map or full-table expansion.
    try {
    for(size_t slot=0;slot<tables.size();++slot) {
        auto c=std::make_unique<state::cache>();
        const size_t per_table=tables.empty()?0:opts.cache_bytes/tables.size();
        c->sets=per_table/(4*(sizeof(state::entry)+size_t(m.engram_dim+m.engram_dim/32)));
        c->tags.resize(c->sets*4);c->data.resize(c->sets*4*(m.engram_dim+m.engram_dim/32));
        impl->caches.emplace_back(std::move(c));
    }
    if(!opts.trace.empty()) {
        impl->trace_fd=open(opts.trace.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
        if(impl->trace_fd<0)throw std::runtime_error("[dsv4f-engram] cannot create new trace file");
        const uint32_t header[]={0x454e4731,1,uint32_t(m.engram_ngram),uint32_t(tables.size())};
        write_exact(impl->trace_fd,header,sizeof(header));
    }
    }catch(...){for(auto & t:tables){if(t.fdw>=0)close(t.fdw);if(t.fds>=0)close(t.fds);}throw;}
    std::fprintf(stderr,"[dsv4f-engram] %zu layer tables; FP8 rows read from %s; decode=%s cache=%zu bytes (cap=%zu) dedup=%d\n",tables.size(),dir.c_str(),opts.lut?"lut":"baseline",cache_bytes(),opts.cache_bytes,opts.dedup);
}
dsv4f_engram::~dsv4f_engram(){for(auto & t:tables){close(t.fdw);close(t.fds);}}
size_t dsv4f_engram::cache_bytes() const {
    size_t n=0;for(const auto & c:impl->caches)n+=c->tags.size()*sizeof(state::entry)+c->data.size();return n;
}
void dsv4f_engram::clear_cache() const {
    for(auto & c:impl->caches){std::lock_guard<std::mutex> lock(c->mutex);std::fill(c->tags.begin(),c->tags.end(),state::entry{});c->clock=0;}
}
void dsv4f_engram::read(int slot,const std::vector<int32_t> & history,int nt,std::vector<float> & out,stats * result,const std::atomic<bool> * cancel) const {
    const auto & m=model;const int N=m.engram_ngram,H=m.engram_heads,D=m.engram_dim,C=(N-1)*H;
    if(slot<0 || size_t(slot)>=tables.size() || nt<0 || history.size()!=size_t(nt)*N)throw std::runtime_error("invalid Engram read geometry");
    const auto & opts=impl->opts;stats st;
    const bool measure=opts.profile || result;const bool count=measure;
    const uint64_t start=measure?now_ns():0;
    auto & cache=*impl->caches[slot];std::unique_lock<std::mutex> lock(cache.mutex,std::defer_lock);
    if(cache.sets)lock.lock(); // Model-shared readers remain safe across contexts.
    if(impl->trace_fd>=0) {
        std::lock_guard<std::mutex> trace_lock(impl->trace_mutex);
        const uint32_t rec[]={uint32_t(slot),uint32_t(nt)};
        write_exact(impl->trace_fd,rec,sizeof(rec));write_exact(impl->trace_fd,history.data(),history.size()*sizeof(int32_t));
    }
    out.resize(size_t(nt)*C*D);std::vector<uint8_t>w(D),sc(D/32);
    // Per-call dedup is bounded to 32768 rows / at most 1 MiB metadata,
    // independently of the persistent cache. It never survives this batch.
    struct seen {uint64_t row=UINT64_MAX,index=0;};
    const size_t unique_limit=opts.dedup && nt>1?std::min(size_t(nt)*C,size_t(32768)):0;
    size_t buckets=1;while(buckets<unique_limit*2)buckets*=2;
    std::vector<seen> duplicate(unique_limit?buckets:0);size_t n_unique=0;
    auto consume=[&](uint64_t row,size_t index) {
        if(count)++st.rows;
        float * dst=out.data()+index*D;
        const uint64_t before=measure?now_ns():0;
        if(unique_limit) {
            size_t p=mix(row)&(buckets-1);
            while(duplicate[p].row!=UINT64_MAX && duplicate[p].row!=row)p=(p+1)&(buckets-1);
            if(duplicate[p].row==row) {
                std::memcpy(dst,out.data()+duplicate[p].index*D,size_t(D)*sizeof(float));
                if(count)++st.duplicates;if(measure)st.copy_ns+=now_ns()-before;return;
            }
            if(n_unique<unique_limit){duplicate[p]={row,index};++n_unique;}
        }
        size_t victim=0;const size_t stride=D+D/32;
        const uint8_t * weights=w.data(),*scales=sc.data();bool hit=false;
        if(cache.sets) {
            const size_t base=(mix(row)%cache.sets)*4;victim=base;
            for(size_t i=base;i<base+4;++i) {
                if(cache.tags[i].row==row) {
                    cache.tags[i].stamp=++cache.clock;
                    weights=cache.data.data()+i*stride;scales=weights+D;
                    hit=true;if(count)++st.hits;break;
                }
                if(cache.tags[i].stamp<cache.tags[victim].stamp)victim=i;
            }
        }
        if(measure)st.copy_ns+=now_ns()-before;
        uint64_t at=measure?now_ns():0;
        if(!hit) {
            read_exact(tables[slot].fdw,w.data(),D,tables[slot].ow+row*D,count?&st:nullptr);
            read_exact(tables[slot].fds,sc.data(),D/32,tables[slot].os+row*(D/32),count?&st:nullptr);
        }
        if(measure){st.io_ns+=now_ns()-at;at=now_ns();}
        if(opts.lut)dsv4f_engram_fp8::lut(weights,scales,dst,D);
        else dsv4f_engram_fp8::baseline(weights,scales,dst,D);
        if(measure){st.fp8_ns+=now_ns()-at;at=now_ns();}
        if(!hit && cache.sets) {
            std::memcpy(cache.data.data()+victim*stride,w.data(),D);
            std::memcpy(cache.data.data()+victim*stride+D,sc.data(),D/32);
            cache.tags[victim]={row,++cache.clock};
        }
        if(measure)st.copy_ns+=now_ns()-at;
    };
    uint64_t hash_start=measure?now_ns():0;
    for(int t=0;t<nt;++t) {
        if(cancel && cancel->load(std::memory_order_relaxed))throw std::runtime_error("Engram prefetch cancelled");
        uint64_t rolling=0,offset=0;bool blocked=false;
        for(int n=0;n<N;++n) {
            int id=history[size_t(t)*N+n];if(id<0)blocked=true;
            if(id>=0 && size_t(id)>=m.token_map.size())throw std::runtime_error("invalid Engram history token");
            if(id>=0 && m.token_map[id]==UINT32_MAX)blocked=true;
            const uint64_t token=blocked?m.engram_pad:m.token_map[id];
            rolling^=token*uint64_t(m.multipliers[size_t(slot)*N+n]);
            if(!n)continue;
            for(int h=0;h<H;++h) {
                const int c=(n-1)*H+h;const uint64_t prime=m.primes[size_t(slot)*C+c];
                // Preserve signed int64 remainder followed by positive modulo.
                int64_t rem=int64_t(rolling)%int64_t(prime);if(rem<0)rem+=prime;
                const uint64_t row=offset+rem;offset+=prime;
                if(measure)st.hash_ns+=now_ns()-hash_start;
                consume(row,size_t(t)*C+c);
                if(measure)hash_start=now_ns();
            }
        }
    }
    if(measure){st.hash_ns+=now_ns()-hash_start;st.total_ns=now_ns()-start;}
    if(result)*result=st;
    if(opts.profile)std::fprintf(stderr,"[dsv4f-engram-profile] slot=%d tokens=%d rows=%llu reads=%llu bytes=%llu hits=%llu dedup=%llu hash_us=%.3f io_us=%.3f fp8_us=%.3f copy_us=%.3f total_us=%.3f\n",slot,nt,(unsigned long long)st.rows,(unsigned long long)st.reads,(unsigned long long)st.bytes,(unsigned long long)st.hits,(unsigned long long)st.duplicates,st.hash_ns/1000.,st.io_ns/1000.,st.fp8_ns/1000.,st.copy_ns/1000.,st.total_ns/1000.);
}
