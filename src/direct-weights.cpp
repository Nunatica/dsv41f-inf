#include "direct-weights.h"
#include "ggml-backend-impl.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr size_t alignment=4096;
constexpr uint64_t MiB=1ull<<20;
uint64_t env_mib(const char * name,uint64_t fallback) {
    const char * v=std::getenv(name);
    if(!v)return fallback;
    if(!*v)throw std::runtime_error(std::string(name)+" must be an integer in MiB");
    uint64_t n=0;
    for(const char * p=v;*p;++p) {
        if(*p<'0'||*p>'9'||n>(std::numeric_limits<uint64_t>::max()/MiB-(*p-'0'))/10)
            throw std::runtime_error(std::string(name)+" must be a nonnegative integer in MiB");
        n=n*10+(*p-'0');
    }
    return n*MiB;
}
size_t rounded(size_t n) {
    if(n>SIZE_MAX-alignment)throw std::runtime_error("direct I/O size overflow");
    return (n+alignment-1)&~(alignment-1);
}
struct region {
    void * data=MAP_FAILED;
    size_t bytes=0;
    explicit region(size_t n,int prot=PROT_READ|PROT_WRITE):bytes(rounded(n)) {
        data=mmap(nullptr,bytes,prot,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0);
        if(data==MAP_FAILED)throw std::runtime_error("cannot reserve direct I/O memory");
    }
    ~region(){if(data!=MAP_FAILED)munmap(data,bytes);}
};
void read_aligned(int fd,void * dst,size_t n,uint64_t off,size_t required) {
    // A short final sector is legal only when every requested payload byte arrived.
    size_t done=0;
    while(done<n) {
        const size_t chunk=std::min<size_t>(n-done,64*MiB);
        ssize_t got=pread(fd,(char*)dst+done,chunk,off+done);
        if(got<0 && errno==EINTR)continue;
        if(got<0)throw std::runtime_error(std::string("O_DIRECT read failed: ")+std::strerror(errno));
        if(got==0 || (size_t(got)<chunk && done+got<required))throw std::runtime_error("truncated direct I/O payload");
        done+=got;
        if(size_t(got)<chunk)break;
    }
}
struct source {
    std::unique_ptr<region> address;
    struct span {uintptr_t begin;size_t bytes,expert;int fd;uint64_t offset;};
    std::map<uintptr_t,span> spans;
    std::vector<int> fds;
    ~source(){for(int fd:fds)close(fd);}
};
struct entry {
    std::unique_ptr<region> storage;
    const void * pointer=nullptr;
    size_t users=0;
    source * owner=nullptr;
    std::list<uintptr_t>::iterator age;
};
struct cache {
    std::mutex mutex;
    std::map<uintptr_t,source*> sources;
    std::map<uintptr_t,entry> entries;
    std::map<const void*,uintptr_t> leases;
    std::list<uintptr_t> ages;
    uint64_t used=0,hits=0,reads=0,bytes=0,evictions=0;
    ~cache()=default;
    uint64_t limit() const {
        const uint64_t budget=dsv4f_ram_budget();
        const uint64_t reserve=env_mib("DSV4F_RAM_RESERVE_MIB",4096*MiB);
        if(budget<=reserve)throw std::runtime_error("RAM budget must exceed DSV4F_RAM_RESERVE_MIB");
        uint64_t ceiling=std::min(env_mib("DSV4F_RAM_CACHE_MIB",budget-reserve),budget-reserve);
        // Account for actual process RSS (KV/history, Engram, GPU host buffers, stacks, etc.).
        // mmap/munmap for cache entries returns evicted memory immediately to the OS.
        unsigned long pages=0,resident=0;
        if(FILE * f=std::fopen("/proc/self/statm","r")) {
            const int n=std::fscanf(f,"%lu %lu",&pages,&resident);std::fclose(f);
            if(n==2) {
                const uint64_t rss=uint64_t(resident)*uint64_t(sysconf(_SC_PAGESIZE));
                const uint64_t other=rss>used?rss-used:0;
                ceiling=std::min(ceiling,budget-reserve>other?budget-reserve-other:0);
            }
        }
        return ceiling;
    }
    void erase(std::map<uintptr_t,entry>::iterator it) {
        used-=it->second.storage->bytes;leases.erase(it->second.pointer);
        ages.erase(it->second.age);entries.erase(it);++evictions;
    }
};
// Outlives model/backend static teardown; it contains no allocations after model unload.
cache & pool(){static auto * p=new cache;return *p;}
const void * acquire(const void * ptr,size_t n) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    const auto key=(uintptr_t)ptr;
    auto s=c.sources.upper_bound(key);
    if(s==c.sources.begin())return ptr;
    --s;source * owner=s->second;
    if(key-s->first>=owner->address->bytes)return ptr;
    auto sp=owner->spans.upper_bound(key);
    if(sp==owner->spans.begin())throw std::runtime_error("invalid direct weight address");
    --sp;const auto & span=sp->second;
    const size_t local=key-span.begin;
    if(local>=span.bytes || n!=span.expert || local%span.expert || n>span.bytes-local)
        throw std::runtime_error("direct weights must be leased one complete expert plane at a time");
    auto it=c.entries.find(key);
    if(it!=c.entries.end()) {
        ++it->second.users;++c.hits;c.ages.splice(c.ages.end(),c.ages,it->second.age);
        return it->second.pointer;
    }
    const uint64_t offset=span.offset+local;
    const size_t delta=offset%alignment,total=rounded(delta+n);
    const uint64_t cap=c.limit();
    if(total>cap)throw std::runtime_error("RAM budget has no room for an expert plane; reduce context/batch or reserve");
    while(c.used>cap-total) {
        auto victim=std::find_if(c.ages.begin(),c.ages.end(),[&](uintptr_t k){return c.entries.at(k).users==0;});
        if(victim==c.ages.end())throw std::runtime_error("RAM expert cache is exhausted by active leases");
        c.erase(c.entries.find(*victim));
    }
    auto storage=std::make_unique<region>(total);
    read_aligned(span.fd,storage->data,total,offset-delta,delta+n);
    const void * output=(char*)storage->data+delta;
    c.ages.push_back(key);
    entry e;e.storage=std::move(storage);e.pointer=output;e.users=1;e.owner=owner;e.age=std::prev(c.ages.end());
    c.entries.emplace(key,std::move(e));c.leases.emplace(output,key);c.used+=total;++c.reads;c.bytes+=total;
    if(c.reads%4096==0)std::fprintf(stderr,"[dsv4f-ram] resident=%.2f MiB hit=%llu reads=%llu direct_mib=%.2f evictions=%llu\n",c.used/double(MiB),(unsigned long long)c.hits,(unsigned long long)c.reads,c.bytes/double(MiB),(unsigned long long)c.evictions);
    return output;
}
void release(const void * ptr) {
    auto & c=pool();std::lock_guard<std::mutex> lock(c.mutex);
    auto it=c.leases.find(ptr);if(it==c.leases.end())return;
    auto & e=c.entries.at(it->second);GGML_ASSERT(e.users);--e.users;
}
void free_buffer(ggml_backend_buffer_t buffer) {
    auto * owner=(source*)buffer->context;
    auto & c=pool();
    {std::lock_guard<std::mutex> lock(c.mutex);
        for(auto it=c.entries.begin();it!=c.entries.end();) {
            if(it->second.owner==owner){GGML_ASSERT(!it->second.users);auto old=it++;c.erase(old);}else ++it;
        }
        c.sources.erase((uintptr_t)owner->address->data);
        if(c.sources.empty())ggml_dsv4f_host_weights={};
    }
    delete owner;
}
void * base(ggml_backend_buffer_t b){return ((source*)b->context)->address->data;}
void get_tensor(ggml_backend_buffer_t,const ggml_tensor * t,void * output,size_t off,size_t n) {
    // Unexpected generic tensor uploads cannot dereference the protected descriptor address.
    size_t done=0;
    while(done<n) {
        const size_t eid=(off+done)/t->nb[2],delta=(off+done)%t->nb[2];
        ggml_dsv4f_host_lease lease((char*)t->data+eid*t->nb[2],t->nb[2]);
        const size_t count=std::min(n-done,t->nb[2]-delta);
        std::memcpy((char*)output+done,(const char*)lease.data+delta,count);done+=count;
    }
}
}
uint64_t dsv4f_ram_budget(){return env_mib("DSV4F_RAM_BUDGET_MIB",0);}
bool dsv4f_direct_weights_enabled(){return dsv4f_ram_budget()!=0;}
void dsv4f_direct_read(int fd,void * output,size_t n,uint64_t off) {
    if(!n)return;
    const size_t delta=off%alignment;
    thread_local std::unique_ptr<region> scratch;
    const size_t count=rounded(delta+n);
    if(!scratch || scratch->bytes<count)scratch=std::make_unique<region>(count);
    read_aligned(fd,scratch->data,count,off-delta,delta+n);
    std::memcpy(output,(char*)scratch->data+delta,n);
}
ggml_backend_buffer_t dsv4f_direct_weights_buffer(const std::vector<dsv4f_direct_tensor> & tensors) {
    auto owner=std::make_unique<source>();size_t bytes=0;
    for(const auto & t:tensors)bytes+=rounded(ggml_nbytes(t.tensor));
    owner->address=std::make_unique<region>(bytes,PROT_NONE);
    ggml_backend_buffer_i iface{};iface.free_buffer=free_buffer;iface.get_base=base;iface.get_tensor=get_tensor;
    std::map<int,int> opened;size_t at=0;
    // Prepare everything that can fail before registering the source.
    for(const auto & t:tensors) {
        if(!opened.count(t.fd)) {
            const std::string path="/proc/self/fd/"+std::to_string(t.fd);
            const int fd=open(path.c_str(),O_RDONLY|O_DIRECT|O_CLOEXEC);
            if(fd<0)throw std::runtime_error(std::string("cannot open O_DIRECT weight source: ")+std::strerror(errno));
            owner->fds.push_back(fd);opened[t.fd]=fd;
        }
        const auto key=(uintptr_t)owner->address->data+at;
        owner->spans.emplace(key,source::span{key,ggml_nbytes(t.tensor),t.tensor->nb[2],opened.at(t.fd),t.offset});
        at+=rounded(ggml_nbytes(t.tensor));
    }
    auto & c=pool();
    std::lock_guard<std::mutex> lock(c.mutex);
    (void)c.limit();
    auto * b=ggml_backend_buffer_init(ggml_backend_cpu_buffer_type(),iface,owner.get(),owner->address->bytes);
    at=0;for(const auto & t:tensors){ggml_backend_tensor_alloc(b,t.tensor,(char*)owner->address->data+at);at+=rounded(ggml_nbytes(t.tensor));}
    c.sources.emplace((uintptr_t)owner->address->data,owner.get());
    ggml_dsv4f_host_weights.acquire=acquire;ggml_dsv4f_host_weights.release=release;
    std::fprintf(stderr,"[dsv4f-ram] O_DIRECT expert source: %.2f GiB on SSD, process budget %.2f GiB, cache ceiling %.2f GiB\n",bytes/double(1ull<<30),dsv4f_ram_budget()/double(1ull<<30),c.limit()/double(1ull<<30));
    owner.release();return b;
}
