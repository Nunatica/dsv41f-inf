#include "direct-weights.h"
#include "llama-mmap.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

static void require(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
int main() {
    setenv("DSV4F_RAM_BUDGET_MIB","131072",1);
    setenv("DSV4F_RAM_CACHE_MIB","1",1);
    setenv("DSV4F_RAM_RESERVE_MIB","4",1);
    char path[]="/tmp/dsv4f-direct-XXXXXX";
    const int fd=mkstemp(path);require(fd>=0,"create fixture");
    constexpr size_t offset=123,plane=65536,count=64;
    std::vector<unsigned char> expected(offset+plane*count);
    for(size_t i=0;i<expected.size();++i)expected[i]=(i*71+i/97)%251;
    require(write(fd,expected.data(),expected.size())==(ssize_t)expected.size(),"write fixture");
    require(fsync(fd)==0,"flush fixture");
    require(posix_fadvise(fd,0,0,POSIX_FADV_DONTNEED)==0,"drop test file pages");
    const size_t file_pages=(expected.size()+4095)/4096;
    void * file_map=mmap(nullptr,file_pages*4096,PROT_READ,MAP_SHARED,fd,0);
    require(file_map!=MAP_FAILED,"mincore baseline mapping");
    std::vector<unsigned char> baseline(file_pages);
    require(mincore(file_map,file_pages*4096,baseline.data())==0,"mincore baseline");
    auto * ctx=ggml_init({1<<20,nullptr,true});require(ctx,"context");
    auto * t=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,128,count);
    auto * b=dsv4f_direct_weights_buffer({{t,fd,offset}});
    // A live lease survives many evictions and is shared by simultaneous readers.
    {
        ggml_dsv4f_host_lease first(t->data,plane);
        for(size_t i=1;i<count;++i) {
            ggml_dsv4f_host_lease lease((char*)t->data+i*plane,plane);
            require(!std::memcmp(lease.data,expected.data()+offset+i*plane,plane),"direct expert bytes");
        }
        require(!std::memcmp(first.data,expected.data()+offset,plane),"live lease was evicted");
        std::atomic<bool> ok{true};
        std::vector<std::thread> threads;
        for(int i=0;i<4;++i)threads.emplace_back([&]{
            for(int j=0;j<40;++j){ggml_dsv4f_host_lease lease(t->data,plane);if(lease.data!=first.data || std::memcmp(lease.data,expected.data()+offset,plane))ok=false;}
        });
        for(auto & thread:threads)thread.join();require(ok,"concurrent leases");
    }
    // Generic read copies through leases too; malformed/truncated payloads fail closed.
    std::vector<unsigned char> got(plane*2+111);
    ggml_backend_tensor_get(t,got.data(),plane-13,got.size());
    require(!std::memcmp(got.data(),expected.data()+offset+plane-13,got.size()),"partial generic read");
    bool rejected=false;
    try{ggml_dsv4f_host_lease invalid(t->data,plane+1);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"invalid range accepted");
    // The engine never faults weight file pages into the OS file cache.
    std::vector<unsigned char> residency(file_pages);
    require(mincore(file_map,file_pages*4096,residency.data())==0,"mincore");
    for(size_t i=0;i<file_pages;++i)require(!(residency[i]&1) || (baseline[i]&1),"O_DIRECT populated the file page cache");
    munmap(file_map,file_pages*4096);
    ggml_backend_buffer_free(b);ggml_free(ctx);
    // Consecutive unaligned direct chunks must keep the logical file cursor.
    {
        llama_file input(path,"rb",true);
        require(input.has_direct_io(),"direct chunk reader");
        input.seek(offset,SEEK_SET);
        input.read_raw(got.data(),17001);
        input.read_raw(got.data()+17001,19003);
        require(input.tell()==offset+36004,"direct logical cursor");
        require(!std::memcmp(got.data(),expected.data()+offset,36004),"consecutive direct chunks");
    }
    // Source removal must leave the resident path usable.
    {ggml_dsv4f_host_lease ordinary(expected.data(),16);require(ordinary.data==expected.data(),"resident fallback");}
    const int direct=open(path,O_RDONLY|O_DIRECT);require(direct>=0,"direct fd");
    require(ftruncate(fd,16)==0,"truncate fixture");rejected=false;
    try{dsv4f_direct_read(direct,got.data(),plane,offset);}catch(const std::runtime_error &){rejected=true;}
    require(rejected,"truncation accepted");close(direct);close(fd);unlink(path);
    std::puts("PASS: direct SSD bytes, bounded eviction, concurrent leases, EOF, no file page cache");
}
