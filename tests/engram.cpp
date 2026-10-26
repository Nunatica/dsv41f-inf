#include "engram.h"
#include "engram-fp8.h"
#include "models/deepseek41.h"
#include "nlohmann/json.hpp"
#include <array>
#include <cfenv>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <atomic>
#include <unistd.h>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
static void require(bool ok,const char * why) {if(!ok)throw std::runtime_error(why);}
static void fp8_test() {
    std::array<uint8_t,256> w{};for(int b=0;b<256;++b)w[b]=b;
    std::array<uint8_t,8> sc{};std::array<float,256> a{},b{};
    const int saved=std::fegetround();
#ifdef __SSE__
    const unsigned mx=_mm_getcsr();
#endif
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
        std::fesetround(rounding);
        for(int flags=0;flags<4;++flags) {
#ifdef __SSE__
            _mm_setcsr((_mm_getcsr()&~0x8040u)|((flags&1)?0x8000:0)|((flags&2)?0x40:0));
#endif
            for(int s=0;s<256;++s) {
                sc.fill(s);
                dsv4f_engram_fp8::baseline(w.data(),sc.data(),a.data(),256);
                dsv4f_engram_fp8::lut(w.data(),sc.data(),b.data(),256);
                if(std::memcmp(a.data(),b.data(),sizeof(a))) {
                    for(int j=0;j<256;++j)if(std::memcmp(&a[j],&b[j],4))std::fprintf(stderr,"fp8 mismatch byte=%d scale=%d round=%d ftz_daz=%d ref=%a got=%a\n",j,s,rounding,flags,a[j],b[j]);
                    throw std::runtime_error("FP8/scale bit parity");
                }
            }
        }
    }
    std::fesetround(saved);
#ifdef __SSE__
    _mm_setcsr(mx);
#endif
}
static void shard(const std::filesystem::path & p,const llama_model_deepseek41 & m) {
    nlohmann::json j;std::vector<uint8_t> data;
    for(size_t slot=0;slot<m.engram_layers.size();++slot)for(int scale=0;scale<2;++scale) {
        const auto key="layers."+std::to_string(m.engram_layers[slot])+".engram.embed."+(scale?"scale":"weight");
        const size_t n=size_t(m.engram_rows[slot])*(scale?m.engram_dim/32:m.engram_dim),off=data.size();
        j[key]={{"dtype",scale?"F8_E8M0":"F8_E4M3"},{"shape",{m.engram_rows[slot],scale?m.engram_dim/32:m.engram_dim}},{"data_offsets",{off,off+n}}};
        for(size_t i=0;i<n;++i)data.push_back(uint8_t(i*71+slot*13+(scale?7:0)));
    }
    const auto h=j.dump();const uint64_t n=h.size();std::ofstream f(p,std::ios::binary);f.write((const char*)&n,8);f.write(h.data(),h.size());f.write((const char*)data.data(),data.size());
}
static void async_test(const dsv4f_engram & reader,ggml_backend_t backend) {
    for(bool later_only:{false,true})for(int nt:{1,2,4,32,257,4096}) {
        auto * ctx=ggml_init({2<<20,nullptr,true});
        auto * anchor=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1);ggml_set_input(anchor);
        dsv4f_engram_async pending(reader,nt,6*64,2);
        auto * a=later_only?ggml_new_tensor_2d(ctx,GGML_TYPE_F32,6*64,nt):
            pending.tensor(ctx,anchor,0,ggml_backend_get_device(backend));
        if(later_only)ggml_set_input(a);
        require(pending.enabled(0)!=later_only && !pending.enabled(1),"selected async slots");
        // A second consumption depends on the first and can use its own readiness.
        auto * b=pending.tensor(ctx,a,1,ggml_backend_get_device(backend));
        auto * graph=ggml_new_graph_custom(ctx,32,false);ggml_build_forward_expand(graph,b);
        ggml_set_output(a);ggml_set_output(b);
        auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"async graph allocation");
        float zero=0;ggml_backend_tensor_set(anchor,&zero,0,4);
        const int replays=nt>4?4:32;
        for(int replay=0;replay<replays;++replay) {
            std::vector<int32_t> history;
            for(int t=0;t<nt;++t)for(int n=0;n<4;++n)history.push_back(replay%7==0?-1:replay%7==1?10:(replay*7+t*3+n)%32);
            const int rounding[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
            std::fesetround(rounding[replay%4]);
            if(later_only){std::vector<float> rows;reader.read(0,history,nt,rows);ggml_backend_tensor_set(a,rows.data(),0,rows.size()*4);}
            pending.start(history);
            require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"async graph compute");
            for(int slot=0;slot<2;++slot) {
                std::vector<float> actual(nt*6*64),expected;
                ggml_backend_tensor_get(slot?b:a,actual.data(),0,actual.size()*4);
                reader.read(slot,history,nt,expected);
                require(!std::memcmp(actual.data(),expected.data(),actual.size()*4),"async rows/replay bit parity");
            }
        }
        std::fesetround(FE_TONEAREST);
        ggml_backend_synchronize(backend);ggml_backend_buffer_free(buffer);ggml_free(ctx);
        std::printf("Engram async nt=%d later_only=%d %d changing-history replays bitwise PASS\n",nt,int(later_only),replays);
    }
}
static void prefetch_test(const dsv4f_engram & reader){
    dsv4f_engram_prefetch pool(reader,4,2);
    std::vector<int32_t> tokens(117);for(size_t i=0;i<tokens.size();++i)tokens[i]=(i*7)%32;
    auto history=[&](int pos,int nt){std::vector<int32_t> h;for(int t=0;t<nt;++t)for(int n=0;n<4;++n)h.push_back(pos+t-n<0?-1:tokens[pos+t-n]);return h;};
    auto check=[&](const std::vector<int32_t>& h,int nt,int pos,int seq){
        auto j=pool.begin(h,nt,pos,seq);require(bool(j),"prefetch job");
        for(int s=0;s<2;++s){std::vector<float>a,b;dsv4f_engram_prefetch::take(j,s,a);reader.read(s,h,nt,b);require(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*4),"prefetched rows bit parity");}
    };
    for(int rounding:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){
        std::fesetround(rounding);pool.hint(tokens.data(),tokens.size(),3,32);
        for(int pos=0;pos<int(tokens.size());pos+=32){const int nt=std::min(32,int(tokens.size())-pos);check(history(pos,nt),nt,pos,3);}
        pool.clear();
    }
    std::fesetround(FE_TONEAREST);
    // An abandoned lookahead, changed prompt, sequence, FP mode or suffix must
    // fall back to authoritative history instead of returning stale rows.
    pool.hint(tokens.data(),tokens.size(),3,32);check(history(0,32),32,0,3);
    auto changed=history(32,32);changed[1]=17;check(changed,32,32,3);
    check(history(64,32),32,64,4);
    pool.hint(tokens.data(),tokens.size(),3,32);check(history(0,32),32,0,3);
    std::fesetround(FE_DOWNWARD);check(history(32,32),32,32,3);std::fesetround(FE_TONEAREST);
    tokens[0]=29;pool.hint(tokens.data(),tokens.size(),3,32);check(history(0,32),32,0,3);
    pool.clear();require(!pool.begin(history(32,32),32,32,3),"cleared hint fallback");
    pool.hint(tokens.data(),tokens.size(),3,32);
    auto j=pool.begin(history(0,32),32,0,3);pool.clear();
    bool rejected=false;try{std::vector<float> rows;dsv4f_engram_prefetch::take(j,0,rows);}catch(const std::runtime_error&){rejected=true;}
    // A completed read is also valid if it wins the race against cancellation.
    (void)rejected;
    std::puts("Engram next-batch prefetch boundary/tail/FP-mode/stale-hint/cancellation parity PASS");
}
int main(int argc,char ** argv) {
    try {
        fp8_test();
        auto m=std::make_unique<llama_model_deepseek41>(llama_model_default_params());
        m->engram_dim=64;m->engram_ngram=4;m->engram_heads=2;m->engram_pad=2;m->engram_layers={1,2};
        m->primes={7,11,13,17,19,23,11,13,17,19,23,29};m->engram_rows={90,112};
        m->multipliers={UINT64_C(0xeafffffffffffff1),31,71,97,17,UINT64_C(0xefffffffffffff1),117,197};
        for(int i=0;i<32;++i)m->token_map.push_back(i==10?UINT32_MAX:i);
        char temp[]="/tmp/dsv4f-engram-XXXXXX";const auto dir=std::filesystem::path(mkdtemp(temp));
        struct cleanup {std::filesystem::path p;~cleanup(){std::filesystem::remove_all(p);}} cleanup{dir};
        shard(dir/"fixture.safetensors",*m);
        dsv4f_engram::options base;base.batch_io=false;dsv4f_engram reference(*m,dir.string(),base);
        std::vector<int32_t> history;for(int t=0;t<64;++t)for(int n=0;n<4;++n)history.push_back(t%4==0?-1:t%4==1?10:(t*7-n+32)%32);
        std::vector<std::vector<float>> expected(2);for(int slot=0;slot<2;++slot)reference.read(slot,history,64,expected[slot]);
        {auto o=base;o.lut=true;o.batch_io=true;dsv4f_engram batched(*m,dir.string(),o);
         std::vector<int32_t> one(history.begin(),history.begin()+4);
         for(int slot=0;slot<2;++slot){std::vector<float>a,b;dsv4f_engram::stats sa,sb;reference.read(slot,one,1,a,&sa);batched.read(slot,one,1,b,&sb);
             require(a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4),"batched O_DIRECT row bit parity");
             require(sa.reads==sb.reads&&sa.bytes==sb.bytes,"batched O_DIRECT accounting");}}
        prefetch_test(reference);
        for(int mode=0;mode<5;++mode) {
            auto opt=base;opt.lut=mode!=0;opt.dedup=mode==2||mode==4;opt.cache_bytes=mode==3?4096:mode==4?1<<20:0;
            dsv4f_engram reader(*m,dir.string(),opt);require(reader.cache_bytes()<=opt.cache_bytes,"cache cap");
            for(int pass=0;pass<3;++pass)for(int slot=0;slot<2;++slot) {
                std::vector<float> out;dsv4f_engram::stats st;reader.read(slot,history,64,out,&st);
                require(out.size()==expected[slot].size()&&!std::memcmp(out.data(),expected[slot].data(),out.size()*4),"row decode/cache/hash parity");
                require(st.rows==64*6 && st.reads==2*(st.rows-st.hits-st.duplicates),"pread accounting");
                if(mode==4 && pass>0)require(!st.reads,"warm cache hit");
                if(opt.dedup)require(st.duplicates>0,"prefill dedup hit");
            }
            reader.clear_cache();std::atomic<bool> ok{true};std::vector<std::thread> threads;
            for(int i=0;i<4;++i)threads.emplace_back([&,i]{for(int r=0;r<16;++r){std::vector<float> out;reader.read(i%2,history,64,out);if(std::memcmp(out.data(),expected[i%2].data(),out.size()*4))ok=false;}});
            for(auto & t:threads)t.join();require(ok,"concurrent context cache parity");
            reader.clear_cache();std::vector<float> out;dsv4f_engram::stats st;reader.read(0,history,64,out,&st);require(st.reads>0,"clear cache");
            bool rejected=false;try{reader.read(0,{99,0,0,0},1,out);}catch(const std::runtime_error&){rejected=true;}require(rejected,"invalid token");
        }
        if(argc>1 && !std::strcmp(argv[1],"async-gpu")) {
            ggml_backend_load_all();auto * gpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU,nullptr);
            if(!gpu)return 77;
            dsv4f_engram::options o;o.lut=true;o.cache_bytes=4096;
            dsv4f_engram reader(*m,dir.string(),o);async_test(reader,gpu);ggml_backend_free(gpu);
        } else {
            ggml_backend_load_all();auto * cpu=ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr);
            require(cpu,"CPU backend");async_test(reference,cpu);ggml_backend_free(cpu);
        }
        // Cached rows must be re-decoded in the caller's floating-point mode.
        {dsv4f_engram::options o;o.lut=true;o.cache_bytes=1<<20;dsv4f_engram reader(*m,dir.string(),o);
         for(int round:{FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}){std::fesetround(round);for(int slot=0;slot<2;++slot){std::vector<float>a,b;reference.read(slot,history,64,a);reader.read(slot,history,64,b);require(!std::memcmp(a.data(),b.data(),a.size()*4),"cached row rounding parity");}}std::fesetround(FE_TONEAREST);}
        dsv4f_engram::options trace;trace.trace=(dir/"rows.trace").string();
        {dsv4f_engram reader(*m,dir.string(),trace);std::vector<float> out;reader.read(0,history,64,out);}
        const auto fds=std::distance(std::filesystem::directory_iterator("/proc/self/fd"),std::filesystem::directory_iterator{});
        for(int k=0;k<4;++k){bool rejected=false;try{dsv4f_engram reader(*m,dir.string(),trace);}catch(const std::runtime_error&){rejected=true;}require(rejected,"trace overwrite protection");}
        require(fds==std::distance(std::filesystem::directory_iterator("/proc/self/fd"),std::filesystem::directory_iterator{}),"constructor fd leak");
        // Saturate the 32768-entry batch dedup cap: tracked duplicates still
        // hit after saturation, while untracked rows continue through I/O.
        {auto large=std::make_unique<llama_model_deepseek41>(llama_model_default_params());
         large->engram_dim=32;large->engram_ngram=2;large->engram_heads=1;large->engram_pad=2;large->engram_layers={1};large->engram_rows={65537};large->primes={65537};large->multipliers={1,0};
         for(int i=0;i<65537;++i)large->token_map.push_back(i);
         auto sub=dir/"large";std::filesystem::create_directory(sub);shard(sub/"fixture.safetensors",*large);
         std::vector<int32_t> h;for(int i=0;i<65536;++i){h.push_back(i);h.push_back(0);}h.insert(h.end(),{32769,0,1,0});
         dsv4f_engram::options o;o.lut=true;o.dedup=true;dsv4f_engram optimized(*large,sub.string(),o),original(*large,sub.string(),base);
         std::vector<float>a,b;dsv4f_engram::stats st;original.read(0,h,65538,a);optimized.read(0,h,65538,b,&st);
         require(!std::memcmp(a.data(),b.data(),a.size()*4),"dedup saturated output");require(st.duplicates==1&&st.reads==2*(65538-1),"dedup saturated fallback");}
        std::filesystem::resize_file(dir/"fixture.safetensors",1);
        bool rejected=false;try{std::vector<float> out;reference.read(0,history,64,out);}catch(const std::runtime_error&){rejected=true;}require(rejected,"truncated read");
        std::puts("[dsv4f-engram-test] all FP8/scale bytes, fenv, cache eviction/cap, dedup, concurrent contexts, fd cleanup, truncation: PASS");
        return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"[dsv4f-engram-test] FAIL: %s\n",e.what());return 1;}
}
