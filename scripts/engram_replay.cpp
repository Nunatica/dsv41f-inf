#include "engram.h"
#include "models/deepseek41.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
struct record {uint32_t slot,nt;std::vector<int32_t> history;};
static uint64_t hash(uint64_t h,const std::vector<float> & out){for(float f:out){uint32_t b;std::memcpy(&b,&f,4);h=(h^b)*UINT64_C(1099511628211);}return h;}
int main(int argc,char ** argv) {
    if(argc!=7){std::fprintf(stderr,"usage: engram-replay MODEL SHARDS TRACE MODE[baseline|lut|dedup|cache|all] FILTER[decode|prefill|all] REPS\n");return 2;}
    try {
        const std::string mode=argv[4],filter=argv[5];const int reps=std::stoi(argv[6]);if(reps<1||reps>100)throw std::runtime_error("reps");
        dsv4f_engram::options opt;
        if(mode=="lut"||mode=="dedup"||mode=="cache"||mode=="all")opt.lut=true;
        else if(mode!="baseline")throw std::runtime_error("mode");
        opt.dedup=mode=="dedup"||mode=="all";opt.cache_bytes=mode=="cache"||mode=="all"?32<<20:0;
        if(filter!="decode"&&filter!="prefill"&&filter!="all")throw std::runtime_error("filter");
        std::ifstream trace(argv[3],std::ios::binary);uint32_t hdr[4];trace.read((char*)hdr,sizeof(hdr));
        if(!trace||hdr[0]!=0x454e4731||hdr[1]!=1||hdr[2]<2||hdr[2]>8)throw std::runtime_error("trace header");
        std::vector<record> records;
        while(trace.peek()!=EOF){uint32_t r[2];trace.read((char*)r,sizeof(r));if(!trace||r[0]>=hdr[3]||!r[1]||r[1]>65536)throw std::runtime_error("trace record");record rec{r[0],r[1],std::vector<int32_t>(size_t(r[1])*hdr[2])};trace.read((char*)rec.history.data(),rec.history.size()*4);if(!trace)throw std::runtime_error("trace truncated");if(filter=="all"||(filter=="decode"&&rec.nt==1)||(filter=="prefill"&&rec.nt>1))records.push_back(std::move(rec));}
        if(records.empty())throw std::runtime_error("no records");
        auto mp=llama_model_default_params();mp.no_alloc=true;mp.n_gpu_layers=0;mp.use_extra_bufts=false;mp.load_mode=LLAMA_LOAD_MODE_NONE;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> base(llama_model_load_from_file(argv[1],mp),llama_model_free);
        auto * m=dynamic_cast<llama_model_deepseek41*>(base.get());if(!m||m->engram_ngram!=int(hdr[2])||m->engram_layers.size()!=hdr[3])throw std::runtime_error("model/trace mismatch");
        dsv4f_engram reader(*m,argv[2],opt);std::vector<float> out;
        // Numerical verification and accounting are outside the timed region.
        dsv4f_engram::options refopt;dsv4f_engram reference(*m,argv[2],refopt);
        dsv4f_engram::stats sum;uint64_t checksum=UINT64_C(1469598103934665603),tokens=0;
        for(const auto & rec:records){std::vector<float> expected;reference.read(rec.slot,rec.history,rec.nt,expected);dsv4f_engram::stats st;reader.read(rec.slot,rec.history,rec.nt,out,&st);if(out.size()!=expected.size()||std::memcmp(out.data(),expected.data(),out.size()*4))throw std::runtime_error("replay bit mismatch");checksum=hash(checksum,out);tokens+=rec.nt;sum.rows+=st.rows;sum.reads+=st.reads;sum.bytes+=st.bytes;sum.hits+=st.hits;sum.duplicates+=st.duplicates;sum.hash_ns+=st.hash_ns;sum.io_ns+=st.io_ns;sum.fp8_ns+=st.fp8_ns;sum.copy_ns+=st.copy_ns;sum.total_ns+=st.total_ns;}
        // OS page cache has been warmed identically by reference verification.
        // Cold-row-cache passes reset only our bounded cache, never global caches.
        for(const char * locality:{"empty","retained"}) {
            std::vector<double> times;
            for(int rep=0;rep<reps;++rep){if(!std::strcmp(locality,"empty"))reader.clear_cache();const auto start=std::chrono::steady_clock::now();for(const auto & rec:records){std::vector<float> fresh;reader.read(rec.slot,rec.history,rec.nt,fresh);}const auto end=std::chrono::steady_clock::now();times.push_back(std::chrono::duration<double,std::milli>(end-start).count());}
            std::sort(times.begin(),times.end());const double median=times[times.size()/2];
            std::printf("{\"mode\":\"%s\",\"filter\":\"%s\",\"locality\":\"%s\",\"accounting\":\"empty_verification\",\"records\":%zu,\"slot_tokens\":%llu,\"cache_bytes\":%zu,\"reps\":%d,\"median_ms\":%.6f,\"us_per_slot_token\":%.6f,\"checksum\":\"%016llx\",\"rows\":%llu,\"reads\":%llu,\"bytes\":%llu,\"hits\":%llu,\"dedup\":%llu,\"hash_ms\":%.6f,\"io_ms\":%.6f,\"fp8_ms\":%.6f,\"copy_ms\":%.6f}\n",mode.c_str(),filter.c_str(),locality,records.size(),(unsigned long long)tokens,reader.cache_bytes(),reps,median,median*1000/tokens,(unsigned long long)checksum,(unsigned long long)sum.rows,(unsigned long long)sum.reads,(unsigned long long)sum.bytes,(unsigned long long)sum.hits,(unsigned long long)sum.duplicates,sum.hash_ns/1e6,sum.io_ns/1e6,sum.fp8_ns/1e6,sum.copy_ns/1e6);
        }
        return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"[dsv4f-engram-replay] %s\n",e.what());return 1;}
}
