#include "routing-table.h"
#include "ggml-backend-moe-cache.h"
#include "nlohmann/json.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace {
using json=nlohmann::json;
constexpr int max_experts=512;
struct row {
    // Advisory frequencies only: deliberately unsynchronized plain increments.
    // Concurrent updates may be lost; readers need not see the latest value.
    std::array<intptr_t,max_experts> counts{};
};
struct table {
    std::mutex mutex,save_mutex;
    std::map<std::string,std::unique_ptr<row>> rows;
    std::atomic<bool> recording{false};
    std::string model,path;
    bool enabled=false;
    bool loaded_history=false;
};
table & history(){static auto * t=new table;return *t;}
std::string row_name(const std::string & name) {
    for(const char * projection:{"ffn_gate_exps","ffn_up_exps","ffn_down_exps"}) {
        const auto at=name.find(projection);
        if(at!=std::string::npos)return name.substr(0,at)+"ffn_exps"+name.substr(at+std::strlen(projection));
    }
    return {};
}
void * bind(const ggml_tensor * tensor) {
    if(tensor->ne[2]<1 || tensor->ne[2]>max_experts)return nullptr;
    const auto name=row_name(tensor->name);if(name.empty())return nullptr;
    auto & t=history();std::lock_guard<std::mutex> lock(t.mutex);
    auto & p=t.rows[name];if(!p)p=std::make_unique<row>();return p.get();
}
void observe(void * handle,const int32_t * ids,const float * weights,int n) {
    if(!history().recording.load(std::memory_order_relaxed))return;
    auto & r=*(row *)handle;
    for(int i=0;i<n;++i)if(ids[i]>=0 && ids[i]<max_experts && weights[i]!=0) {
        auto & c=r.counts[ids[i]];
        if(c<INTPTR_MAX)++c;
    }
}
intptr_t count(const void * handle,int id) {
    if(id<0 || id>=max_experts)return 0;
    const auto & r=*(const row *)handle;
    return r.counts[id];
}
}

void dsv4f_routing_init(const std::string & model_path) {
    auto & t=history();std::lock_guard<std::mutex> saving(t.save_mutex);
    std::lock_guard<std::mutex> lock(t.mutex);
    t.recording=false;ggml_dsv4f_routing={};t.rows.clear();t.enabled=false;t.loaded_history=false;
    const char * env=std::getenv("DSV4F_ADAPTIVE_ROUTING");
    if(env && !std::strcmp(env,"0"))return;
    if(env && std::strcmp(env,"1"))throw std::runtime_error("DSV4F_ADAPTIVE_ROUTING must be 0 or 1");
    t.model=std::filesystem::path(model_path).stem().string();
    // Identify split models by the first GGUF shard's filename, keeping the
    // shard suffix. A different entry shard still resolves to that same name.
    t.model=std::regex_replace(t.model,std::regex("-[0-9]{5}-of-([0-9]{5})$"),"-00001-of-$1");
    if(t.model.empty())throw std::runtime_error("adaptive routing requires a model filename");
    t.path=(std::filesystem::path("models")/(t.model+".routing.json")).string();
    try {
        if(std::filesystem::exists(t.path)) {
            if(std::filesystem::file_size(t.path)>64*1024*1024)throw std::runtime_error("routing file exceeds 64 MiB");
            std::ifstream input(t.path);if(!input)throw std::runtime_error("cannot open routing file");
            json data;input>>data;
            if(data.at("version")!=1 || data.at("model")!=t.model || !data.at("experts").is_object())
                throw std::runtime_error("routing schema/model mismatch");
            const auto & experts=data.at("experts");
            if(experts.size()>4096)throw std::runtime_error("too many routing rows");
            std::map<std::string,std::unique_ptr<row>> loaded;
            bool has_counts=false;
            const std::regex valid_name("blk\\.[0-9]+\\.ffn_exps(\\.[a-z0-9_]+)?\\.weight");
            for(auto it=experts.begin();it!=experts.end();++it) {
                if(!std::regex_match(it.key(),valid_name) || !it.value().is_object())throw std::runtime_error("invalid routing row");
                auto p=std::make_unique<row>();
                for(auto c=it.value().begin();c!=it.value().end();++c) {
                    const auto & id=c.key();
                    if(id.empty() || id.size()>3 || id.find_first_not_of("0123456789")!=std::string::npos ||
                       (id.size()>1 && id[0]=='0') || !c.value().is_number_integer())throw std::runtime_error("invalid expert counter");
                    const int eid=std::stoi(id);if(eid>=max_experts)throw std::runtime_error("expert ID out of range");
                    if(c.value().is_number_unsigned()?c.value().get<uint64_t>()>uint64_t(INTPTR_MAX):c.value().get<int64_t>()<0)
                        throw std::runtime_error("expert counter must be in 0..INTPTR_MAX");
                    p->counts[eid]=c.value().get<intptr_t>();
                    has_counts|=p->counts[eid]>0;
                }
                loaded.emplace(it.key(),std::move(p));
            }
            t.rows=std::move(loaded);
            t.loaded_history=has_counts;
            std::fprintf(stderr,"[dsv4f-routing] loaded %zu rows from %s\n",t.rows.size(),t.path.c_str());
        }
    } catch(const std::exception & e) {
        t.rows.clear();std::fprintf(stderr,"[dsv4f-routing] ignoring %s: %s\n",t.path.c_str(),e.what());
    }
    t.enabled=true;ggml_dsv4f_routing={bind,observe,count};
    std::fprintf(stderr,"[dsv4f-routing] enabled: %s\n",t.path.c_str());
}
void dsv4f_routing_start(){
    if(history().enabled && history().loaded_history && ggml_dsv4f_experts.gpu_warm_start)
        ggml_dsv4f_experts.gpu_warm_start();
    if(history().enabled && ggml_dsv4f_experts.gpu_drain)ggml_dsv4f_experts.gpu_drain();
    history().recording.store(history().enabled,std::memory_order_relaxed);
}
bool dsv4f_routing_save() {
    auto & t=history();std::lock_guard<std::mutex> saving(t.save_mutex);
    if(!t.enabled)return true;
    std::string temporary;int fd=-1;
    try {
        // GPU completion can precede the service worker's final all-hit
        // observation. Drain those records before taking a counter snapshot.
        if(ggml_dsv4f_experts.gpu_drain)ggml_dsv4f_experts.gpu_drain();
        json data={{"version",1},{"model",t.model},{"experts",json::object()}};
        {
            std::lock_guard<std::mutex> lock(t.mutex);
            for(const auto & item:t.rows) {
                auto counters=json::object();
                for(int id=0;id<max_experts;++id) {
                    const auto n=item.second->counts[id];
                    if(n)counters[std::to_string(id)]=n;
                }
                if(!counters.empty())data["experts"][item.first]=std::move(counters);
            }
        }
        std::filesystem::create_directories(std::filesystem::path(t.path).parent_path());
        std::string pattern=t.path+".tmp.XXXXXX";std::vector<char> name(pattern.begin(),pattern.end());name.push_back(0);
        fd=mkstemp(name.data());if(fd<0)throw std::runtime_error(std::strerror(errno));temporary=name.data();
        const auto bytes=data.dump(2)+"\n";size_t at=0;
        while(at<bytes.size()) {
            const auto n=write(fd,bytes.data()+at,bytes.size()-at);
            if(n<0 && errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("cannot write routing counters");
            at+=size_t(n);
        }
        if(fsync(fd))throw std::runtime_error("cannot sync routing counters");
        const int error=close(fd);fd=-1;if(error)throw std::runtime_error("cannot close routing counters");
        if(rename(temporary.c_str(),t.path.c_str()))throw std::runtime_error(std::strerror(errno));
        return true;
    } catch(const std::exception & e) {
        if(fd>=0)close(fd);
        if(!temporary.empty())unlink(temporary.c_str());
        std::fprintf(stderr,"[dsv4f-routing] save failed for %s: %s\n",t.path.c_str(),e.what());return false;
    }
}
