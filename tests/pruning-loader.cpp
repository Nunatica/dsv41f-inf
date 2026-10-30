#include "routing-table.h"
#include "direct-weights.h"
#include "ggml-backend-moe-cache.h"
#include "llama.h"
#include "llama-model.h"
#include "models/deepseek41.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <stdexcept>
#include <vector>
#include <unistd.h>
using json=nlohmann::json;
static void require(bool value,const char * why){if(!value)throw std::runtime_error(why);}
struct source_tier {int type;std::vector<int32_t> local;std::string names[3];size_t plane[3];};
static std::vector<std::vector<source_tier>> original;
static std::vector<std::set<int>> removed;
static size_t expected_active=0,expected_skipped=0;
static int top_k=0;
static std::string model_path;
static gguf_context * file_info;
static llama_model * load(bool compact,bool gpu,bool no_alloc=false,bool tiered=false) {
    setenv("DSV4F_RAM_BUDGET_MIB",tiered?"184320":"0",1);
    setenv("DSV4F_RAM_CACHE_MIB","2",1);
    dsv4f_routing_init(model_path);
    auto p=llama_model_default_params();p.n_gpu_layers=gpu?999:0;p.no_alloc=no_alloc;
    p.no_host=!gpu;p.use_extra_bufts=false;
    p.load_mode=compact?LLAMA_LOAD_MODE_DIRECT_IO:LLAMA_LOAD_MODE_MMAP;
    const llama_model_tensor_buft_override overrides[]={{"\\.ffn_(gate|up|down)_exps",ggml_backend_cpu_buffer_type()},{nullptr,nullptr}};
    p.tensor_buft_overrides=overrides;
    auto * m=llama_model_load_from_file(model_path.c_str(),p);require(m,"load model");return m;
}
static std::vector<float> run(bool compact,bool gpu,bool tiered=false) {
    auto * base=load(compact,gpu,false,tiered);auto * m=dynamic_cast<llama_model_deepseek41*>(base);require(m,"V4.1 model");
    auto cp=llama_context_default_params();cp.n_ctx=128;cp.n_batch=64;cp.n_ubatch=32;cp.n_seq_max=1;
    cp.n_threads=4;cp.n_threads_batch=4;cp.offload_kqv=gpu;
    cp.type_k=cp.type_v=GGML_TYPE_Q8_0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx=llama_init_from_model(base,cp);require(ctx,"context");
    dsv4f_preload_result preload;
    ggml_dsv4f_host_weights_stats loaded{};
    if(tiered) {
        setenv("DSV4F_RAM_PRELOAD","1",1);preload=dsv4f_preload_experts();
        require(!preload.budget_limited,"retained experts fit the cache");
        ggml_dsv4f_host_weights.statistics(&loaded);
    }
    size_t active=0,cache_bytes=0,planes=0;
    std::ifstream input(model_path,std::ios::binary);
    for(size_t l=0;l<m->tiers.size();++l) {
        for(const auto & source:original[l]) {
            const auto it=std::find_if(m->tiers[l].begin(),m->tiers[l].end(),[&](const auto & e){return e.type==source.type;});
            std::vector<int> retained;
            for(size_t id=0;id<source.local.size();++id)if(source.local[id]>=0 && !removed[l].count(id))retained.push_back(id);
            if(compact && retained.empty()) {
                require(it==m->tiers[l].end(),"empty quantization tier removed from graph");
                for(const auto & name:source.names)require(ggml_nbytes(m->weights.at(name))==0,"no allocation for discarded tier");
                continue;
            }
            require(it!=m->tiers[l].end(),"active quantization tier present");
            ggml_tensor * tensors[]={it->gate,it->up,it->down};
            if(!compact)continue;
            for(int projection=0;projection<3;++projection) {
                auto * t=tensors[projection];require(t->ne[2]==int64_t(retained.size()),"compact allocation dimensions");active+=ggml_nbytes(t);
                std::vector<char> want(source.plane[projection]),got(want.size());
                const int tid=gguf_find_tensor(file_info,source.names[projection].c_str());require(tid>=0,"source tensor");
                for(size_t packed=0;packed<retained.size();++packed) {
                    const int global=retained[packed],old_local=source.local[global];
                    require(it->local[global]==int(packed),"global to packed expert IDs");
                    const size_t offset=gguf_get_data_offset(file_info)+gguf_get_tensor_offset(file_info,tid)+old_local*want.size();
                    cache_bytes+=(offset%4096+want.size()+4095)/4096*4096;++planes;
                    input.seekg(offset);
                    input.read(want.data(),want.size());require(bool(input),"read reference plane");
                    ggml_backend_tensor_get(t,got.data(),packed*got.size(),got.size());
                    require(got==want,"retained expert bytes match original file planes");
                }
            }
            for(int global:removed[l])require(it->local[global]<0,"pruned global IDs unmapped");
        }
    }
    if(compact)require(active==expected_active,"exact compact expert byte count");
    if(tiered)require(preload.planes==planes && preload.resident_bytes==cache_bytes && preload.read_bytes==cache_bytes,
        "preload allocated/read discarded planes or missed retained planes");
    const int nv=llama_vocab_n_tokens(llama_model_get_vocab(base));std::vector<float> logits;
    for(int first=0;first<9;) {
        const int n=first==0?8:1;auto batch=llama_batch_init(n,0,1);batch.n_tokens=n;
        for(int i=0;i<n;++i){batch.token[i]=2+first+i;batch.pos[i]=first+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1;}
        require(llama_decode(ctx,batch)==0,"prefill/decode");
        for(int i=0;i<n;++i){const float * p=llama_get_logits_ith(ctx,i);logits.insert(logits.end(),p,p+nv);}
        llama_batch_free(batch);first+=n;
    }
    if(tiered) {
        ggml_dsv4f_host_weights_stats inferred{};ggml_dsv4f_host_weights.statistics(&inferred);
        require(inferred.reads==loaded.reads && inferred.read_bytes==loaded.read_bytes && inferred.evictions==loaded.evictions,
            "first prefill/decode reread or evicted preloaded experts");
        require(inferred.resident_bytes==cache_bytes && !inferred.transient_bytes,"inference changed fitting expert residency");
        std::printf("tiered pruning preload: planes=%zu resident_bytes=%zu first_inference_read_bytes=0 PASS\n",planes,cache_bytes);
    }
    llama_free(ctx);llama_model_free(base);return logits;
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;model_path=std::filesystem::absolute(argv[1]).string();
    setenv("DSV4F_RAM_BUDGET_MIB","0",1);setenv("DSV4F_ADAPTIVE_ROUTING","0",1);setenv("DSV4F_ADAPTIVE_PRUNING","0",1);
    setenv("DSV4F_MTP","0",1);setenv("DSV4F_WHOLE_EXPERT","0",1);setenv("DSV4F_GPU_SERVICE","0",1);setenv("GGML_CUDA_MOE_CACHE","0",1);
    setenv("DSV4F_ENGRAM_DIR",std::filesystem::path(model_path).replace_extension(".engram").c_str(),1);
    ggml_backend_load_all();
    auto * base=load(true,false,true);auto * model=dynamic_cast<llama_model_deepseek41*>(base);require(model,"fixture model");
    top_k=model->hparams.n_expert_used(0);
    json history={{"version",1},{"model",std::filesystem::path(model_path).stem().string()},{"experts",json::object()}};
    std::vector<std::vector<int>> counts;
    for(const auto & layer:model->tiers) {
        original.emplace_back();counts.emplace_back(model->hparams.n_expert);
        for(const auto & e:layer) {
            source_tier source;source.type=e.type;source.local=e.local;ggml_tensor * tensors[]={e.gate,e.up,e.down};
            for(int p=0;p<3;++p){source.names[p]=tensors[p]->name;source.plane[p]=tensors[p]->nb[2];}
            // Make both q3_k experts least used, including a completely removed
            // quantization tier at 50%, and keep non-prefix IDs in other tiers.
            const int offset=e.type==GGML_TYPE_Q3_K?0:e.type==GGML_TYPE_Q2_0?10:100;
            std::string row=e.gate->name;row.replace(row.find("ffn_gate_exps"),13,"ffn_exps");
            for(size_t id=0;id<e.local.size();++id)if(e.local[id]>=0){int n=offset+e.local[id]+1;counts.back()[id]=n;history["experts"][row][std::to_string(e.local[id])]=n;}
            original.back().push_back(std::move(source));
        }
    }
    llama_model_free(base);file_info=gguf_init_from_file(model_path.c_str(),{true,nullptr});require(file_info,"GGUF metadata");
    const auto cwd=std::filesystem::current_path();char temp[]="/tmp/dsv4f-pruning-loader-XXXXXX";require(mkdtemp(temp),"temporary directory");
    std::filesystem::current_path(temp);std::filesystem::create_directory("models");
    {std::ofstream f(std::filesystem::path("models")/(std::filesystem::path(model_path).stem().string()+".routing.json"));f<<history;}
    for(int pct:{25,50}) {
        setenv("DSV4F_ADAPTIVE_PRUNING",std::to_string(pct).c_str(),1);removed.clear();expected_active=expected_skipped=0;
        struct candidate {double score;size_t layer,id,bytes;};
        std::vector<candidate> candidates;
        size_t total=0,saved=0;std::vector<size_t> active;
        for(size_t l=0;l<counts.size();++l) {
            double calls=0;for(int n:counts[l])calls+=n;
            removed.emplace_back();active.push_back(counts[l].size());
            for(size_t i=0;i<counts[l].size();++i) {
                size_t bytes=0;
                for(const auto & e:original[l])if(e.local[i]>=0)for(size_t plane:e.plane)bytes+=plane;
                candidates.push_back({double(counts[l][i])/calls/double(bytes),l,i,bytes});total+=bytes;
            }
        }
        std::sort(candidates.begin(),candidates.end(),[](const candidate & a,const candidate & b){return std::tie(a.score,a.layer,a.id)<std::tie(b.score,b.layer,b.id);});
        const size_t target=(total*pct+99)/100;
        // Keep the fixture top-k even when cheap candidates concentrate in one layer.
        for(const auto & c:candidates)if(saved<target && active[c.layer]>size_t(top_k)) {
            removed[c.layer].insert(c.id);--active[c.layer];saved+=c.bytes;
        }
        require(saved>=target,"reference reaches memory target");
        expected_skipped=saved;expected_active=total-saved;
        for(bool gpu:{false,true}) {
            if(gpu && !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU))continue;
            for(const char * maps:{"0","1"})for(const char * pipeline:{"0","1"}) {
                setenv("DSV4F_STATIC_EXPERT_MAPS",maps,1);
                setenv("DSV4F_LOAD_PIPELINE_MIB",pipeline,1);
                const auto reference=run(false,gpu);
                for(bool tiered:{false,true}) {
                    const auto packed=run(true,gpu,tiered);require(reference.size()==packed.size(),"logit size");
                    double max_error=0;for(size_t i=0;i<packed.size();++i){require(std::isfinite(packed[i]),"finite logits");max_error=std::max(max_error,double(std::abs(reference[i]-packed[i])));require(std::abs(reference[i]-packed[i])<2e-4f,"packed vs original expert inference parity");}
                    std::printf("pruning loader pct=%d backend=%s tiered=%d static_maps=%s pipeline_mib=%s active_bytes=%zu skipped_bytes=%zu max_error=%g PASS\n",pct,gpu?"GPU":"CPU",tiered,maps,pipeline,expected_active,expected_skipped,max_error);
                }
            }
        }
    }
    gguf_free(file_info);std::filesystem::current_path(cwd);std::filesystem::remove_all(temp);
}
