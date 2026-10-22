#include "dsv4f-service.cuh"
#include "dsv4f-prefill.cuh"
#include "dsv4f-memory.cuh"
#include "ggml-backend-moe-cache.h"
#include "../ggml-dsv4f-cache-policy.h"
#include "../ggml-dsv4f-timeline.h"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"
#include "dsv4f-v41.cuh"
#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__linux__)
#include <time.h>
#endif

namespace {
constexpr int RING=32, MAX_LAYERS=256, MAX_EXPERTS=512, MAX_USED=64, MAX_EMBD=5120;
// Format services may now be submitted together. They share one persistent
// CPU worker pool, so serialize CPU computations while their GPU work proceeds.
std::mutex cpu_lane_mutex;
struct alignas(64) request {
    volatile unsigned long long seq,done,consumed;
    int layer,nk,single,kind,bank;
    uint64_t hits,prefetch_hits;
    float limit;
    int32_t ids[MAX_USED];
    float weights[MAX_USED],x[MAX_EMBD],answer[MAX_EMBD];
};
// A slot's state is its reader count, or -1 while a loader owns it. The loader
// must acquire it before invalidating/replacing a weight. No global GPU drain.
__device__ bool acquire(int * state,int slot) {
    int old=atomicAdd(state+slot,0);
    while(old>=0) { const int got=atomicCAS(state+slot,old,old+1);if(got==old)return true;old=got; }
    return false;
}
__global__ void route(request * ring,unsigned long long * sequence,int * table,int * readers,
                      const float * x,const int32_t * ids,const float * weights,int weight_stride,
                      int layer,int ne,int nk,float limit,int32_t * plan,float * plan_w,unsigned long long * issued,
                      int * speculative,bool single) {
    __shared__ unsigned long long seq;
    __shared__ uint64_t mask;
    if(threadIdx.x==0) {
        seq=atomicAdd(sequence,1ull)+1ull; *issued=seq; mask=0;
        request * r=ring+(seq%RING);
        const unsigned long long previous=seq>RING?seq-RING:0;
        while(r->consumed!=previous) { /* GPU queue cannot overwrite an unconsumed request. */ }
        r->layer=layer;r->nk=nk;r->limit=limit;r->single=single;r->kind=0;r->prefetch_hits=0;
        for(int i=0;i<nk;++i) {
            const int eid=ids[i];
            const int key=layer*MAX_EXPERTS+eid;
            int slot=(weights[i*weight_stride]!=0 && eid>=0 && eid<MAX_EXPERTS)?atomicAdd(table+key,0):-1;
            if(slot>=0) {
                if(!acquire(readers,slot))slot=-1;
                else if(atomicAdd(table+key,0)!=slot){atomicSub(readers+slot,1);slot=-1;}
            }
            if(slot>=0 && speculative && atomicExch(speculative+key,0))r->prefetch_hits|=uint64_t(1)<<i;
            plan[i]=slot;plan_w[i]=weights[i*weight_stride];
            r->ids[i]=eid;r->weights[i]=plan_w[i];
            if(slot>=0 || plan_w[i]==0)mask|=uint64_t(1)<<i;
        }
        r->hits=mask;
    }
    __syncthreads();
    request * r=ring+(seq%RING);
    if(mask!=((uint64_t(1)<<nk)-1))for(int i=threadIdx.x;i<ne;i+=blockDim.x)r->x[i]=x[i];
    __threadfence_system();__syncthreads();
    if(threadIdx.x==0){r->seq=seq;__threadfence_system();}
}
// Publish current-token predictions without synchronously reading IDs back.
// These records share sequence ownership with real CPU requests but never
// overwrite their issued[] join handles or activation/answer buffers.
__global__ void prefetch_request(request * ring,unsigned long long * sequence,
    const int32_t * ids,const char * mapping,size_t map_stride,int global_count,
    int layer,int nk,int bank,float * output) {
    if(threadIdx.x)return;
    const unsigned long long seq=atomicAdd(sequence,1ull)+1ull;
    request * r=ring+(seq%RING);
    const unsigned long long previous=seq>RING?seq-RING:0;
    while(r->consumed!=previous) {}
    r->layer=layer;r->nk=nk;r->kind=1;r->bank=bank;
    for(int k=0;k<nk;++k){const int id=ids[k];r->ids[k]=id>=0 && id<global_count?*(const int32_t *)(mapping+id*map_stride):-1;}
    *output=0;__threadfence_system();r->seq=seq;__threadfence_system();
}
__global__ void release_reduce(const float * down,const float * weights,const int32_t * plan,
                               int * readers,int ne,int nk,float * output) {
    const int r=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<ne){float v=0;for(int i=0;i<nk;++i)if(plan[i]>=0)v+=weights[i]*down[i*ne+r];output[r]=v;}
    // Reader protection must outlive ALL down reads: release in a subsequent kernel.
}
__global__ void release_readers(int * readers,const int32_t * plan,int nk) {
    const int i=threadIdx.x;if(i<nk && plan[i]>=0)atomicSub(readers+plan[i],1);
}
__global__ void cpu_join(request * ring,const unsigned long long * sequence,float * out,int ne,bool poll) {
    const unsigned long long seq=*sequence;const request * r=ring+(seq%RING);
    __shared__ int timeout;
    if(threadIdx.x==0){timeout=0;if(poll){const long long start=clock64();while(r->done!=seq)if(clock64()-start>(1ll<<36)){timeout=1;break;}}}
    __syncthreads();__threadfence_system();
    for(int i=threadIdx.x;i<ne;i+=blockDim.x)out[i]=timeout?NAN:out[i]+((volatile const float*)r->answer)[i];
}
__global__ void reserve_slot(int * readers,int * table,int slot,int old_key,int * claimed) {
    if(threadIdx.x)return;
    *claimed=(old_key<0 || atomicCAS(readers+slot,0,-1)==0)?1:0;
    if(*claimed && old_key>=0)atomicCAS(table+old_key,slot,-1);
    __threadfence_system();
}
__global__ void publish_slot(int * readers,int * table,int slot,int key,int * speculative,bool predicted) {
    if(threadIdx.x)return;
    if(speculative)atomicExch(speculative+key,int(predicted));
    __threadfence();atomicExch(readers+slot,0);atomicExch(table+key,slot);
}
__global__ void prefill_lease(int * readers, int * table, const int * updates, int count, bool publish) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const int slot = updates[3*i], old_key = updates[3*i+1], new_key = updates[3*i+2];
    if (publish) {
        atomicExch(readers+slot, 0);
        if (new_key >= 0) atomicExch(table+new_key, slot);
    } else {
        if (old_key >= 0) atomicCAS(table+old_key, slot, -1);
        atomicExch(readers+slot, -1);
    }
}
// MXFP4 tile epilogues: no F32 hidden spill, no per-expert down output spill.
__global__ void mxfp4_gu(const char * const * cache,size_t up_offset,const int32_t * plan,
                        const block_q8_1 * xq,block_q8_1 * hq,int ne,int nf,float limit) {
    const int expert=blockIdx.y,slot=plan[expert];if(slot<0)return;
    const int lane=threadIdx.x&31,warp=threadIdx.x/32,row0=blockIdx.x*32+warp*4;
    const char * g=cache[slot];
    const char * u=g+up_offset;const int nb=ne/QK_MXFP4;
    __shared__ float hidden[32];
    float a[4]={},b[4]={};
    for(int k=lane/(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ);k<nb;k+=VDR_MXFP4_Q8_1_MMVQ*32/QI_MXFP4){
        const int q=VDR_MXFP4_Q8_1_MMVQ*(lane%(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ));
#pragma unroll
        for(int j=0;j<4;++j){const int index=(row0+j)*nb+k;a[j]+=vec_dot_mxfp4_q8_1(g,xq+k,index,q);b[j]+=vec_dot_mxfp4_q8_1(u,xq+k,index,q);}
    }
#pragma unroll
    for(int j=0;j<4;++j){a[j]=warp_reduce_sum<32>(a[j]);b[j]=warp_reduce_sum<32>(b[j]);if(!lane){const float gv=fminf(a[j],limit),uv=fminf(fmaxf(b[j],-limit),limit);hidden[warp*4+j]=(gv/(1.f+expf(-gv)))*uv;}}
    __syncthreads();
    if(!warp){const float v=hidden[lane],maximum=warp_reduce_max<32>(fabsf(v)),sum=warp_reduce_sum<32>(v);const float scale=maximum/127.f;
        auto * dest=hq+expert*(nf/32)+blockIdx.x;dest->qs[lane]=maximum==0?0:roundf(v/scale);if(!lane)dest->ds=make_half2(scale,sum);}
}
__global__ void mxfp4_down_sum(const char * const * cache,size_t down_offset,const int32_t * plan,
                              const float * weights,const block_q8_1 * hq,int nf,int ne,int nk,float * out) {
    const int lane=threadIdx.x&31,expert=threadIdx.x/32,row0=blockIdx.x*4,nb=nf/QK_MXFP4;
    __shared__ float parts[8][4];float a[4]={};
    if(expert<nk && plan[expert]>=0){const int slot=plan[expert];
        const char * d=cache[slot]+down_offset;const auto * x=hq+expert*nb;
        for(int k=lane/(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ);k<nb;k+=VDR_MXFP4_Q8_1_MMVQ*32/QI_MXFP4){const int q=VDR_MXFP4_Q8_1_MMVQ*(lane%(QI_MXFP4/VDR_MXFP4_Q8_1_MMVQ));
#pragma unroll
            for(int j=0;j<4;++j)a[j]+=vec_dot_mxfp4_q8_1(d,x+k,(row0+j)*nb+k,q);}}
#pragma unroll
    for(int j=0;j<4;++j){const float v=warp_reduce_sum<32>(a[j]);if(!lane)parts[expert][j]=v;}
    __syncthreads();if(threadIdx.x<4){float v=0;for(int e=0;e<nk;++e)if(plan[e]>=0)v+=weights[e]*parts[e][threadIdx.x];out[row0+threadIdx.x]=v;}
}

struct descriptor {ggml_tensor g,u,d;};
struct job {int slot,key,old_key;bool predicted=false;};
// Host clocks only: callback entry proves the request is visible, but is NOT
// the GPU publication timestamp. In particular, callback wait overlaps experts.
using service_clock = std::chrono::steady_clock;
static uint64_t service_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(service_clock::now().time_since_epoch()).count();
}
static uint64_t service_cpu_ns() {
#if defined(__linux__)
    timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;
#else
    return 0;
#endif
}
struct service_metric {
    uint64_t count=0,total=0,maximum=0;
    std::array<uint64_t,32> bins{};
    void add(uint64_t ns) {
        ++count;total+=ns;maximum=std::max(maximum,ns);
        unsigned bin=0;for(uint64_t us=ns/1000;us && bin+1<bins.size();us>>=1)++bin;
        ++bins[bin];
    }
    double p95_upper_us() const {
        uint64_t n=0;for(unsigned i=0;i<bins.size();++i){n+=bins[i];if(n>=(count*95+99)/100)return double(uint64_t(1)<<i);}
        return 0;
    }
    void print(const char * format,const char * stage,const char * scope) const {
        std::fprintf(stderr,"[dsv4f-service-profile] format=%s scope=%s stage=%s n=%llu mean_us=%.3f p95_upper_us=%.0f max_us=%.3f\n",
            format,scope,stage,(unsigned long long)count,count?total/1000./count:0.,count?p95_upper_us():0.,maximum/1000.);
    }
};
class service {
    std::mutex mutex,answer_mutex,wake_mutex;
    std::condition_variable copies,answered,ready;
    std::thread worker,loader,lookahead_loader;
    std::atomic<bool> stop{false};
    std::deque<descriptor> layers;
    std::unordered_map<const void*,int> layer_ids;
    ggml_dsv4f_cache_policy policy;
    ggml_dsv4f_lookahead_guard guard{policy};
    uint64_t guard_hits=0,guard_cancelled=0,guard_reclaim_denied=0,guard_compacted=0;
    std::unordered_set<int> pending;
    std::deque<job> queue,lookahead_queue;
    std::vector<ggml_dsv4f_cache_policy::slot> & slots=policy.slots;
    FILE * trace=nullptr;
    request * host=nullptr,*device=nullptr;
    unsigned long long * seq=nullptr,*issued=nullptr;
    int * table=nullptr,*readers=nullptr,*claim_h=nullptr,*claim_d=nullptr;
    int * speculative=nullptr;
    int * lookahead_claim_h=nullptr,*lookahead_claim_d=nullptr;
    // 1/2: Q3 temporal; 3/4: early-router shadow/copy; 5: no-copy cache guard.
    // Predictions never add frequency to the actual-demand cache policy.
    int lookahead=0;
    int early_depth=8,early_read=8;
    int temporary_slots=0,prefetch_cursor=0;
    bool prefetch_pending=false;
    std::array<bool,32> temporary_busy{};
    std::array<std::vector<int>,MAX_LAYERS> early_ids;
    std::array<std::vector<int>,MAX_LAYERS> early_read_ids;
    std::array<bool,MAX_LAYERS> early_valid{};
    uint64_t early_batches=0,early_candidates=0,early_matches=0,early_actual=0;
    uint64_t early_ready=0,early_matched_ready=0,early_actual_miss=0,early_matched_miss=0;
    uint64_t early_new_candidates=0,early_cached_candidates=0;
    uint64_t early_armed_batches=0,early_skipped_batches=0;
    uint64_t early_read_candidates=0,early_read_matches=0,early_read_ready=0,early_read_miss=0;
    std::array<std::vector<int>,MAX_LAYERS> last_routes;
    std::array<int,MAX_LAYERS> forecast;
    uint64_t predictions=0,forecast_used=0,forecast_missed=0,forecast_consumed=0;
    uint64_t prefetch_copies=0,prefetch_hits=0,prefetch_timely=0,prefetch_rejected=0,prefetch_failed=0;
    double prefetch_copy_us=0;
    int * prefill_updates=nullptr;
    std::vector<int> prefill_changes;
    bool prefill_leased=false;
    char * slab=nullptr;
    char ** extra=nullptr;
    struct cache_block { char * data; size_t count; };
    std::vector<cache_block> cache_blocks;
    // Fused kernels always read through this stable GPU pointer table, even
    // for the initial cache. Graph replay survives shrinking and regrowing it.
    std::vector<char *> cache_ptrs;
    int base_slots=0,max_slots=0;
    size_t format_ceiling=SIZE_MAX;
    size_t reclaim_free=0,reclaim_hip_free=0;
    float * gate=nullptr,*down=nullptr,*weights=nullptr;
    int32_t * plan=nullptr;
    void * aq=nullptr,*hq=nullptr;
    cudaStream_t copy=nullptr,lookahead_copy=nullptr;
    int ne=0,nf=0;
    size_t gs=0,us=0,ds=0,uo=0,doff=0,stride=0;
    ggml_type gt{},ut{},dt{};
    const void * selected_source=nullptr;
    bool configured=false;
    bool poll=true;
    bool notify_wait=false,profile=false;
    unsigned callback_spin_us=0;
    unsigned idle_spins=200000;
    std::atomic<unsigned long long> completed{0};
    // Separate from the mapped ring: no new GPU allocation or device writes.
    struct stamps {std::atomic<uint64_t> seen{0},done{0};};
    std::array<stamps,RING> timing;
    service_metric descriptor_time,expert_time,cpu_lane_wait,publish_time,post_time,sleep_time,detect_bound,idle_cpu;
    service_metric callback_time,callback_detect,callback_tail;
    uint64_t callbacks_pending=0,callbacks_unseen=0,callbacks=0;
    void print_worker_profile(const char * scope) {
        for(auto entry:{std::make_pair("descriptor_lock",&descriptor_time),{"experts",&expert_time},{"cpu_lane_wait",&cpu_lane_wait},{"publish",&publish_time},
                       {"post_publish",&post_time},{"idle_sleep",&sleep_time},{"detect_upper_bound_after_sleep",&detect_bound},{"idle_thread_cpu",&idle_cpu}})
            entry.second->print(ggml_type_name(gt),entry.first,scope);
    }
    void print_callback_profile() {
        for(auto entry:{std::make_pair("callback",&callback_time),{"visible_to_seen_lower_bound",&callback_detect},{"done_to_callback_return",&callback_tail}})
            entry.second->print(ggml_type_name(gt),entry.first,"final");
        std::fprintf(stderr,"[dsv4f-service-profile] format=%s callbacks=%llu pending_at_entry=%llu unseen_at_entry=%llu\n",
            ggml_type_name(gt),(unsigned long long)callbacks,(unsigned long long)callbacks_pending,(unsigned long long)callbacks_unseen);
    }
    bool fused_mxfp4=false,block_cache=false,v41=false;
    uint64_t routes=0,hits=0,admissions=0,admit_bytes=0,requests=0;
    double cpu_ms=0;
    // Reserve temporary slots INSIDE the existing format budget. Host ready=false
    // excludes them from demand-policy victims; GPU table/readers still decide
    // whether a completed temporary entry can be consumed safely.
    void reserve_temporary() {
        for(int i=0;i<temporary_slots && i<int(slots.size());++i)slots[i].ready=true;
        temporary_slots=lookahead==4?std::min(2*early_depth,int(slots.size())):
            lookahead==2 && !slots.empty()?std::min(4,std::max(1,int(slots.size())/8)):0;
        for(int i=0;i<temporary_slots;++i)slots[i].ready=false;
        prefetch_cursor=0;
    }
    bool temporary_contains(int key) const {
        for(int i=0;i<temporary_slots;++i)if(slots[i].key==key)return true;
        return false;
    }
    bool admit(const int * misses,int n,bool predicted=false,int preferred=-1) {
        auto & jobs=predicted && lookahead==4?lookahead_queue:queue;
        if(jobs.size()>=size_t(predicted && lookahead==4?early_read:8))return false;
        ggml_dsv4f_cache_policy::decision chosen;
        if(predicted) {
            if(n!=1 || !temporary_slots || (lookahead==2 && prefetch_pending) || pending.count(misses[0]) ||
                policy.contains(misses[0]) || temporary_contains(misses[0]))return false;
            int index=preferred;
            if(index<0){index=prefetch_cursor;prefetch_cursor=(prefetch_cursor+1)%temporary_slots;}
            if(index>=temporary_slots || temporary_busy[index])return false;
            chosen={misses[0],index};temporary_busy[index]=true;prefetch_pending=true;
        } else chosen=policy.select(misses,n,pending);
        if(chosen.index<0)return false;
        const int key=chosen.key,victim=chosen.index;
        pending.insert(key);slots[victim].ready=false;jobs.push_back({victim,key,slots[victim].key,predicted});
        if(trace)std::fprintf(trace,"Q %d %d %d\n",key,slots[victim].key,victim);
        copies.notify_all();return true;
    }
    // mutex held; submit before current CPU experts so the loader can overlap
    // them and the following attention. No blocking GPU readback is introduced.
    void predict(const request & r) {
        if(!lookahead)return;
        prefetch_hits+=__builtin_popcountll(r.prefetch_hits);
        if(lookahead>=3) {
            if(r.single && early_valid[r.layer]) {
                const auto & predicted=early_ids[r.layer];
                for(int k=0;k<r.nk;++k)if(r.weights[k]!=0) {
                    ++early_actual;const bool hit=(r.hits>>k)&1;early_ready+=hit;early_actual_miss+=!hit;
                    const int key=r.layer*MAX_EXPERTS+r.ids[k];
                    if(lookahead==5 && hit && policy.protected_key(key))++guard_hits;
                    const auto & selected=early_read_ids[r.layer];
                    if(std::find(selected.begin(),selected.end(),key)!=selected.end()) {
                        ++early_read_matches;early_read_ready+=hit;early_read_miss+=!hit;
                    }
                    if(std::find(predicted.begin(),predicted.end(),key)!=predicted.end()) {
                        ++early_matches;early_matched_ready+=hit;early_matched_miss+=!hit;
                    }
                }
            }
            if(lookahead==5){if(r.single)guard.consume(r.layer);else guard.clear_layer(r.layer);}
            early_valid[r.layer]=false;return;
        }
        if(!r.single){forecast[r.layer]=-1;last_routes[r.layer].clear();return;}
        const int expected=forecast[r.layer];
        if(expected>=0) {
            ++forecast_consumed;
            for(int i=0;i<r.nk;++i)if(r.weights[i]!=0 && r.layer*MAX_EXPERTS+r.ids[i]==expected) {
                ++forecast_used;if(!(r.hits&(uint64_t(1)<<i)))++forecast_missed;
                if(r.prefetch_hits&(uint64_t(1)<<i))++prefetch_timely;break;
            }
        }
        forecast[r.layer]=-1;
        auto & previous=last_routes[r.layer];previous.clear();
        for(int i=0;i<r.nk;++i)if(r.weights[i]!=0)previous.push_back(r.layer*MAX_EXPERTS+r.ids[i]);
        const int target=r.layer+1;
        if(target>=int(layers.size()) || slots.empty())return;
        int best=-1;
        for(int key:last_routes[target])if(!policy.contains(key) && !temporary_contains(key) && !pending.count(key) &&
            (best<0 || policy.score(key)>policy.score(best)))best=key;
        if(best<0)return;
        forecast[target]=best;++predictions;
        if(lookahead==2 && !admit(&best,1,true))++prefetch_rejected;
    }
    void print_lookahead() const {
        if(!lookahead)return;
        if(lookahead==5)std::fprintf(stderr,"[dsv4f-protect] format=%s predictions=%llu resident_leases=%llu retained_next=%llu lost_next=%llu protected_hits=%llu held_keys=%zu scan_skips=%llu no_victim=%llu queued_cancelled=%llu reclaim_denied=%llu compacted=%llu routes=%llu hits=%llu admissions=%llu demand_copy_mib=%.3f\n",
            ggml_type_name(gt),(unsigned long long)guard.predictions,(unsigned long long)guard.resident_leases,
            (unsigned long long)guard.retained_next,(unsigned long long)guard.lost_next,(unsigned long long)guard_hits,
            policy.protection_count(),(unsigned long long)policy.pin_scan_skips,(unsigned long long)policy.pin_no_victim,
            (unsigned long long)guard_cancelled,(unsigned long long)guard_reclaim_denied,(unsigned long long)guard_compacted,
            (unsigned long long)routes,(unsigned long long)hits,(unsigned long long)admissions,double(admit_bytes)/1048576.);
        if(lookahead>=3) {
            std::fprintf(stderr,"[dsv4f-early] mode=%d format=%s depth=%d read=%d read_candidates=%llu read_matches=%llu read_ready=%llu read_miss=%llu batches=%llu armed_batches=%llu skipped_batches=%llu candidates=%llu matches=%llu actual=%llu actual_ready=%llu matched_ready=%llu actual_miss=%llu matched_miss=%llu new_candidates=%llu cached_candidates=%llu copies=%llu first_hits=%llu rejected=%llu failed=%llu copy_mib=%.3f copy_mean_us=%.3f\n",
                lookahead,ggml_type_name(gt),early_depth,early_read,(unsigned long long)early_read_candidates,
                (unsigned long long)early_read_matches,(unsigned long long)early_read_ready,(unsigned long long)early_read_miss,
                (unsigned long long)early_batches,(unsigned long long)early_armed_batches,
                (unsigned long long)early_skipped_batches,(unsigned long long)early_candidates,
                (unsigned long long)early_matches,(unsigned long long)early_actual,(unsigned long long)early_ready,
                (unsigned long long)early_matched_ready,(unsigned long long)early_actual_miss,(unsigned long long)early_matched_miss,
                (unsigned long long)early_new_candidates,(unsigned long long)early_cached_candidates,
                (unsigned long long)prefetch_copies,(unsigned long long)prefetch_hits,(unsigned long long)prefetch_rejected,
                (unsigned long long)prefetch_failed,double(prefetch_copies)*(gs+us+ds)/1048576.,prefetch_copies?prefetch_copy_us/prefetch_copies:0.);
            return;
        }
        std::fprintf(stderr,"[dsv4f-lookahead] mode=%d format=%s requests=%llu predictions=%llu consumed=%llu selected=%llu selected_miss=%llu copies=%llu first_hits=%llu timely=%llu rejected=%llu failed=%llu copy_mib=%.3f copy_mean_us=%.3f\n",
            lookahead,ggml_type_name(gt),(unsigned long long)requests,(unsigned long long)predictions,
            (unsigned long long)forecast_consumed,(unsigned long long)forecast_used,(unsigned long long)forecast_missed,
            (unsigned long long)prefetch_copies,(unsigned long long)prefetch_hits,(unsigned long long)prefetch_timely,(unsigned long long)prefetch_rejected,
            (unsigned long long)prefetch_failed,double(prefetch_copies)*(gs+us+ds)/1048576.,prefetch_copies?prefetch_copy_us/prefetch_copies:0.);
    }
    // mutex held. Keep the complete top-k for recall/precision, including hot
    // cache hits. Only missing experts require a physical weight transfer.
    void early_predict(const request & r) {
        if(lookahead<3)return;
        ++early_batches;auto & ids=early_ids[r.layer];ids.clear();early_valid[r.layer]=true;
        auto & selected=early_read_ids[r.layer];selected.clear();
        if(lookahead==5)guard.begin(r.layer);
        // Match OpenBMB arm(): one batch in flight; if it has not completed,
        // skip the new batch rather than accumulating stale layer predictions.
        const bool busy=std::any_of(temporary_busy.begin(),temporary_busy.end(),[](bool b){return b;});
        bool has_new=false,armed=false;
        int next_slot=0;
        const int bank_begin=r.bank*early_depth,bank_end=std::min(bank_begin+early_depth,temporary_slots);
        for(int k=0;k<r.nk;++k)if(r.ids[k]>=0 && r.ids[k]<layers[r.layer].g.ne[2]) {
            const int key=r.layer*MAX_EXPERTS+r.ids[k];ids.push_back(key);++early_candidates;
            if(lookahead==5)guard.add(r.layer,key);
            // Rank is global, BEFORE format/cache filtering. A resident top-1
            // never causes a lower-ranked expert to be copied as a replacement.
            if(k>=early_read)continue;
            selected.push_back(key);++early_read_candidates;
            if(policy.contains(key) || temporary_contains(key)){++early_cached_candidates;continue;}
            ++early_new_candidates;has_new=true;
            if(lookahead!=4)continue;
            if(busy){++prefetch_rejected;continue;}
            int slot=-1;
            for(int i=bank_begin+next_slot;i<bank_end;++i)if(!temporary_busy[i]){slot=i;next_slot=i-bank_begin+1;break;}
            if(slot<0 || !admit(&key,1,true,slot))++prefetch_rejected;
            else armed=true;
        }
        early_armed_batches+=armed;early_skipped_batches+=lookahead==4 && busy && has_new;
    }
    void serve() {
        unsigned long long next=1;unsigned idle=0;uint64_t last_sleep_start=0;
        uint64_t idle_cpu_start=profile?service_cpu_ns():0;
        while(!stop.load(std::memory_order_acquire)) {
            request * r=host+(next%RING);
            // Bracket the failed read BEFORE sleeping; no clock call in the
            // normal busy polling loop. GPU publication lies after that read.
            const uint64_t probe=profile && idle>=idle_spins?service_ns():0;
            if(r->seq!=next){
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
                last_sleep_start=0;
                if(++idle>idle_spins){
                    if(profile)last_sleep_start=probe;
                    if(notify_wait && !poll) {
                        // callback is a fallback wake after existing GPU work;
                        // polling still starts CPU experts as early as before.
                        // The same mutex in callback prevents a lost wakeup.
                        std::unique_lock<std::mutex> lock(wake_mutex);
                        ready.wait_for(lock,std::chrono::microseconds(100),[&]{return r->seq==next || stop.load();});
                    } else std::this_thread::sleep_for(std::chrono::microseconds(100));
                    if(profile)sleep_time.add(service_ns()-last_sleep_start);
                    idle=idle_spins;
                }continue;
            }
            idle=0;std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t seen=profile?service_ns():0;
            if(profile)idle_cpu.add(service_cpu_ns()-idle_cpu_start);
            if(profile){timing[next%RING].seen.store(seen,std::memory_order_relaxed);if(last_sleep_start)detect_bound.add(seen-last_sleep_start);}
            last_sleep_start=0;
            if(r->kind==1) {
                {std::lock_guard<std::mutex> lock(mutex);early_predict(*r);}
                std::atomic_thread_fence(std::memory_order_release);r->consumed=next;++next;
                if(profile)idle_cpu_start=service_cpu_ns();
                continue;
            }
            descriptor layer;
            {std::lock_guard<std::mutex> lock(mutex);layer=layers[r->layer];predict(*r);}
            if(profile)descriptor_time.add(service_ns()-seen);
            const uint64_t all=(uint64_t(1)<<r->nk)-1;
            std::unique_lock<std::mutex> cpu_lane(cpu_lane_mutex,std::defer_lock);
            if(r->hits!=all) {
                const uint64_t wait_start=profile?service_ns():0;
                {dsv4f_timeline::scope waiting("wait/cpu_lane");cpu_lane.lock();}
                if(profile)cpu_lane_wait.add(service_ns()-wait_start);
            }
            const auto start=std::chrono::steady_clock::now();
            {
            dsv4f_timeline::scope span(gt==GGML_TYPE_Q3_K?"experts/q3_K":gt==GGML_TYPE_Q2_0?"experts/q2_0":gt==GGML_TYPE_PTQ1_0?"experts/ptq1_0":"experts/other");
            if(r->hits!=all)ggml_dsv4f_cpu_experts(&layer.g,&layer.u,&layer.d,r->x,r->ids,r->weights,r->nk,r->hits,r->limit,r->answer);
            else std::fill_n(r->answer,ne,0.f);
            }
            const auto end=service_clock::now();
            if(cpu_lane.owns_lock())cpu_lane.unlock();
            cpu_ms+=std::chrono::duration<double,std::milli>(end-start).count();
            if(profile)expert_time.add(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
            uint64_t done=0;
            {std::lock_guard<std::mutex> lock(answer_mutex);std::atomic_thread_fence(std::memory_order_release);r->done=next;
                if(profile){done=service_ns();timing[next%RING].done.store(done,std::memory_order_relaxed);}
                // Host-only acquire/release mirror permits a callback fast path.
                if(notify_wait || callback_spin_us || profile)completed.store(next,std::memory_order_release);
            }answered.notify_all();
            const uint64_t published=profile?service_ns():0;
            if(profile)publish_time.add(published-std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count());
            {std::lock_guard<std::mutex> lock(mutex);
                ++requests;
                int keys[MAX_USED],active[MAX_USED],misses[MAX_USED],nactive=0,nmiss=0;
                for(int i=0;i<r->nk;++i){
                    keys[i]=r->layer*MAX_EXPERTS+r->ids[i];
                    // Other format classes are represented by zero-weight routes.
                    // They must not inflate hit counters or heat dummy expert zero.
                    if(r->weights[i]==0)continue;
                    active[nactive++]=keys[i];hits+=(r->hits>>i)&1;
                    if(!((r->hits>>i)&1))misses[nmiss++]=keys[i];
                }
                routes+=nactive;
                policy.observe(active,nactive);
                if(trace){std::fprintf(trace,"R %llu",(unsigned long long)r->hits);for(int i=0;i<r->nk;++i)std::fprintf(trace," %d",keys[i]);std::fprintf(trace,"\n");}
                admit(misses,nmiss);
                if(lookahead && requests%4300==0)print_lookahead();
                if(requests%4300==0)std::fprintf(stderr,"[dsv4f-service] routes=%llu hit=%.2f%% cpu_ms/token=%.3f admissions=%llu copy_mib=%.2f queued=%zu format=%s\n",(unsigned long long)routes,100.*hits/std::max<uint64_t>(routes,1),(v41?double(layers.size()):43.)*cpu_ms/requests,(unsigned long long)admissions,double(admit_bytes)/(1<<20),queue.size(),ggml_type_name(gt));
            }
            std::atomic_thread_fence(std::memory_order_release);r->consumed=next;++next;
            if(profile)post_time.add(service_ns()-published);
            if(profile)idle_cpu_start=service_cpu_ns();
        }
        if(profile)print_worker_profile("final");
    }
    void load(bool predicted_lane=false) {
        ggml_cuda_set_device(0);
        auto & jobs=predicted_lane?lookahead_queue:queue;
        const auto work_stream=predicted_lane?lookahead_copy:copy;
        auto * claimed_host=predicted_lane?lookahead_claim_h:claim_h;
        auto * claimed_device=predicted_lane?lookahead_claim_d:claim_d;
        std::unique_lock<std::mutex> lock(mutex);
        while(true) {
            copies.wait(lock,[&]{return stop.load()||!jobs.empty();});if(stop.load())return;
            job j=jobs.front();jobs.pop_front();descriptor l=layers[j.key/MAX_EXPERTS];
            const bool guarded=lookahead==5;
            if(guarded && policy.protected_key(j.old_key)) {
                slots[j.slot].ready=true;pending.erase(j.key);++guard_cancelled;copies.notify_all();continue;
            }
            // Linearize lease grants with GPU invalidation. In guard mode the
            // mutex covers only reservation; bulk weight DMA stays outside it.
            if(!guarded)lock.unlock();
            const auto copy_start=j.predicted?service_clock::now():service_clock::time_point{};
            reserve_slot<<<1,32,0,work_stream>>>(readers,table,j.slot,j.old_key,claimed_device);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(work_stream));
            const bool success=*claimed_host!=0;
            if(guarded){if(success)policy.unpublish(j.slot);lock.unlock();}
            if(success){const int eid=j.key%MAX_EXPERTS;
                char * dest=block_cache?cache_ptrs[j.slot]:slab+j.slot*stride;
                CUDA_CHECK(cudaMemcpyAsync(dest,(const char*)l.g.data+eid*gs,gs,cudaMemcpyHostToDevice,work_stream));
                CUDA_CHECK(cudaMemcpyAsync(dest+uo,(const char*)l.u.data+eid*us,us,cudaMemcpyHostToDevice,work_stream));
                CUDA_CHECK(cudaMemcpyAsync(dest+doff,(const char*)l.d.data+eid*ds,ds,cudaMemcpyHostToDevice,work_stream));
                publish_slot<<<1,32,0,work_stream>>>(readers,table,j.slot,j.key,speculative,j.predicted);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(work_stream));
            }
            lock.lock();if(j.predicted){if(success){++prefetch_copies;prefetch_copy_us+=std::chrono::duration<double,std::micro>(service_clock::now()-copy_start).count();}else ++prefetch_failed;}
            if(success){policy.publish(j.slot,j.key);++admissions;admit_bytes+=gs+us+ds;}
            slots[j.slot].ready=j.slot>=temporary_slots;
            if(j.predicted){prefetch_pending=false;temporary_busy[j.slot]=false;}
            pending.erase(j.key);copies.notify_all();
            if(trace)std::fprintf(trace,"A %d %d\n",j.key,int(success));
        }
    }
    static void callback(void * arg) {
        auto & self=*static_cast<service*>(arg);
        dsv4f_timeline::scope span(self.gt==GGML_TYPE_Q3_K?"callback/q3_K":self.gt==GGML_TYPE_Q2_0?"callback/q2_0":self.gt==GGML_TYPE_PTQ1_0?"callback/ptq1_0":"callback/other");
        const uint64_t entry=self.profile?service_ns():0;
        // All preceding device work has completed, so the latest request is
        // visible. No CUDA/HIP API may be called from this runtime callback.
        unsigned long long latest=0;
        for(int i=0;i<RING;++i)latest=std::max(latest,(unsigned long long)(self.host[i].seq));
        const bool fast=self.notify_wait || self.callback_spin_us;
        if(self.profile){++self.callbacks;if(self.completed.load(std::memory_order_acquire)<latest)++self.callbacks_pending;}
        if(self.notify_wait && self.completed.load(std::memory_order_acquire)<latest) {
            {std::lock_guard<std::mutex> lock(self.wake_mutex);}
            self.ready.notify_one();
        }
        if(self.callback_spin_us) {
            const auto deadline=service_clock::now()+std::chrono::microseconds(self.callback_spin_us);
            while(self.completed.load(std::memory_order_acquire)<latest && !self.stop.load() && service_clock::now()<deadline) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
            }
        }
        if(!fast || self.completed.load(std::memory_order_acquire)<latest) {
            std::unique_lock<std::mutex> lock(self.answer_mutex);
            self.answered.wait(lock,[&]{return self.host[latest%RING].done==latest || self.stop.load();});
        }
        if(self.profile) {
            const uint64_t returned=service_ns();
            // answer_mutex (legacy) or completed acquire (fast) publishes stamps.
            self.completed.load(std::memory_order_acquire);
            const uint64_t seen=self.timing[latest%RING].seen.load(std::memory_order_relaxed);
            const uint64_t done=self.timing[latest%RING].done.load(std::memory_order_relaxed);
            self.callback_time.add(returned-entry);
            if(seen>entry){++self.callbacks_unseen;self.callback_detect.add(seen-entry);}
            self.callback_tail.add(returned-std::max(entry,done));
        }
    }
    void init(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,int nk) {
        if(host){GGML_ASSERT(g->type==gt && u->type==ut && d->type==dt && g->ne[0]==ne && g->ne[1]==nf);return;}
        const char * cache_mode=std::getenv("DSV4F_CACHE_POLICY");if(!cache_mode)cache_mode="legacy";
        ggml_dsv4f_cache_policy::mode selected;
        if(!std::strcmp(cache_mode,"legacy"))selected=ggml_dsv4f_cache_policy::mode::legacy;
        else if(!std::strcmp(cache_mode,"scored"))selected=ggml_dsv4f_cache_policy::mode::scored;
        else if(!std::strcmp(cache_mode,"aged"))selected=ggml_dsv4f_cache_policy::mode::aged;
        else throw std::runtime_error("DSV4F_CACHE_POLICY must be legacy, scored or aged");
        auto setting=[](const char * name,unsigned fallback,unsigned maximum) {
            const char * s=std::getenv(name);if(!s)return fallback;
            char * end=nullptr;const unsigned long v=std::strtoul(s,&end,10);
            if(end==s || *end || *s=='-' || v<1 || v>maximum)throw std::runtime_error(std::string("invalid cache setting: ")+name);
            return unsigned(v);
        };
        const unsigned decay=setting("DSV4F_CACHE_DECAY_TOKENS",256,65536);
        double ratio=1.25;
        if(const char * value=std::getenv("DSV4F_CACHE_REPLACE_RATIO")) {
            char * end=nullptr;ratio=std::strtod(value,&end);
            if(end==value || *end || !std::isfinite(ratio) || ratio<1 || ratio>16)throw std::runtime_error("invalid DSV4F_CACHE_REPLACE_RATIO");
        }
        policy.configure(selected,43ULL*decay,ratio);
        std::fprintf(stderr,"[dsv4f-service] cache_policy=%s decay_tokens=%u replacement_ratio=%.3f\n",cache_mode,decay,ratio);
        if(const char * path=std::getenv("DSV4F_CACHE_TRACE")) {
            trace=std::fopen(path,"w");if(!trace)throw std::runtime_error("cannot open DSV4F_CACHE_TRACE");
            std::setvbuf(trace,nullptr,_IOFBF,1<<20);
        }
        ne=g->ne[0];nf=g->ne[1];gt=g->type;ut=u->type;dt=d->type;gs=g->nb[2];us=u->nb[2];ds=d->nb[2];
        const size_t align=std::lcm(std::lcm(ggml_type_size(gt),ggml_type_size(ut)),ggml_type_size(dt));
        uo=(gs+ggml_type_size(ut)-1)/ggml_type_size(ut)*ggml_type_size(ut);
        doff=(uo+us+ggml_type_size(dt)-1)/ggml_type_size(dt)*ggml_type_size(dt);stride=(doff+ds+align-1)/align*align;
        const char * mode=std::getenv("DSV4F_GPU_WAIT");poll=!mode || std::strcmp(mode,"callback")!=0;
        const char * wait=std::getenv("DSV4F_SERVICE_WAIT");
        if(wait && std::strcmp(wait,"legacy") && std::strcmp(wait,"notify"))throw std::runtime_error("DSV4F_SERVICE_WAIT must be legacy or notify");
        notify_wait=wait && !std::strcmp(wait,"notify");
        callback_spin_us=0;
        if(const char * value=std::getenv("DSV4F_SERVICE_CALLBACK_SPIN_US")) {
            char * end=nullptr;const unsigned long v=std::strtoul(value,&end,10);
            if(end==value || *end || *value=='-' || v>1000)throw std::runtime_error("DSV4F_SERVICE_CALLBACK_SPIN_US must be in 0..1000");
            callback_spin_us=v;
        }
        idle_spins=200000;
        if(const char * value=std::getenv("DSV4F_SERVICE_IDLE_SPINS")) {
            char * end=nullptr;const unsigned long v=std::strtoul(value,&end,10);
            if(end==value || *end || *value=='-' || v>2000000)throw std::runtime_error("DSV4F_SERVICE_IDLE_SPINS must be in 0..2000000");
            idle_spins=v;
        }
        profile=std::getenv("DSV4F_SERVICE_PROFILE") && std::atoi(std::getenv("DSV4F_SERVICE_PROFILE"));
        const char * fusion=std::getenv("DSV4F_MXFP4_FUSED");fused_mxfp4=nk<=8 && gt==GGML_TYPE_MXFP4 && dt==GGML_TYPE_MXFP4 && ne%128==0 && nf%32==0 && (!fusion || atoi(fusion)!=0);
        v41=gt==GGML_TYPE_Q3_K || gt==GGML_TYPE_Q2_0 || gt==GGML_TYPE_PTQ1_0;
        block_cache=fused_mxfp4 || v41;
        int requested_lookahead=0;
        if(const char * value=std::getenv("DSV4F_EXPERT_LOOKAHEAD")) {
            if(std::strcmp(value,"0") && std::strcmp(value,"1") && std::strcmp(value,"2") && std::strcmp(value,"3") && std::strcmp(value,"4") && std::strcmp(value,"5"))throw std::runtime_error("DSV4F_EXPERT_LOOKAHEAD must be 0..5");
            requested_lookahead=std::atoi(value);
        }
        lookahead=((requested_lookahead>=3 && v41) || gt==GGML_TYPE_Q3_K)?requested_lookahead:0;forecast.fill(-1);
        early_depth=setting("DSV4F_EXPERT_LOOKAHEAD_DEPTH",8,16);
        early_read=setting("DSV4F_EXPERT_LOOKAHEAD_READ",early_depth,early_depth);
        const size_t scratch=5*sizeof(unsigned long long)+MAX_LAYERS*MAX_EXPERTS*sizeof(int)*(1+(lookahead==2 || lookahead==4))+
            MAX_USED*(size_t(nf+ne+1)*sizeof(float)+sizeof(int32_t))+
            (GGML_PAD(ne,MATRIX_ROW_PADDING)+MAX_USED*GGML_PAD(nf,MATRIX_ROW_PADDING))/QK8_1*sizeof(block_q8_1)+
            43*g->ne[2]*(sizeof(char *)+4*sizeof(int));
        if(!ggml_dsv4f_reserve_vram(0,scratch))throw std::runtime_error("DSV4F VRAM headroom cannot be preserved for service scratch");
        CUDA_CHECK(cudaHostAlloc((void**)&host,RING*sizeof(request),cudaHostAllocMapped));std::memset(host,0,RING*sizeof(request));CUDA_CHECK(cudaHostGetDevicePointer((void**)&device,host,0));
        CUDA_CHECK(cudaMalloc((void**)&issued,4*sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&seq,sizeof(unsigned long long)));CUDA_CHECK(cudaMemset(seq,0,sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc((void**)&table,MAX_LAYERS*MAX_EXPERTS*sizeof(int)));CUDA_CHECK(cudaMemset(table,0xff,MAX_LAYERS*MAX_EXPERTS*sizeof(int)));
        if(lookahead==2 || lookahead==4){CUDA_CHECK(cudaMalloc((void**)&speculative,MAX_LAYERS*MAX_EXPERTS*sizeof(int)));CUDA_CHECK(cudaMemset(speculative,0,MAX_LAYERS*MAX_EXPERTS*sizeof(int)));}
        CUDA_CHECK(cudaMalloc((void**)&gate,MAX_USED*nf*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&down,MAX_USED*ne*sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&weights,MAX_USED*sizeof(float)));CUDA_CHECK(cudaMalloc((void**)&plan,MAX_USED*sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&aq,GGML_PAD(ne,MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)));
        CUDA_CHECK(cudaMalloc(&hq,MAX_USED*GGML_PAD(nf,MATRIX_ROW_PADDING)/QK8_1*sizeof(block_q8_1)));
        CUDA_CHECK(cudaHostAlloc((void**)&claim_h,sizeof(int),cudaHostAllocMapped));CUDA_CHECK(cudaHostGetDevicePointer((void**)&claim_d,claim_h,0));
        max_slots=43*MAX_EXPERTS;
        if(v41) {
            const size_t ceiling=ggml_dsv4f_env_mib("DSV4F_EXPERT_CACHE_MIB",SIZE_MAX);
            const size_t free=ggml_dsv4f_free_vram(),reserve=ggml_dsv4f_headroom()+ggml_dsv4f_guard();
            const size_t share=(ceiling==SIZE_MAX?(free>reserve?free-reserve:0):ceiling)/3;
            const char * setting=gt==GGML_TYPE_Q3_K?"DSV4F_V41_CACHE_Q3_MIB":gt==GGML_TYPE_Q2_0?"DSV4F_V41_CACHE_Q2_MIB":"DSV4F_V41_CACHE_PTQ_MIB";
            const size_t capacity=ggml_dsv4f_env_mib(setting,share);
            // Preserve the preceding minimum-one-slot policy for automatic
            // shares. Explicit per-format ceilings, including zero, are strict.
            format_ceiling=std::getenv(setting)?capacity:SIZE_MAX;
            max_slots=std::min<size_t>(max_slots,capacity/stride);
            max_slots=std::max(max_slots,1); // metadata allocations remain valid for a zero-byte cache ceiling.
        }
        CUDA_CHECK(cudaMalloc((void**)&prefill_updates,3*max_slots*sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&extra,max_slots*sizeof(char *)));
        CUDA_CHECK(cudaMalloc((void**)&readers,max_slots*sizeof(int)));CUDA_CHECK(cudaMemset(readers,0xff,max_slots*sizeof(int)));
        int count;
        if(block_cache) { grow_cache();count=cache_ptrs.size(); }
        else count=ggml_dsv4f_allocate_cache(&slab,stride,max_slots,0,"dsv4f-service");
        base_slots=v41?0:count;
        slots.resize(count);
        if(trace)std::fprintf(trace,"S %d %d\n",base_slots,count);
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy,cudaStreamNonBlocking));
        if(lookahead==4) {
            CUDA_CHECK(cudaHostAlloc((void**)&lookahead_claim_h,sizeof(int),cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer((void**)&lookahead_claim_d,lookahead_claim_h,0));
            CUDA_CHECK(cudaStreamCreateWithFlags(&lookahead_copy,cudaStreamNonBlocking));
        }
        if(lookahead)std::fprintf(stderr,"[dsv4f-lookahead] enabled mode=%d predictor=%s depth=%d read=%d temporary_slots=%d format=%s\n",
            lookahead,lookahead>=3?"current-hidden-next-router":"previous-token-next-layer",lookahead>=3?early_depth:1,early_read,temporary_slots,ggml_type_name(gt));
        stop=false;worker=std::thread([this]{serve();});loader=std::thread([this]{load();});
        if(lookahead==4)lookahead_loader=std::thread([this]{load(true);});
        std::fprintf(stderr,"[dsv4f-service] GPU routing + CPU service: %d slots %.2f GiB, wait=%s\n",count,double(count*stride)/(1ULL<<30),poll?"poll":"callback");
        if(profile || notify_wait || callback_spin_us || idle_spins!=200000)std::fprintf(stderr,"[dsv4f-service] host_wait=%s callback_spin_us=%u idle_spins=%u profile=%d format=%s\n",notify_wait?"notify":"legacy",callback_spin_us,idle_spins,int(profile),ggml_type_name(gt));
    }
public:
    ~service(){clear();}
    bool compatible(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d,int nk) {
        std::lock_guard<std::mutex> lock(mutex);
        const char * phase=std::getenv("DSV4F_PHASE_WORKSPACE");
        // The pointer-table epilogue has eight expert warps. Unsupported routing
        // geometries retain the CPU/legacy path before any extension is used.
        if(nk>8 && (!cache_blocks.empty() || (phase && std::atoi(phase)!=0)))return false;
        if(configured)return gt==g->type && ut==u->type && dt==d->type && ne==g->ne[0] && nf==g->ne[1] && gs==g->nb[2] && us==u->nb[2] && ds==d->nb[2];
        selected_source=g->data;configured=true;gt=g->type;ut=u->type;dt=d->type;ne=g->ne[0];nf=g->ne[1];gs=g->nb[2];us=u->nb[2];ds=d->nb[2];return true;
    }
    void statistics(uint64_t * r,uint64_t * h,uint64_t * a){std::lock_guard<std::mutex> lock(mutex);*r=routes;*h=hits;*a=admissions;}
    void grow_cache() {
        // Cap each block at 64 MiB (or one complete expert) so small temporary
        // buffers do not force the entire initial cache to be surrendered.
        const size_t block_slots=std::min<size_t>(64,std::max<size_t>(1,(64ULL<<20)/stride));
        const size_t target=std::min<size_t>(max_slots,format_ceiling/stride);
        while(cache_ptrs.size()<target) {
            char * block=nullptr;
            const size_t existing=cache_ptrs.size();
            const size_t count=ggml_dsv4f_allocate_cache(&block,stride,
                std::min(block_slots,target-existing),0,"dsv4f-service",existing*stride);
            if(!count)break;
            cache_blocks.push_back({block,count});
            for(size_t i=0;i<count;++i)cache_ptrs.push_back(block+i*stride);
        }
        if(!cache_ptrs.empty())CUDA_CHECK(cudaMemcpy(extra,cache_ptrs.data(),cache_ptrs.size()*sizeof(char *),cudaMemcpyHostToDevice));
        slots.resize(cache_ptrs.size());
        reserve_temporary();
    }
    void drain() {
        CUDA_CHECK(cudaDeviceSynchronize());
        for(int i=0;i<RING;++i)while(host[i].consumed!=host[i].seq)std::this_thread::yield();
    }
    // mutex held, service/loader drained; keep is a block boundary.
    void shrink_cache(size_t keep) {
        if(keep>=slots.size())return;
        GGML_ASSERT(policy.protected_residents()<=keep);
        if(block_cache && keep) {
            std::vector<int> base(keep),tail(slots.size()-keep);
            std::iota(base.begin(),base.end(),0);std::iota(tail.begin(),tail.end(),keep);
            // Protected tail entries must move into the retained prefix before
            // freeing blocks. Key-based leases survive that D2D compaction.
            auto score=[&](int i){return std::make_pair(policy.protected_key(slots[i].key),policy.score(slots[i].key));};
            std::sort(base.begin(),base.end(),[&](int a,int b){return score(a)<score(b);});
            std::sort(tail.begin(),tail.end(),[&](int a,int b){return score(a)>score(b);});
            for(size_t i=0;i<std::min(base.size(),tail.size());++i) {
                const int dst=base[i],src=tail[i];
                if(slots[src].key<0 || (slots[dst].key>=0 && score(src)<=score(dst)))break;
                if(policy.protected_key(slots[src].key))++guard_compacted;
                CUDA_CHECK(cudaMemcpyAsync(cache_ptrs[dst],cache_ptrs[src],stride,cudaMemcpyDeviceToDevice,copy));
                slots[dst].key=slots[src].key;
            }
            CUDA_CHECK(cudaStreamSynchronize(copy));
        }
        // Invalidate every removed lookup before returning its storage. Kernels
        // captured with the old capacity may still run, but cannot hit it.
        std::vector<int> lookup(MAX_LAYERS*MAX_EXPERTS,-1),states(slots.size(),-1);
        for(size_t i=0;i<keep;++i)if(slots[i].key>=0){lookup[slots[i].key]=i;states[i]=0;}
        CUDA_CHECK(cudaMemcpy(table,lookup.data(),lookup.size()*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(readers,states.data(),states.size()*sizeof(int),cudaMemcpyHostToDevice));
        if(block_cache) {
            size_t count=cache_ptrs.size();
            while(count>keep) {
                const auto block=cache_blocks.back();
                GGML_ASSERT(count-block.count>=keep);
                ggml_dsv4f_free_cache(block.data,block.count*stride);count-=block.count;cache_blocks.pop_back();
            }
            cache_ptrs.resize(keep);
        } else {
            GGML_ASSERT(keep==0);ggml_dsv4f_free_cache(slab,slots.size()*stride);slab=nullptr;
        }
        base_slots=std::min<size_t>(base_slots,keep);
        slots.resize(keep);policy.rebuild();reserve_temporary();
        if(trace){std::fprintf(trace,"S %d %zu\n",base_slots,slots.size());std::fflush(trace);}
    }
    size_t reclaim(size_t bytes) {
        // MMQ can grow its temporary pool while the custom op owns cache slots.
        // Do not surrender weights before all three projections have retired.
        if(prefill_leased)return 0;
        if(!host || slots.empty() || !bytes)return 0;
        ggml_cuda_set_device(0);drain();
        std::unique_lock<std::mutex> lock(mutex);
        copies.wait(lock,[&]{return pending.empty();});
        CUDA_CHECK(cudaStreamSynchronize(copy));
        const size_t minimum=policy.protected_residents();
        size_t keep=slots.size(),released=0;
        if(block_cache) {
            for(auto it=cache_blocks.rbegin();it!=cache_blocks.rend() && released<bytes;++it) {
                if(keep-it->count<minimum)break;
                keep-=it->count;released+=it->count*stride;
            }
        } else if(!minimum){released=keep*stride;keep=0;}
        if(!released){if(lookahead==5 && minimum)++guard_reclaim_denied;return 0;}
        shrink_cache(keep);
        std::fprintf(stderr,"[dsv4f-service] reclaimed=%.2f MiB remaining_slots=%zu initial_slots=%d\n",released/1048576.,keep,base_slots);
        return released;
    }
    bool resize(bool decode) {
        if(!host)return false;
        ggml_cuda_set_device(0);drain();
        {
            std::unique_lock<std::mutex> lock(mutex);
            copies.wait(lock,[&]{return pending.empty();});
            CUDA_CHECK(cudaStreamSynchronize(copy));
            guard.clear();
            forecast.fill(-1);for(auto & history:last_routes)history.clear();
            early_valid.fill(false);for(auto & ids:early_ids)ids.clear();
            for(auto & ids:early_read_ids)ids.clear();
            if(!decode) {
                // Capture both counters BEFORE freeing cache blocks. HIP sees
                // cudaFree immediately, while AMDGPU's physical counter lags.
                // Sampling after shrink would omit those bytes from the next
                // settle target and stop decode cache growth several GiB early,
                // especially when target and MTP workspaces switch in sequence.
                reclaim_free=ggml_dsv4f_free_vram();
                size_t total;CUDA_CHECK(cudaMemGetInfo(&reclaim_hip_free,&total));
                shrink_cache(base_slots);
            } else if(block_cache) {
#if defined(GGML_USE_HIP) && defined(__linux__)
                size_t hip_free,total;CUDA_CHECK(cudaMemGetInfo(&hip_free,&total));
                constexpr size_t tolerance=64ULL<<20;
                if(reclaim_hip_free && hip_free>reclaim_hip_free+tolerance) {
                    const size_t expected=reclaim_free+(hip_free-reclaim_hip_free)-tolerance;
                    const auto start=std::chrono::steady_clock::now();
                    const auto deadline=start+std::chrono::milliseconds(200);
                    while(ggml_dsv4f_free_vram()<expected && std::chrono::steady_clock::now()<deadline)
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    std::fprintf(stderr,"[dsv4f-service] VRAM accounting: released=%.0f MiB expected_free=%.0f MiB physical_free=%.0f MiB wait_ms=%.2f\n",
                        (hip_free-reclaim_hip_free)/1048576.,expected/1048576.,ggml_dsv4f_free_vram()/1048576.,
                        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
                }
#endif
                grow_cache();
            }
        }
        // Include other contexts' workspaces and driver allocations. A cache
        // smaller than its original prefill size is valid, including zero.
        if(!ggml_dsv4f_reserve_vram(0,0))throw std::runtime_error("DSV4F VRAM headroom cannot be preserved after phase switch");
        std::fprintf(stderr,"[dsv4f-service] phase=%s base_slots=%d extra_slots=%zu total_slots=%zu cache=%.2f MiB free=%.0f MiB\n",
            decode?"decode":"prefill",base_slots,slots.size()-base_slots,slots.size(),slots.size()*stride/1048576.,ggml_dsv4f_free_vram()/1048576.);
        if(trace){std::fprintf(trace,"S %d %zu\n",base_slots,slots.size());std::fflush(trace);}
        return true;
    }
    void clear() {
        if(!host){configured=false;selected_source=nullptr;return;}ggml_cuda_set_device(0);CUDA_CHECK(cudaDeviceSynchronize());
        {std::lock_guard<std::mutex> lock(wake_mutex);stop.store(true,std::memory_order_release);}copies.notify_all();answered.notify_all();ready.notify_all();worker.join();loader.join();if(lookahead_loader.joinable())lookahead_loader.join();
        print_lookahead();
        guard.reset();guard_hits=guard_cancelled=guard_reclaim_denied=guard_compacted=0;
        if(profile)print_callback_profile();
        for(auto block:cache_blocks)ggml_dsv4f_free_cache(block.data,block.count*stride);cache_blocks.clear();cache_ptrs.clear();
        CUDA_CHECK(cudaFree(prefill_updates));prefill_updates=nullptr;
        prefill_changes.clear();prefill_leased=false;
        cudaFree(extra);ggml_dsv4f_free_cache(slab,block_cache?0:slots.size()*stride);cudaFree(seq);cudaFree(issued);cudaFree(table);cudaFree(readers);cudaFree(speculative);speculative=nullptr;cudaFree(gate);cudaFree(down);cudaFree(weights);cudaFree(plan);cudaFree(aq);cudaFree(hq);cudaFreeHost(host);cudaFreeHost(claim_h);cudaStreamDestroy(copy);
        if(lookahead_claim_h){cudaFreeHost(lookahead_claim_h);lookahead_claim_h=lookahead_claim_d=nullptr;}
        if(lookahead_copy){cudaStreamDestroy(lookahead_copy);lookahead_copy=nullptr;}
        if(trace){std::fclose(trace);trace=nullptr;}
        configured=false;selected_source=nullptr;host=device=nullptr;slab=nullptr;extra=nullptr;base_slots=max_slots=0;format_ceiling=SIZE_MAX;layers.clear();layer_ids.clear();policy.clear();pending.clear();queue.clear();lookahead_queue.clear();routes=hits=admissions=admit_bytes=requests=0;cpu_ms=0;
        lookahead=temporary_slots=prefetch_cursor=0;prefetch_pending=false;temporary_busy.fill(false);
        early_valid.fill(false);for(auto & ids:early_ids)ids.clear();
        for(auto & ids:early_read_ids)ids.clear();
        early_read_candidates=early_read_matches=early_read_ready=early_read_miss=0;
        early_armed_batches=early_skipped_batches=early_batches=early_candidates=early_matches=early_actual=early_ready=early_matched_ready=early_actual_miss=early_matched_miss=early_new_candidates=early_cached_candidates=0;forecast.fill(-1);for(auto & history:last_routes)history.clear();
        predictions=forecast_used=forecast_missed=forecast_consumed=prefetch_copies=prefetch_hits=prefetch_timely=prefetch_rejected=prefetch_failed=0;prefetch_copy_us=0;
        completed=0;descriptor_time={};expert_time={};publish_time={};post_time={};sleep_time={};detect_bound={};idle_cpu={};callback_time={};callback_detect={};callback_tail={};callbacks=callbacks_pending=callbacks_unseen=0;
        for(auto & t:timing){t.seen=0;t.done=0;}
    }
    void invalidate(const void * base,size_t bytes) {
        const uintptr_t lo=(uintptr_t)base,hi=lo+bytes;bool found=selected_source && (uintptr_t)selected_source>=lo && (uintptr_t)selected_source<hi;
        {std::lock_guard<std::mutex> lock(mutex);for(const auto & l:layers)for(const void * p:{l.g.data,l.u.data,l.d.data})if((uintptr_t)p>=lo && (uintptr_t)p<hi)found=true;}
        if(found)clear();
    }
    bool prefill_begin(const ggml_tensor * g, const ggml_tensor * u, const ggml_tensor * d,
                       int nk, int group, std::vector<int> & active, ggml_dsv4f_prefill_cache_plan & output) {
        const char * enabled=std::getenv("DSV4F_GPU_SERVICE");
        if(!enabled || !std::atoi(enabled) || nk>8 || g->ne[0]>MAX_EMBD ||
           !compatible(g,u,d,nk))return false;
        init(g,u,d,nk);
        if(!fused_mxfp4 || cache_ptrs.empty())return false;
        // The service's background admission must retire before borrowing its
        // slots. Once leased, host policy and GPU readers both exclude them.
        drain();
        std::unique_lock<std::mutex> lock(mutex);
        copies.wait(lock,[&]{return pending.empty();});
        CUDA_CHECK(cudaStreamSynchronize(copy));
        GGML_ASSERT(!prefill_leased);
        int layer;
        auto found=layer_ids.find(g->data);
        if(found==layer_ids.end()) {
            layer=layers.size();GGML_ASSERT(layer<MAX_LAYERS);
            layers.push_back({*g,*u,*d});layer_ids[g->data]=layer;
        } else layer=found->second;
        std::vector<int> location(g->ne[2],-1);
        for(size_t i=0;i<slots.size();++i) {
            const int key=slots[i].key;
            if(slots[i].ready && key/MAX_EXPERTS==layer && key>=0 && key%MAX_EXPERTS<g->ne[2])
                location[key%MAX_EXPERTS]=i;
        }
        std::vector<int> ordered=active;
        std::stable_partition(ordered.begin(),ordered.end(),[&](int e){return location[e]>=0;});
        int nhits=0;for(int e:ordered)nhits+=location[e]>=0;
        std::vector<bool> pinned(slots.size(),false);
        for(int e:ordered)if(location[e]>=0)pinned[location[e]]=true;
        // Only ring positions actually used by misses need replacement slots.
        std::vector<int> ring(2*group,-1),final_key(2*group,-1),victims;
        for(size_t i=nhits;i<ordered.size();++i)final_key[i%(2*group)]=layer*MAX_EXPERTS+ordered[i];
        int needed=0;for(int key:final_key)needed+=key>=0;
        for(size_t i=0;i<slots.size();++i)if(!pinned[i] && slots[i].ready)victims.push_back(i);
        if(int(victims.size())<needed)return false;
        std::stable_sort(victims.begin(),victims.end(),[&](int a,int b) {
            const bool empty_a=slots[a].key<0,empty_b=slots[b].key<0;
            return empty_a!=empty_b?empty_a:policy.score(slots[a].key)<policy.score(slots[b].key);
        });
        prefill_changes.clear();
        auto lease=[&](int index,int key) {
            prefill_changes.insert(prefill_changes.end(),{index,slots[index].key,key});
            slots[index].ready=false;
        };
        int next=0;
        for(int i=0;i<2*group;++i)if(final_key[i]>=0) {
            ring[i]=victims[next++];lease(ring[i],final_key[i]);
        }
        output={};output.hits=nhits;output.borrowed=needed;
        output.up_offset=uo;output.down_offset=doff;
        for(size_t i=0;i<ordered.size();++i) {
            const int hit=location[ordered[i]];
            if(hit>=0)lease(hit,layer*MAX_EXPERTS+ordered[i]);
            output.resident.push_back(hit>=0);
            output.experts.push_back(cache_ptrs[hit>=0?hit:ring[i%(2*group)]]);
        }
        active=std::move(ordered);
        prefill_leased=true;
        CUDA_CHECK(cudaMemcpy(prefill_updates,prefill_changes.data(),prefill_changes.size()*sizeof(int),cudaMemcpyHostToDevice));
        prefill_lease<<<(prefill_changes.size()/3+127)/128,128,0,copy>>>(readers,table,prefill_updates,prefill_changes.size()/3,false);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(copy));
        return true;
    }
    void prefill_end(cudaStream_t stream) {
        GGML_ASSERT(prefill_leased);
        // Complete the layer before handing its borrowed slots back to decode
        // admission, memory reclamation, graph capture, or another backend.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::lock_guard<std::mutex> lock(mutex);
        for(size_t i=0;i<prefill_changes.size();i+=3)policy.publish(prefill_changes[i],prefill_changes[i+2]);
        prefill_lease<<<(prefill_changes.size()/3+127)/128,128,0,copy>>>(readers,table,prefill_updates,prefill_changes.size()/3,true);
        CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaStreamSynchronize(copy));
        prefill_changes.clear();prefill_leased=false;
    }
    int descriptor_id(const ggml_tensor * g,const ggml_tensor * u,const ggml_tensor * d) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it=layer_ids.find(g->data);if(it!=layer_ids.end())return it->second;
        const int layer=layers.size();GGML_ASSERT(layer<MAX_LAYERS);
        layers.push_back({*g,*u,*d});layer_ids[g->data]=layer;return layer;
    }
    void prefetch(ggml_backend_cuda_context & ctx,ggml_tensor * node) {
        const auto * g=node->src[3],*u=node->src[4],*d=node->src[5];init(g,u,d,1);
        const int layer=descriptor_id(g,u,d);
        prefetch_request<<<1,32,0,ctx.stream()>>>(device,seq,(const int32_t *)node->src[1]->data,
            (const char *)node->src[2]->data,node->src[2]->nb[1],node->src[2]->ne[1],
            layer,node->src[1]->ne[0],node->op_params[12]&1,(float *)node->data);
        CUDA_CHECK(cudaGetLastError());
    }
    void route_one(ggml_backend_cuda_context & ctx,ggml_tensor * dst,int token_index,int32_t * route_plan,float * route_weights,bool single) {
        const auto * g=dst->src[3],*u=dst->src[4],*d=dst->src[5];init(g,u,d,dst->src[1]->ne[0]);
        const int layer=descriptor_id(g,u,d);
        float limit;std::memcpy(&limit,&dst->op_params[8],sizeof(limit));const int nk=dst->src[1]->ne[0];auto stream=ctx.stream();
        route<<<1,256,0,stream>>>(device,seq,table,readers,(const float*)dst->src[0]->data,(const int32_t*)dst->src[1]->data,(const float*)dst->src[2]->data,dst->src[2]->nb[1]/sizeof(float),layer,ne,nk,limit,route_plan,route_weights,issued+token_index,speculative,single);
    }
    void prepare(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
        const int nk=dst->src[1]->ne[0];
        for(int64_t t=0;t<dst->ne[1];++t) {
            ggml_tensor token=*dst,x=*dst->src[0],ids=*dst->src[1],w=*dst->src[2];
            x.data=(char*)x.data+t*x.nb[1];x.ne[1]=1;
            ids.data=(char*)ids.data+t*ids.nb[1];ids.ne[1]=1;
            w.data=(char*)w.data+t*w.nb[2];w.ne[2]=1;
            token.ne[1]=1;token.src[0]=&x;token.src[1]=&ids;token.src[2]=&w;
            auto * plan=(int32_t *)((char *)dst->data+t*dst->nb[1]);
            route_one(ctx,&token,t,plan,(float *)(plan+nk),dst->ne[1]==1);
        }
    }
    void enqueue_one(ggml_backend_cuda_context & ctx,ggml_tensor * dst,int token_index,bool single) {
        const auto * g=dst->src[3],*u=dst->src[4],*d=dst->src[5];init(g,u,d,dst->src[1]->ne[0]);
        const int nk=dst->src[1]->ne[0];float limit;std::memcpy(&limit,&dst->op_params[8],sizeof(limit));
        auto stream=ctx.stream();auto * plan=this->plan;auto * weights=this->weights;
        if(dst->src[6] && dst->src[6]->op_params[9]==GGML_DSV4F_PREPARE_TAG) {
            plan=(int32_t *)((char *)dst->src[6]->data+token_index*dst->src[6]->nb[1]);
            weights=(float *)(plan+nk);
        } else route_one(ctx,dst,token_index,plan,weights,single);
        if(slots.empty() && !fused_mxfp4){
            // No space for a complete expert: keep the GPU service/routing and
            // compute all routed experts on the CPU, without a fatal assertion.
            CUDA_CHECK(cudaMemsetAsync(dst->data,0,ne*sizeof(float),stream));
            return;
        }
        quantize_row_q8_1_cuda((const float*)dst->src[0]->data,nullptr,aq,ut,ne,ne,1,1,ne,1,1,1,stream);
        GGML_ASSERT(gt==ut);
        if(fused_mxfp4 && nk<=8){
            mxfp4_gu<<<dim3(nf/32,nk),256,0,stream>>>(extra,uo,plan,(const block_q8_1*)aq,(block_q8_1*)hq,ne,nf,limit);
            mxfp4_down_sum<<<ne/4,256,0,stream>>>(extra,doff,plan,weights,(const block_q8_1*)hq,nf,ne,nk,(float*)dst->data);
            release_readers<<<1,64,0,stream>>>(readers,plan,nk);CUDA_CHECK(cudaGetLastError());
            return;
        }
        if(v41) {
            dsv4f_ptr_product(gt,extra,uo,plan,aq,gate,ne,nf,nk,false,true,limit,stream);
            quantize_row_q8_1_cuda(gate,nullptr,hq,dt,nf,nf,nk*nf,1,nf,nk,1,1,stream);
            dsv4f_ptr_product(dt,extra,doff,plan,hq,down,nf,ne,nk,true,false,limit,stream);
            release_reduce<<<(ne+255)/256,256,0,stream>>>(down,weights,plan,readers,ne,nk,(float*)dst->data);
            release_readers<<<1,64,0,stream>>>(readers,plan,nk);CUDA_CHECK(cudaGetLastError());return;
        }
        ggml_cuda_moe_cache_mmv(slab+uo,ut,(const char*)aq,plan,gate,ne,nf,slots.size(),stride,nk,1,stream,slab,GGML_GLU_OP_SWIGLU_CLAMP,limit);
        quantize_row_q8_1_cuda(gate,nullptr,hq,dt,nf,nf,nk*nf,1,nf,nk,1,1,stream);
        ggml_cuda_moe_cache_mmv(slab+doff,dt,(const char*)hq,plan,down,nf,ne,slots.size(),stride,nk,nk,stream,nullptr,0);
        release_reduce<<<(ne+255)/256,256,0,stream>>>(down,weights,plan,readers,ne,nk,(float*)dst->data);
        release_readers<<<1,64,0,stream>>>(readers,plan,nk);CUDA_CHECK(cudaGetLastError());
    }
    void enqueue(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
        // GPU expert scratch is reused in stream order, while each token has a
        // separate ring request/answer. Submit all CPU work before the join so
        // both verification tokens can overlap the GPU and shared expert work.
        for (int64_t t=0;t<dst->ne[1];++t) {
            ggml_tensor token=*dst, x=*dst->src[0], ids=*dst->src[1], w=*dst->src[2];
            x.data=(char*)x.data+t*x.nb[1]; x.ne[1]=1;
            ids.data=(char*)ids.data+t*ids.nb[1]; ids.ne[1]=1;
            w.data=(char*)w.data+t*w.nb[2]; w.ne[2]=1;
            token.data=(char*)dst->data+t*dst->nb[1]; token.ne[1]=1;
            token.src[0]=&x; token.src[1]=&ids; token.src[2]=&w;
            enqueue_one(ctx,&token,t,dst->ne[1]==1);
        }
        if(!dst->op_params[10])finish(ctx,dst);
    }
    void finish(ggml_backend_cuda_context & ctx,ggml_tensor * dst) {
        if(!poll)CUDA_CHECK(cudaLaunchHostFunc(ctx.stream(),callback,this));
        for(int64_t t=0;t<dst->ne[1];++t)
            cpu_join<<<1,256,0,ctx.stream()>>>(device,issued+t,(float*)((char*)dst->data+t*dst->nb[1]),ne,poll);
        CUDA_CHECK(cudaGetLastError());
    }
};
std::array<service,4> & all_services(){static std::array<service,4> values;return values;}
service & instance(ggml_type t=GGML_TYPE_MXFP4){return all_services()[t==GGML_TYPE_Q3_K?1:t==GGML_TYPE_Q2_0?2:t==GGML_TYPE_PTQ1_0?3:0];}
service & for_node(ggml_tensor * n){if(n->op_params[9]==GGML_DSV4F_JOIN_TAG)n=n->src[0];return instance(n->src[3]->type);}
thread_local service * prefill_owner=nullptr;

// The original graph casts/repeats the mapping, gathers, clamps/casts IDs,
// creates a validity mask and multiplies probabilities in separate kernels.
static __global__ void route_plan_kernel(const char * mapping,const char * ids,const char * prob,
        char * out,int nk,int nt,size_t map_stride,size_t id_stride,size_t token_stride,
        size_t prob_stride,size_t prob_token_stride) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nk*nt)return;
    const int t=i/nk,k=i%nk;
    const int global=*(const int32_t *)(ids+t*token_stride+k*id_stride);
    const float local=float(*(const int32_t *)(mapping+global*map_stride));
    ((int32_t *)out)[i]=int32_t(fmaxf(local,0.f));
    const float p=*(const float *)(prob+t*prob_token_stride+k*prob_stride);
    ((float *)out)[nk*nt+i]=__fmul_rn(p,local+.5f>0.f?1.f:0.f);
}
}
void ggml_cuda_op_dsv4f_route_plan(ggml_backend_cuda_context & ctx,ggml_tensor * node) {
    const auto * m=node->src[0],*ids=node->src[1],*prob=node->src[2];
    route_plan_kernel<<<(node->ne[0]*node->ne[1]+127)/128,128,0,ctx.stream()>>>(
        (const char *)m->data,(const char *)ids->data,(const char *)prob->data,(char *)node->data,
        node->ne[0],node->ne[1],m->nb[1],ids->nb[0],ids->nb[1],prob->nb[1],prob->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
bool ggml_dsv4f_service_supported(const ggml_tensor * n) {
    if (n->op == GGML_OP_CUSTOM && n->op_params[11] == 1) return ggml_cuda_dsv4f_prefill_supported(n);
    const char * enabled=std::getenv("DSV4F_GPU_SERVICE");
    if(!enabled||atoi(enabled)==0||!ggml_dsv4f_cpu_experts||n->op!=GGML_OP_CUSTOM)return false;
    if(n->op_params[9]==GGML_DSV4F_JOIN_TAG)return true;
    if(n->op_params[9]!=GGML_DSV4F_MOE_TAG && n->op_params[9]!=GGML_DSV4F_PREPARE_TAG && n->op_params[9]!=GGML_DSV4F_PREFETCH_TAG)return false;
    const auto * g=n->src[3],*u=n->src[4],*d=n->src[5];
    return g&&u&&d&&g->type==u->type&&(g->type==GGML_TYPE_MXFP4 || g->type==GGML_TYPE_Q4_K || g->type==GGML_TYPE_Q6_K || g->type==GGML_TYPE_Q3_K || g->type==GGML_TYPE_Q2_0 || g->type==GGML_TYPE_PTQ1_0)&&(d->type==g->type || d->type==GGML_TYPE_MXFP4 || d->type==GGML_TYPE_Q4_K || d->type==GGML_TYPE_Q6_K)&&!g->extra&&!u->extra&&!d->extra&&
        g->ne[0]<=MAX_EMBD && n->src[0]->ne[1]>=1 && n->src[0]->ne[1]<=4 && n->src[1]->ne[0]<MAX_USED && g->ne[2]<=MAX_EXPERTS &&
        g->buffer&&u->buffer&&d->buffer&&ggml_backend_buffer_is_host(g->buffer)&&ggml_backend_buffer_is_host(u->buffer)&&ggml_backend_buffer_is_host(d->buffer) && instance(g->type).compatible(g,u,d,n->op_params[9]==GGML_DSV4F_PREFETCH_TAG?1:n->src[1]->ne[0]);
}
void ggml_dsv4f_service_compute(void * ctx,ggml_tensor * node){
    auto & cuda_ctx = *static_cast<ggml_backend_cuda_context*>(ctx);
    if (node->op_params[11] == 1) ggml_cuda_dsv4f_prefill(cuda_ctx, node);
    else for_node(node).enqueue(cuda_ctx,node);
}
void ggml_dsv4f_service_prefetch(void * ctx,ggml_tensor * node){instance(node->src[3]->type).prefetch(*static_cast<ggml_backend_cuda_context*>(ctx),node);}
void ggml_dsv4f_service_prepare(void * ctx,ggml_tensor * node){for_node(node).prepare(*static_cast<ggml_backend_cuda_context*>(ctx),node);}
void ggml_dsv4f_service_finalize(void * ctx,ggml_tensor * node){for_node(node).finish(*static_cast<ggml_backend_cuda_context*>(ctx),node);}
void ggml_dsv4f_service_invalidate(const void * base,size_t bytes){for(auto & s:all_services())s.invalidate(base,bytes);}

void ggml_dsv4f_service_statistics(uint64_t * r,uint64_t * h,uint64_t * a){*r=*h=*a=0;for(auto & s:all_services()){uint64_t rr,hh,aa;s.statistics(&rr,&hh,&aa);*r+=rr;*h+=hh;*a+=aa;}}
bool ggml_dsv4f_service_resize(bool decode){bool changed=false;for(auto & s:all_services())changed=s.resize(decode)||changed;return changed;}
bool ggml_dsv4f_prefill_cache_begin(const ggml_tensor * g, const ggml_tensor * u,
    const ggml_tensor * d, int nk, int group, std::vector<int> & active,
    ggml_dsv4f_prefill_cache_plan & plan) {auto & s=instance(g->type);bool ok=s.prefill_begin(g,u,d,nk,group,active,plan);if(ok)prefill_owner=&s;return ok;}
void ggml_dsv4f_prefill_cache_end(cudaStream_t stream) {GGML_ASSERT(prefill_owner);prefill_owner->prefill_end(stream);prefill_owner=nullptr;}


bool ggml_dsv4f_reserve_vram(int device,size_t bytes,bool allow_reclaim) {
    const char * enabled=std::getenv("DSV4F_GPU_SERVICE");
    if(!enabled || !std::atoi(enabled))return true;
    ggml_cuda_set_device(device);
    const size_t configured=ggml_dsv4f_headroom();
    const size_t library_guard=ggml_dsv4f_guard();
    // Temporary allocations must not consume the allowance for lazy library
    // code/workspaces. Replenish it at every allocation and graph boundary,
    // including when target and MTP contexts alternate their workspaces.
    const size_t headroom=configured>SIZE_MAX-library_guard?SIZE_MAX:configured+library_guard;
    size_t free=ggml_dsv4f_free_vram();
    size_t deficit=ggml_dsv4f_vram_deficit(free,headroom,bytes);
    if(!deficit)return true;
    // Service cache lives on logical device zero. Never reclaim from another
    // device or synchronize/free while a stream is being captured.
    if(device==0 && allow_reclaim) {
#if defined(GGML_USE_HIP) && defined(__linux__)
        // Cache/workspace frees are visible to HIP before AMDGPU sysfs. Give
        // the physical counter time to settle before evicting useful experts.
        // A real desktop allocation remains counted after the bounded wait.
        size_t hip_free,total;CUDA_CHECK(cudaMemGetInfo(&hip_free,&total));
        if(!ggml_dsv4f_vram_deficit(hip_free,headroom,bytes)) {
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
            do {
                free=ggml_dsv4f_free_vram();deficit=ggml_dsv4f_vram_deficit(free,headroom,bytes);
                if(!deficit)return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while(std::chrono::steady_clock::now()<deadline);
        }
#endif
        // Avoid repeated tiny trims as driver accounting settles. The larger
        // library allowance is already included in headroom above.
        const size_t guard=64ULL<<20;
        const size_t request=deficit>SIZE_MAX-guard?SIZE_MAX:deficit+guard;
        size_t released=0;for(auto & s:all_services()){if(released>=request)break;released+=s.reclaim(request-released);}
        if(released) {
            // AMDGPU sysfs may lag frees. Remeasure physical VRAM rather than
            // assuming all released bytes have already become available.
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(200);
            do {
                free=ggml_dsv4f_free_vram();deficit=ggml_dsv4f_vram_deficit(free,headroom,bytes);
                if(!deficit)return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            } while(std::chrono::steady_clock::now()<deadline);
        }
    }
    std::fprintf(stderr,"[dsv4f-service] insufficient VRAM: allocation=%.2f MiB free=%.2f MiB headroom=%.2f MiB guard=%.2f MiB reclaim=%s\n",
        bytes/1048576.,free/1048576.,configured/1048576.,library_guard/1048576.,allow_reclaim?"allowed":"capture");
    return false;
}


void ggml_cuda_op_dsv4f_engram(ggml_backend_cuda_context & ctx,ggml_tensor * node) {
    ggml_custom_op_params params;std::memcpy(&params,node->op_params,sizeof(params));
    auto * transfer=(ggml_dsv4f_engram_transfer *)params.userdata;
    CUDA_CHECK(cudaLaunchHostFunc(ctx.stream(),transfer->wait,transfer->userdata));
    // Source is pinned and stable throughout graph replay. The stream callback
    // only waits for the reader; runtime APIs are issued here, outside callback.
    CUDA_CHECK(cudaMemcpyAsync(node->data,transfer->data,ggml_nbytes(node),cudaMemcpyHostToDevice,ctx.stream()));
}

// One warp per token. Selection keeps bias out of the returned probabilities,
// and ties choose the smaller expert ID, matching the generic argsort path.
static __global__ void router_kernel(const char * logits,const float * bias,char * output,
                                     int ne,int nk,int nt,size_t stride,bool normalize,float scale) {
    const int lane=threadIdx.x,t=blockIdx.x;
    float scores[16],rank[16];int selected[16];
    for(int j=0;j<16;++j) {
        const int i=lane+j*WARP_SIZE;
        float score=-INFINITY;
        if(i<ne) {const float x=*(const float *)(logits+t*stride+i*4);score=sqrtf(x>20.f?x:logf(1.f+expf(x)));}
        scores[j]=score;rank[j]=i<ne?score+bias[i]:-INFINITY;selected[j]=0;
        // Keep IDs valid even if an upstream operation produces a NaN.
        if(__isnanf(rank[j]))rank[j]=-FLT_MAX;
    }
    float weight=0.f;
    for(int k=0;k<nk;++k) {
        float best=-INFINITY;int id=INT_MAX;
        for(int j=0;j<16;++j) {
            const int i=lane+j*WARP_SIZE;
            if(i<ne && !selected[j] && (id==INT_MAX || rank[j]>best || (rank[j]==best && i<id))) {best=rank[j];id=i;}
        }
        for(int offset=WARP_SIZE/2;offset;offset/=2) {
            const float other=__shfl_xor_sync(0xffffffff,best,offset,WARP_SIZE);
            const int oi=__shfl_xor_sync(0xffffffff,id,offset,WARP_SIZE);
            if(other>best || (other==best && oi<id)){best=other;id=oi;}
        }
        const float score=__shfl_sync(0xffffffff,scores[id/WARP_SIZE],id%WARP_SIZE,WARP_SIZE);
        if(lane==id%WARP_SIZE)selected[id/WARP_SIZE]=1;
        if(lane==k){((int32_t *)output)[t*nk+k]=id;weight=score;}
    }
    // Match SUM_ROWS' warp reduction, followed by SCALE_BIAS, DIV and SCALE.
    const float sum=warp_reduce_sum(weight);
    if(lane<nk)((float *)output)[nk*nt+t*nk+lane]=scale*(normalize?weight/(sum+1e-20f):weight)+0.f;
}
void ggml_cuda_op_dsv4f_router(ggml_backend_cuda_context & ctx,ggml_tensor * node) {
    float scale;std::memcpy(&scale,&node->op_params[12],4);
    router_kernel<<<node->ne[1],WARP_SIZE,0,ctx.stream()>>>((const char *)node->src[0]->data,
        (const float *)node->src[1]->data,(char *)node->data,node->src[0]->ne[0],node->ne[0],
        node->ne[1],node->src[0]->nb[1],node->op_params[10],scale);
    CUDA_CHECK(cudaGetLastError());
}
