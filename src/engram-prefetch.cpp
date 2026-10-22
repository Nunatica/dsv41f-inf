#include "engram.h"
#include <algorithm>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace {
double now_us(){return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();}
unsigned fp_control(){
#ifdef __SSE__
    return _mm_getcsr() & ~0x3fu; // Ignore accrued exception status, retain rounding/FTZ/DAZ.
#else
    return 0;
#endif
}
}
struct dsv4f_engram_prefetch::job {
    int nt,pos,seq,round=std::fegetround();
    unsigned control=fp_control();
    std::fenv_t environment{};
    std::vector<int32_t> history;
    std::atomic<bool> cancel{false};
    struct table {bool ready=false;std::vector<float> rows;std::exception_ptr error;};
    std::vector<table> tables;
    std::mutex mutex;
    std::condition_variable cv;
    job(std::vector<int32_t> h,int n,int p,int s,int count):nt(n),pos(p),seq(s),history(std::move(h)),tables(count){std::fegetenv(&environment);}
    bool matches(const std::vector<int32_t> & h,int n,int p,int s)const{
        return !cancel && nt==n && pos==p && seq==s && history==h && round==std::fegetround() && control==fp_control();
    }
};
struct dsv4f_engram_prefetch::state {
    const dsv4f_engram & reader;
    const int ngram,tables;
    const bool profile=[] {const char * v=std::getenv("DSV4F_ENGRAM_ASYNC_PROFILE");return v && std::atoi(v)==1;}();
    std::mutex mutex;
    std::condition_variable cv;
    bool stopping=false;
    std::vector<int32_t> tokens;
    int seq=0,batch=0;
    std::deque<std::shared_ptr<job>> queue;
    std::shared_ptr<job> active,future;
    std::thread worker;
    state(const dsv4f_engram & r,int n,int t):reader(r),ngram(n),tables(t),worker([this]{run();}){}
    void cancel_locked(){
        if(active)active->cancel=true;
        for(auto & j:queue){
            j->cancel=true;
            std::lock_guard<std::mutex> guard(j->mutex);
            for(auto & t:j->tables){t.error=std::make_exception_ptr(std::runtime_error("Engram prefetch cancelled"));t.ready=true;}
            j->cv.notify_all();
        }
        queue.clear();future.reset();
    }
    ~state(){
        {std::lock_guard<std::mutex> lock(mutex);cancel_locked();stopping=true;cv.notify_all();}
        worker.join();
    }
    void run(){
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            cv.wait(lock,[&]{return stopping || !queue.empty();});
            if(stopping)return;
            auto j=queue.front();queue.pop_front();active=j;lock.unlock();
            std::fesetenv(&j->environment);
            for(int s=0;s<tables;++s){
                std::vector<float> rows;std::exception_ptr error;
                const double started=profile?now_us():0;
                try{if(j->cancel)throw std::runtime_error("Engram prefetch cancelled");reader.read(s,j->history,j->nt,rows,nullptr,&j->cancel);}
                catch(...){error=std::current_exception();}
                const double read_us=profile?now_us()-started:0;
                {std::lock_guard<std::mutex> guard(j->mutex);auto & t=j->tables[s];t.rows=std::move(rows);t.error=error;t.ready=true;j->cv.notify_all();}
                if(profile)std::fprintf(stderr,"[dsv4f-engram-prefetch-read] pos=%d tokens=%d slot=%d read_us=%.3f cancelled=%d\n",j->pos,j->nt,s,read_us,int(j->cancel.load()));
            }
            lock.lock();active.reset();cv.notify_all();
        }
    }
};
dsv4f_engram_prefetch::dsv4f_engram_prefetch(const dsv4f_engram & r,int n,int t):impl(new state(r,n,t)){}
dsv4f_engram_prefetch::~dsv4f_engram_prefetch()=default;
void dsv4f_engram_prefetch::clear(){
    auto & s=*impl;std::lock_guard<std::mutex> lock(s.mutex);s.cancel_locked();s.tokens.clear();
}
void dsv4f_engram_prefetch::hint(const int32_t * tokens,int count,int seq,int batch){
    if(!tokens || count<=0 || batch<=4){clear();return;}
    auto & s=*impl;std::lock_guard<std::mutex> lock(s.mutex);
    if(s.seq==seq && s.batch==batch && s.tokens.size()==size_t(count) && std::equal(s.tokens.begin(),s.tokens.end(),tokens))return;
    s.cancel_locked();s.tokens.assign(tokens,tokens+count);s.seq=seq;s.batch=batch;
}
std::shared_ptr<dsv4f_engram_prefetch::job> dsv4f_engram_prefetch::begin(const std::vector<int32_t> & history,int nt,int pos,int seq){
    auto & s=*impl;std::lock_guard<std::mutex> lock(s.mutex);
    // No prompt hint means the existing same-batch reader remains in charge.
    if(s.tokens.empty() || nt<=4)return nullptr;
    const bool hit=s.future && s.future->matches(history,nt,pos,seq);
    std::shared_ptr<job> current;
    if(hit){current=std::move(s.future);}
    else{
        s.cancel_locked();
        current=std::make_shared<job>(history,nt,pos,seq,s.tables);s.queue.push_back(current);
    }
    // Verify the hint's preceding n-gram too; cached/edited prefixes must never
    // seed future history. Actual KV-derived history is authoritative.
    bool known=seq==s.seq && pos>=0 && pos<=int(s.tokens.size())-nt && history.size()==size_t(nt)*s.ngram;
    for(int t=0;known && t<nt;++t)for(int n=0;n<s.ngram;++n){
        const int p=pos+t-n;
        if(history[size_t(t)*s.ngram+n]!=(p<0?-1:s.tokens[p])){known=false;break;}
    }
    const int next=pos+nt;
    const int take=known?std::min(s.batch,int(s.tokens.size())-next):0;
    if(take>4){
        std::vector<int32_t> h(size_t(take)*s.ngram);
        for(int t=0;t<take;++t)for(int n=0;n<s.ngram;++n){const int p=next+t-n;h[size_t(t)*s.ngram+n]=p<0?-1:s.tokens[p];}
        s.future=std::make_shared<job>(std::move(h),take,next,seq,s.tables);s.queue.push_back(s.future);
    }
    if(s.profile)std::fprintf(stderr,"[dsv4f-engram-prefetch] pos=%d tokens=%d hit=%d next_tokens=%d\n",pos,nt,int(hit),take>4?take:0);
    s.cv.notify_all();return current;
}
void dsv4f_engram_prefetch::take(const std::shared_ptr<job> & j,int slot,std::vector<float> & rows){
    auto & t=j->tables.at(slot);std::unique_lock<std::mutex> lock(j->mutex);
    j->cv.wait(lock,[&]{return t.ready;});
    if(t.error)std::rethrow_exception(t.error);
    rows=std::move(t.rows);
}
