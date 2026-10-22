#include "engram.h"
#include "ggml-backend-moe-cache.h"
#include "ggml-dsv4f-timeline.h"
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

struct dsv4f_engram_async::state {
    struct slot {
        state * owner;
        int index;
        ggml_backend_buffer_t buffer=nullptr;
        ggml_dsv4f_engram_transfer transfer{};
        bool ready=false;
        double read_us=0;
        std::exception_ptr error;
        ~slot(){if(buffer)ggml_backend_buffer_free(buffer);}
    };
    const dsv4f_engram & reader;
    const int nt,width;
    const bool profile=[] {const char * v=std::getenv("DSV4F_ENGRAM_ASYNC_PROFILE");return v && !std::strcmp(v,"1");}();
    static double now_us(){return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    std::vector<std::unique_ptr<slot>> slots;
    std::mutex mutex;
    std::condition_variable cv;
    bool stopping=false,busy=false;
    std::vector<int32_t> history;
    std::shared_ptr<dsv4f_engram_prefetch::job> pending;
    std::fenv_t environment;
    std::thread worker;

    state(const dsv4f_engram & r,int n,int w,int count):reader(r),nt(n),width(w),slots(count),worker([this]{run();}){}
    ~state(){
        {std::unique_lock<std::mutex> lock(mutex);stopping=true;cv.notify_all();}
        worker.join();
    }
    void run(){
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            cv.wait(lock,[&]{return busy||stopping;});
            if(!busy && stopping)return;
            std::fesetenv(&environment);
            lock.unlock();
            // One worker preserves table ordering and outstanding I/O depth.
            // Each completed table can be consumed without waiting for later ones.
            for(auto & p:slots)if(p){
                std::exception_ptr error;
                const double started=profile?now_us():0;
                try {
                    dsv4f_timeline::scope span(p->index==0?"v41 engram/read slot0":"v41 engram/read later");
                    std::vector<float> rows;
                    if(pending)dsv4f_engram_prefetch::take(pending,p->index,rows);
                    else reader.read(p->index,history,nt,rows);
                    if(rows.size()!=size_t(nt)*width)throw std::runtime_error("Engram async row size mismatch");
                    std::memcpy(ggml_backend_buffer_get_base(p->buffer),rows.data(),rows.size()*sizeof(float));
                } catch(...){error=std::current_exception();}
                const double read_us=profile?now_us()-started:0;
                lock.lock();p->read_us=read_us;p->error=error;p->ready=true;cv.notify_all();lock.unlock();
            }
            lock.lock();pending.reset();busy=false;cv.notify_all();
        }
    }
    static void wait(void * opaque){
        auto & s=*(slot *)opaque;auto & self=*s.owner;
        dsv4f_timeline::scope span(s.index==0?"v41 engram/wait slot0":"v41 engram/wait later");
        const double started=self.profile?now_us():0;
        std::unique_lock<std::mutex> lock(self.mutex);
        self.cv.wait(lock,[&]{return s.ready;});
        if(s.error)std::rethrow_exception(s.error);
        if(self.profile)std::fprintf(stderr,"[dsv4f-engram-async] slot=%d tokens=%d bytes=%zu read_us=%.3f wait_us=%.3f\n",
            s.index,self.nt,size_t(self.nt)*self.width*sizeof(float),s.read_us,now_us()-started);
    }
    static void wait_callback(void * opaque){
        // Runtime stream callbacks cannot propagate C++ exceptions or call HIP.
        try{wait(opaque);}catch(const std::exception & e){
            std::fprintf(stderr,"[dsv4f-engram] asynchronous read failed: %s\n",e.what());std::abort();
        }catch(...){std::fprintf(stderr,"[dsv4f-engram] asynchronous read failed\n");std::abort();}
    }
    static void compute_cpu(ggml_tensor * dst,int ith,int,void * opaque){
        if(ith)return;
        auto & transfer=*(ggml_dsv4f_engram_transfer *)opaque;
        wait(transfer.userdata);std::memcpy(dst->data,transfer.data,ggml_nbytes(dst));
    }
};

dsv4f_engram_async::dsv4f_engram_async(const dsv4f_engram & r,int nt,int width,int count):impl(new state(r,nt,width,count)){}
dsv4f_engram_async::~dsv4f_engram_async()=default;
bool dsv4f_engram_async::enabled(int index) const{return bool(impl->slots.at(index));}
ggml_tensor * dsv4f_engram_async::tensor(ggml_context * ctx,ggml_tensor * anchor,int index,ggml_backend_dev_t device){
    auto & s=*impl;auto & p=s.slots.at(index);GGML_ASSERT(!p);
    p=std::make_unique<state::slot>();p->owner=&s;p->index=index;
    auto buft=ggml_backend_dev_host_buffer_type(device);
    if(!buft && ggml_backend_dev_type(device)==GGML_BACKEND_DEVICE_TYPE_CPU)buft=ggml_backend_dev_buffer_type(device);
    if(!buft)throw std::runtime_error("Engram overlap requires pinned host buffers");
    p->buffer=ggml_backend_buft_alloc_buffer(buft,size_t(s.nt)*s.width*sizeof(float));
    if(!p->buffer)throw std::runtime_error("Engram overlap host allocation failed");
    p->transfer={state::wait_callback,p.get(),ggml_backend_buffer_get_base(p->buffer),device};
    // Anchor consumption to this layer's incoming hidden state. This avoids
    // reading early just because the Engram projection is graph-independent.
    ggml_tensor * args[]={anchor};
    auto * dst=ggml_custom_4d(ctx,GGML_TYPE_F32,s.width,s.nt,1,1,args,1,state::compute_cpu,1,&p->transfer);
    dst->op_params[9]=GGML_DSV4F_ENGRAM_TAG;ggml_set_name(dst,"v41_engram_async_rows");return dst;
}
void dsv4f_engram_async::start(const std::vector<int32_t> & history,std::shared_ptr<dsv4f_engram_prefetch::job> pending){
    auto & s=*impl;std::unique_lock<std::mutex> lock(s.mutex);
    s.cv.wait(lock,[&]{return !s.busy;});
    s.history=history;s.pending=std::move(pending);std::fegetenv(&s.environment);
    for(auto & p:s.slots)if(p){p->ready=false;p->error=nullptr;}
    s.busy=true;s.cv.notify_all();
}
