#include "moe.h"
#include "ggml-backend-moe-cache.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>

static void prefetch_cpu(ggml_tensor * dst,int ith,int,void *) {
    if(!ith)std::memset(dst->data,0,ggml_nbytes(dst));
}
ggml_tensor * dsv4f_expert_prefetch(ggml_context * ctx,ggml_tensor * anchor,
    ggml_tensor * ids,ggml_tensor * mapping,ggml_tensor * gate,
    ggml_tensor * up,ggml_tensor * down,int target_layer) {
    GGML_ASSERT(anchor->ne[1]==1 && ids->type==GGML_TYPE_I32 && ids->ne[1]==1 &&
        ids->ne[0]>0 && ids->ne[0]<=16 && mapping->type==GGML_TYPE_I32 && mapping->ne[0]==1);
    ggml_tensor * args[]={anchor,ids,mapping,gate,up,down};
    auto * node=ggml_custom_4d(ctx,GGML_TYPE_F32,1,1,1,1,args,6,prefetch_cpu,1,nullptr);
    node->op_params[9]=GGML_DSV4F_PREFETCH_TAG;node->op_params[12]=target_layer;
    ggml_set_name(node,"v41_expert_lookahead");return node;
}

static void route_plan_cpu(ggml_tensor * dst,int ith,int,void *) {
    if(ith)return;
    const auto * mapping=dst->src[0],*ids=dst->src[1],*prob=dst->src[2];
    const int nk=dst->ne[0],nt=dst->ne[1];
    auto * output=(char *)dst->data;
    for(int t=0;t<nt;++t)for(int k=0;k<nk;++k) {
        const int global=*(const int32_t *)((const char *)ids->data+t*ids->nb[1]+k*ids->nb[0]);
        GGML_ASSERT(global>=0 && global<mapping->ne[1]);
        const float local=float(*(const int32_t *)((const char *)mapping->data+global*mapping->nb[1]));
        const int32_t lid=int32_t(std::max(local,0.f));
        const float p=*(const float *)((const char *)prob->data+t*prob->nb[2]+k*prob->nb[1]);
        const float weight=p*(local+.5f>0.f?1.f:0.f);
        const size_t index=size_t(t)*nk+k;
        std::memcpy(output+index*4,&lid,4);
        std::memcpy(output+(size_t(nk)*nt+index)*4,&weight,4);
    }
}

void dsv4f_route_plan(ggml_context * ctx,ggml_tensor * mapping,ggml_tensor * ids,
                     ggml_tensor * probabilities,ggml_tensor ** local_ids,
                     ggml_tensor ** local_probabilities) {
    GGML_ASSERT(mapping->type==GGML_TYPE_I32 && mapping->ne[0]==1 &&
                mapping->ne[2]==1 && mapping->ne[3]==1);
    GGML_ASSERT(ids->type==GGML_TYPE_I32 && ids->ne[0]>0 && ids->ne[0]<=64 &&
                ids->ne[1]>0 && ids->ne[1]<=4 && ids->ne[2]==1 && ids->ne[3]==1);
    GGML_ASSERT(probabilities->type==GGML_TYPE_F32 && probabilities->ne[0]==1 &&
                probabilities->ne[1]==ids->ne[0] && probabilities->ne[2]==ids->ne[1] &&
                probabilities->ne[3]==1);
    ggml_tensor * inputs[]={mapping,ids,probabilities};
    const int nk=ids->ne[0],nt=ids->ne[1];
    auto * packed=ggml_custom_4d(ctx,GGML_TYPE_F32,nk,nt,2,1,inputs,3,route_plan_cpu,1,nullptr);
    packed->op_params[9]=GGML_DSV4F_ROUTE_TAG;
    ggml_set_name(packed,"v41_route_plan");
    *local_ids=ggml_view_2d(ctx,packed,nk,nt,nk*4,0);
    // Explicit bit reinterpretation, not numeric float-to-int conversion.
    // The kernel writes the first plane as I32 and the second as F32.
    (*local_ids)->type=GGML_TYPE_I32;
    *local_probabilities=ggml_view_3d(ctx,packed,1,nk,nt,4,nk*4,size_t(nk)*nt*4);
}

// Keep the generic graph for prefill and other backends. This packed operation
// exposes both routing outputs while retaining an independent CPU fallback.
static void router_cpu(ggml_tensor * dst,int ith,int,void *) {
    if(ith)return;
    const auto * logits=dst->src[0],*bias=dst->src[1];
    const int ne=logits->ne[0],nk=dst->ne[0],nt=dst->ne[1];
    float scale;std::memcpy(&scale,&dst->op_params[12],4);
    std::vector<float> scores(ne),rank(ne);std::vector<int32_t> order(ne);
    for(int t=0;t<nt;++t) {
        for(int i=0;i<ne;++i) {
            const float x=*(const float *)((const char *)logits->data+t*logits->nb[1]+i*4);
            scores[i]=std::sqrt(x>20.f?x:std::log(1.f+std::exp(x)));
            rank[i]=scores[i]+((const float *)bias->data)[i];
            if(std::isnan(rank[i]))rank[i]=-std::numeric_limits<float>::max();
            order[i]=i;
        }
        std::partial_sort(order.begin(),order.begin()+nk,order.end(),[&](int a,int b){return rank[a]>rank[b] || (rank[a]==rank[b] && a<b);});
        double sum=0;for(int k=0;k<nk;++k)sum+=scores[order[k]];
        const float denominator=float(sum)+1e-20f;
        for(int k=0;k<nk;++k) {
            const size_t index=size_t(t)*nk+k;
            std::memcpy((char *)dst->data+index*4,&order[k],4);
            const float weight=scale*(dst->op_params[10]?scores[order[k]]/denominator:scores[order[k]])+0.f;
            std::memcpy((char *)dst->data+(size_t(nk)*nt+index)*4,&weight,4);
        }
    }
}
void dsv4f_router(ggml_context * ctx,ggml_tensor * logits,ggml_tensor * bias,
                 int nk,bool normalize,float scale,ggml_tensor ** ids,ggml_tensor ** probabilities) {
    GGML_ASSERT(logits->type==GGML_TYPE_F32 && logits->nb[0]==4 &&
                logits->ne[0]>0 && logits->ne[0]<=512 && logits->ne[1]>0 && logits->ne[1]<=4 &&
                logits->ne[2]==1 && logits->ne[3]==1);
    GGML_ASSERT(bias->type==GGML_TYPE_F32 && ggml_is_contiguous(bias) && ggml_nelements(bias)==logits->ne[0]);
    GGML_ASSERT(nk>0 && nk<=32 && nk<=logits->ne[0]);
    ggml_tensor * inputs[]={logits,bias};const int nt=logits->ne[1];
    auto * packed=ggml_custom_4d(ctx,GGML_TYPE_F32,nk,nt,2,1,inputs,2,router_cpu,1,nullptr);
    packed->op_params[9]=GGML_DSV4F_ROUTER_TAG;packed->op_params[10]=normalize;
    std::memcpy(&packed->op_params[12],&scale,4);ggml_set_name(packed,"v41_router");
    *ids=ggml_view_2d(ctx,packed,nk,nt,nk*4,0);(*ids)->type=GGML_TYPE_I32;
    *probabilities=ggml_view_3d(ctx,packed,1,nk,nt,4,nk*4,size_t(nk)*nt*4);
}
