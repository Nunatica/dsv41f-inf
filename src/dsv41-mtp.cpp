#include "direct-weights.h"
#include "llama-mmap.h"
// Native DeepSeek V4.1 DSpark. Graph arithmetic adapted from Xolotl/Strata's
// tools/v41/engine/ds41_dspark.cpp (MIT):
// Copyright (c) 2026 Niko1221 and the Strata contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "llama-ext.h"
#include "llama-context.h"
#include "models/deepseek41.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

struct llama_dsv41_mtp {
    const llama_model_deepseek41 & model;
    ggml_backend_ptr backend;
    ggml_context_ptr weights_ctx, state_ctx;
    ggml_backend_buffer_ptr weights_buf, state_buf;
    std::map<std::string,ggml_tensor *> weights;
    int D,HC,DH,NH,OG,OL,OGD,NE,NU,V,BS,WIN,MP;
    ggml_tensor * i_main,*i_mpos,*i_mslot,*i_emb,*i_bpos,*i_mask,*i_bonus,*i_pre;
    std::vector<ggml_tensor *> ring;
    std::vector<int32_t> filled;
    ggml_gallocr_ptr alloc;
    struct graph {
        ggml_context_ptr ctx;
        ggml_cgraph * gf=nullptr;
        std::vector<ggml_tensor *> ids,conf;
    };
    std::map<std::pair<int,bool>,graph> graphs;
    std::pair<int,bool> allocated_key{-1,false};
    explicit llama_dsv41_mtp(llama_context * target,const char * path,const char * manifest);
    ggml_tensor * W(const std::string & n) const {return weights.at(n);}
    ggml_tensor * BW(int b,const std::string & n) const {return W("mtp."+std::to_string(b)+"."+n);}
    graph & build(int k,bool draft);
    bool compute(int k,bool draft);
};

bool llama_dsv41_mtp_supported(const llama_model * base) {
    const auto * m=dynamic_cast<const llama_model_deepseek41 *>(base);
    return m && m->dspark_block>0 && m->dspark_layers.size()==3;
}

llama_dsv41_mtp::llama_dsv41_mtp(llama_context * target,const char * path,const char * manifest):
    model(dynamic_cast<const llama_model_deepseek41 &>(*llama_get_model(target))) {
    const auto & m=model;
    D=m.hparams.n_embd;HC=m.hc;DH=m.hparams.n_embd_head_k();NH=m.hparams.n_head();
    OG=m.og;OL=m.ol;OGD=NH*DH/OG;NE=m.dspark_experts;NU=m.dspark_used;
    V=m.vocab.n_tokens();BS=m.dspark_block;WIN=m.hparams.n_swa;MP=GGML_PAD(WIN+BS,256);
    if(!path || !*path || !manifest || !*manifest || BS<1 || BS>32 || WIN<1 || WIN>4096 || NE<1 || NU<1 || NU>NE || m.dspark_rank<1 || m.dspark_noise<0 || m.dspark_noise>=V)
        throw std::runtime_error("invalid DSpark configuration or MTP_MODEL path");
    for(int layer:m.dspark_layers)if(layer<0 || layer>=(int)m.hparams.n_layer())throw std::runtime_error("DSpark target layer out of range");
    if(!m.output || !m.output->buffer || !m.tok_embd || !m.tok_embd->buffer)throw std::runtime_error("DSpark requires allocated target weights");
    auto * dev=ggml_backend_buft_get_device(ggml_backend_buffer_get_type(m.output->buffer));
    if(!dev)dev=ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    backend.reset(ggml_backend_dev_init(dev,nullptr));
    if(!backend)throw std::runtime_error("cannot initialize DSpark backend");
    if(ggml_backend_is_cpu(backend.get()))ggml_backend_cpu_set_n_threads(backend.get(),llama_n_threads(target));
    auto buft=ggml_backend_get_default_buffer_type(backend.get());
    std::ifstream index_file(manifest);
    if(!index_file)throw std::runtime_error("cannot read DSpark JSON manifest");
    nlohmann::json index;index_file>>index;
    struct source {
        ggml_type type;uint64_t offset,bytes;std::vector<int64_t> dims;
    };
    std::map<std::string,source> sources;
    llama_file stream(path,"rb",dsv4f_direct_weights_enabled());
    const auto file_bytes=stream.size();
    if(dsv4f_direct_weights_enabled() && !stream.has_direct_io())throw std::runtime_error("DSpark binary requires O_DIRECT under a RAM budget");
    for(const auto & item:index.at("tensors")) {
        const auto name=item.at(0).get<std::string>();
        if(name.rfind("mtp.",0)!=0 || name.find(".bias_vl")!=std::string::npos)continue;
        const auto type_name=item.at(1).get<std::string>();
        const std::map<std::string,ggml_type> types={{"F32",GGML_TYPE_F32},{"BF16",GGML_TYPE_BF16},
            {"Q8_0",GGML_TYPE_Q8_0},{"IQ2_XXS",GGML_TYPE_IQ2_XXS}};
        auto ty=types.find(type_name);if(ty==types.end())throw std::runtime_error("unsupported DSpark type: "+type_name);
        source src{ty->second,item.at(2).get<uint64_t>(),item.at(3).get<uint64_t>(),item.at(4).get<std::vector<int64_t>>()};
        // JSON uses PyTorch dimension order, opposite ggml's contiguous order.
        std::reverse(src.dims.begin(),src.dims.end());
        if(src.dims.empty() || src.dims.size()>4)throw std::runtime_error("invalid DSpark rank: "+name);
        uint64_t rows=1;
        for(size_t i=0;i<src.dims.size();++i) {
            if(src.dims[i]<1 || src.dims[i]>INT32_MAX)throw std::runtime_error("invalid DSpark dimensions: "+name);
            if(i){if(rows>UINT64_MAX/uint64_t(src.dims[i]))throw std::runtime_error("DSpark dimensions overflow");rows*=src.dims[i];}
        }
        if(src.dims[0]%ggml_blck_size(src.type))throw std::runtime_error("invalid DSpark quantized row: "+name);
        const uint64_t row_bytes=ggml_row_size(src.type,src.dims[0]);
        if(rows>UINT64_MAX/row_bytes || src.bytes!=row_bytes*rows)throw std::runtime_error("DSpark byte size mismatch: "+name);
        if(src.offset>uint64_t(file_bytes) || src.bytes>uint64_t(file_bytes)-src.offset)throw std::runtime_error("truncated DSpark tensor: "+name);
        if(!sources.emplace(name,std::move(src)).second)throw std::runtime_error("duplicate DSpark tensor: "+name);
    }
    std::vector<std::pair<uint64_t,uint64_t>> spans;
    for(const auto & source:sources)spans.emplace_back(source.second.offset,source.second.offset+source.second.bytes);
    std::sort(spans.begin(),spans.end());
    for(size_t i=1;i<spans.size();++i)if(spans[i].first<spans[i-1].second)throw std::runtime_error("overlapping DSpark tensor payloads");
    struct upload {const source * src;ggml_tensor * dst;size_t offset;};
    std::vector<upload> uploads;
    weights_ctx.reset(ggml_init({96*ggml_tensor_overhead(),nullptr,true}));
    if(!weights_ctx)throw std::runtime_error("cannot allocate DSpark tensor metadata");
    auto require=[&](const std::string & name,std::initializer_list<int64_t> dims) {
        std::vector<int64_t> shape(dims);
        std::vector<const source *> parts;
        if(shape.size()==3) {
            const auto at=name.find(".experts.")+std::strlen(".experts.");
            for(int e=0;e<shape[2];++e) {
                const auto key=name.substr(0,at)+std::to_string(e)+"."+name.substr(at);
                auto it=sources.find(key);if(it==sources.end())throw std::runtime_error("DSpark manifest missing "+key);
                parts.push_back(&it->second);
            }
        } else {
            auto it=sources.find(name);if(it==sources.end())throw std::runtime_error("DSpark manifest missing "+name);
            parts.push_back(&it->second);
        }
        for(const auto * part:parts) {
            auto actual=part->dims;
            while(actual.size()>1 && actual.back()==1)actual.pop_back();
            const std::vector<int64_t> expected(shape.begin(),shape.begin()+(shape.size()==3?2:shape.size()));
            if(actual!=expected || part->type!=parts[0]->type)throw std::runtime_error("DSpark tensor shape/type mismatch: "+name);
        }
        const auto type=parts[0]->type==GGML_TYPE_BF16?GGML_TYPE_F32:parts[0]->type;
        auto * t=ggml_new_tensor(weights_ctx.get(),type,shape.size(),shape.data());ggml_set_name(t,name.c_str());weights.emplace(name,t);
        for(size_t i=0;i<parts.size();++i)uploads.push_back({parts[i],t,shape.size()==3?i*t->nb[2]:0});
    };
    // This explicit allowlist excludes vision.*, aligner.*, image_*, and the
    // vision-only router bias_vl. Per-expert source slices share 3D tensors.
    for(int b=0;b<3;++b) {
        const auto prefix="mtp."+std::to_string(b)+".";
        auto req=[&](const char * n,std::initializer_list<int64_t> shape){require(prefix+n,shape);};
        req("attn.attn_sink",{NH});req("attn.kv_norm.weight",{DH});req("attn.q_norm.weight",{m.hparams.n_lora_q});
        req("attn.wkv.weight",{D,DH});req("attn.wo_a.weight",{OGD,OG*OL});req("attn.wo_b.weight",{OG*OL,D});
        req("attn.wq_a.weight",{D,m.hparams.n_lora_q});req("attn.wq_b.weight",{m.hparams.n_lora_q,NH*DH});
        for(auto site:{"attn","ffn"}) {
            req((std::string(site)+"_norm.weight").c_str(),{D});
            const auto hc=std::string("hc_")+site;
            req((hc+"_base").c_str(),{2*HC+HC*HC});req((hc+"_fn").c_str(),{D*HC,2*HC+HC*HC});req((hc+"_scale").c_str(),{3});
        }
        req("ffn.gate.bias",{NE});req("ffn.gate.weight",{D,NE});
        req("ffn.shared_experts.w1.weight",{D,m.ff});req("ffn.shared_experts.w3.weight",{D,m.ff});req("ffn.shared_experts.w2.weight",{m.ff,D});
        req("ffn.experts.w1.weight",{D,m.ff,NE});req("ffn.experts.w3.weight",{D,m.ff,NE});req("ffn.experts.w2.weight",{m.ff,D,NE});
    }
    require("mtp.0.main_proj.weight",{3*D,D});require("mtp.0.main_norm.weight",{D});require("mtp.2.norm.weight",{D});
    require("mtp.2.markov_head.embed.weight",{m.dspark_rank,V});require("mtp.2.markov_head.head.weight",{m.dspark_rank,V});
    require("mtp.2.confidence_head.proj.weight",{D+m.dspark_rank});
    weights_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx.get(),buft));
    if(!weights_buf)throw std::runtime_error("cannot allocate DSpark weights");
    ggml_backend_buffer_set_usage(weights_buf.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<char> chunk(8*1024*1024);std::vector<float> converted;size_t loaded=0;
    for(const auto & upload:uploads) {
        const auto & src=*upload.src;stream.seek(src.offset,SEEK_SET);
        for(size_t at=0;at<src.bytes;) {
            const size_t n=std::min<uint64_t>(chunk.size(),src.bytes-at);
            stream.read_raw(chunk.data(),n);
            if(src.type==GGML_TYPE_BF16) {
                converted.resize(n/sizeof(ggml_bf16_t));
                ggml_bf16_to_fp32_row(reinterpret_cast<const ggml_bf16_t *>(chunk.data()),converted.data(),converted.size());
                ggml_backend_tensor_set(upload.dst,converted.data(),upload.offset+at*2,n*2);
            } else ggml_backend_tensor_set(upload.dst,chunk.data(),upload.offset+at,n);
            at+=n;
        }
        loaded+=src.type==GGML_TYPE_BF16?src.bytes*2:src.bytes;
    }
    weights["output.weight"]=m.output;
    state_ctx.reset(ggml_init({32*ggml_tensor_overhead(),nullptr,true}));
    if(!state_ctx)throw std::runtime_error("cannot allocate DSpark state metadata");
    auto * c=state_ctx.get();
    for(int b=0;b<3;++b)ring.push_back(ggml_new_tensor_2d(c,GGML_TYPE_F32,DH,WIN));
    i_main=ggml_new_tensor_2d(c,GGML_TYPE_F32,3*D,WIN);i_mpos=ggml_new_tensor_1d(c,GGML_TYPE_I32,WIN);i_mslot=ggml_new_tensor_1d(c,GGML_TYPE_I64,WIN);
    i_emb=ggml_new_tensor_2d(c,GGML_TYPE_F32,D,BS);i_bpos=ggml_new_tensor_1d(c,GGML_TYPE_I32,BS);
    i_mask=ggml_new_tensor_2d(c,GGML_TYPE_F32,MP,BS);i_bonus=ggml_new_tensor_1d(c,GGML_TYPE_I32,1);i_pre=ggml_new_tensor_2d(c,GGML_TYPE_F32,HC,BS);
    state_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(c,buft));
    if(!state_buf)throw std::runtime_error("cannot allocate DSpark state");
    ggml_backend_buffer_clear(state_buf.get(),0);
    std::vector<float> pre(HC*BS,0.f);for(int t=0;t<BS;++t)pre[t*HC]=1.f;
    ggml_backend_tensor_set(i_pre,pre.data(),0,pre.size()*sizeof(float));
    alloc.reset(ggml_gallocr_new(buft));if(!alloc)throw std::runtime_error("cannot create DSpark allocator");
    filled.assign(WIN,-1);
    // Allocate both maximum-shape workspaces before the target's expert cache
    // is sized. The same allocator retains sufficient capacity for both.
    if(!ggml_gallocr_reserve(alloc.get(),build(WIN,false).gf) ||
       !ggml_gallocr_reserve(alloc.get(),build(0,true).gf))throw std::runtime_error("cannot reserve DSpark workspace");
    std::fprintf(stderr,"[dsv4f-mtp] loaded %zu text slices into %zu tensors, %.2f MiB on %s; vision tensors loaded=0; block=%d\n",uploads.size(),weights.size()-1,loaded/1048576.,ggml_backend_name(backend.get()),BS);
}

llama_dsv41_mtp::graph & llama_dsv41_mtp::build(int k, bool draft) {
    graph & gr = graphs[{ k, draft }];
    if (gr.gf) return gr;
    ggml_init_params ip = { 16384 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16384, false), nullptr, true };
    gr.ctx.reset(ggml_init(ip));
    ggml_context * c = gr.ctx.get();
    if(!c)throw std::runtime_error("cannot allocate DSpark graph metadata");
    ggml_cgraph * gf = gr.gf = ggml_new_graph_custom(c, 16384, false);
    const auto & Gm = model;
    auto rms_w = [&](ggml_tensor * x, ggml_tensor * wt) { return ggml_mul(c, ggml_rms_norm(c, x, (float) Gm.eps), wt); };
    // RoPE over the last d_rope dims, NORMAL pairs; DSpark layers have compress ratio 0: rope_theta, no YaRN
    auto rope = [&](ggml_tensor * a, ggml_tensor * pos, bool back) {
        const int nd = (int) Gm.hparams.n_rot();
        ggml_tensor * r = back ? ggml_rope_ext_back(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, 0, (float) Gm.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f)
                               : ggml_rope_ext(c, a, pos, nullptr, nd, GGML_ROPE_TYPE_NORMAL, 0, (float) Gm.rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return ggml_rope_set_offset(r, (int) (a->ne[0] - nd));
    };
    // Block.hc_mixes (the same composite as ds41_dense's B::hc_mixes)
    auto hc_mixes = [&](ggml_tensor * x, ggml_tensor * fn, ggml_tensor * sc, ggml_tensor * bs,
                        ggml_tensor ** pre, ggml_tensor ** post, ggml_tensor ** comb) {
        const int64_t nt = x->ne[2];
        ggml_tensor * flat = ggml_rms_norm(c, ggml_reshape_2d(c, x, HC * D, nt), (float) Gm.eps);
        ggml_tensor * mixes = ggml_mul_mat(c, fn, flat);
        ggml_prec_set_acc(mixes, GGML_PREC_F32);
        ggml_tensor * m_pre = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], 0);
        ggml_tensor * m_post = ggml_view_2d(c, mixes, HC, nt, mixes->nb[1], (size_t) HC * mixes->nb[0]);
        *pre = ggml_add(c, ggml_mul(c, m_pre, ggml_view_1d(c, sc, 1, 0)), ggml_view_1d(c, bs, HC, 0));
        *pre = ggml_scale_bias(c, ggml_sigmoid(c, *pre), 1.0f, (float) Gm.hc_eps);
        *post = ggml_add(c, ggml_mul(c, m_post, ggml_view_1d(c, sc, 1, sizeof(float))), ggml_view_1d(c, bs, HC, (size_t) HC * sizeof(float)));
        *post = ggml_scale(c, ggml_sigmoid(c, *post), 2.0f);
        *comb = ggml_dsv4_hc_comb(c, mixes, sc, bs, (float) Gm.hc_eps, (int32_t) Gm.hparams.dsv4_hc_sinkhorn_iters);
    };

    // ---- main_x and the window writes (model.py forward_embed: main_norm(main_proj(main_hidden)))
    std::vector<ggml_tensor *> ring_w=ring;
    if(k>0) {
    ggml_tensor * main = ggml_view_2d(c, i_main, 3 * D, k, i_main->nb[1], 0);
    ggml_tensor * mx = rms_w(ggml_mul_mat(c, W("mtp.0.main_proj.weight"), main), W("mtp.0.main_norm.weight"));   // [D, k]
    ggml_tensor * mpos = ggml_view_1d(c, i_mpos, k, 0);
    ggml_tensor * mslot = ggml_view_1d(c, i_mslot, k, 0);
    for (int b = 0; b < 3; ++b) {   // DSparkAttention: main_kv = kv_norm(wkv(main_x)), RoPE at its position, FP8-rounded
        ggml_tensor * mkv = rms_w(ggml_mul_mat(c, BW(b, "attn.wkv.weight"), mx), BW(b, "attn.kv_norm.weight"));
        mkv = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, mkv, DH, 1, k), mpos, false), DH, k);
        mkv = ggml_dsv41_fake_quant(c, ggml_cont(c, mkv), 0);
        ring_w[b] = ggml_set_rows(c, ring[(size_t) b], mkv, mslot);
    }
    }
    if (!draft) {
        for (auto * t : ring_w) ggml_build_forward_expand(gf, t);
        return gr;
    }

    // ---- the block: [bonus, noise x4] at positions p+1..p+5, hc copies of the embedding, identity pre-mix
    ggml_tensor * bpos = i_bpos;
    ggml_tensor * x = ggml_repeat_4d(c, ggml_reshape_3d(c, i_emb, D, 1, BS), D, HC, BS, 1);
    ggml_tensor * pre = i_pre;
    ggml_tensor * m4 = ggml_cast(c, ggml_reshape_4d(c, i_mask, MP, 1, 1, BS), GGML_TYPE_F16);
    const float lim = (float) Gm.clamp;
    for (int b = 0; b < 3; ++b) {
        // attention (Block.forward: hc_mixes of the input, hc_pre with the previous FFN's pre-mix, attn_norm)
        ggml_tensor * a_pre, * a_post, * a_comb;
        hc_mixes(x, BW(b, "hc_attn_fn"), BW(b, "hc_attn_scale"), BW(b, "hc_attn_base"), &a_pre, &a_post, &a_comb);
        ggml_tensor * xn = rms_w(ggml_dsv4_hc_pre(c, x, pre), BW(b, "attn_norm.weight"));
        ggml_tensor * qr = rms_w(ggml_mul_mat(c, BW(b, "attn.wq_a.weight"), xn), BW(b, "attn.q_norm.weight"));
        ggml_tensor * q = rope(ggml_reshape_3d(c, ggml_mul_mat(c, BW(b, "attn.wq_b.weight"), qr), DH, NH, BS), bpos, false);
        ggml_tensor * kvb = rms_w(ggml_mul_mat(c, BW(b, "attn.wkv.weight"), xn), BW(b, "attn.kv_norm.weight"));
        kvb = ggml_reshape_2d(c, rope(ggml_reshape_3d(c, kvb, DH, 1, BS), bpos, false), DH, BS);
        kvb = ggml_dsv41_fake_quant(c, ggml_cont(c, kvb), 0);
        // keys = [window ring (incl. this step's rows) | the 5 block rows | pad], every query sees all of them
        ggml_tensor * K = ggml_pad(c, ggml_concat(c, ring_w[(size_t) b], kvb, 1), 0, (int) (MP - WIN - BS), 0, 0);   // [DH, MP]
        ggml_tensor * k4 = ggml_cast(c, ggml_repeat_4d(c, ggml_reshape_4d(c, K, DH, MP, 1, 1), DH, MP, 1, BS), GGML_TYPE_F16);
        ggml_tensor * q4 = ggml_permute(c, ggml_reshape_4d(c, q, DH, NH, 1, BS), 0, 2, 1, 3);    // [DH, 1, NH, BS]
        ggml_tensor * o = ggml_flash_attn_ext(c, q4, k4, k4, m4, 1.0f / std::sqrt((float) DH), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, BW(b, "attn.attn_sink"));
        o = rope(ggml_reshape_3d(c, o, DH, NH, BS), bpos, true);
        o = ggml_permute(c, ggml_reshape_3d(c, o, OGD, OG, BS), 0, 2, 1, 3);                      // [OGD, BS, OG]
        ggml_tensor * wo_a = ggml_reshape_3d(c, BW(b, "attn.wo_a.weight"), OGD, OL, OG);
        ggml_tensor * oa = ggml_permute(c, ggml_mul_mat(c, wo_a, o), 0, 2, 1, 3);                 // [OL, OG, BS]
        ggml_tensor * attn = ggml_mul_mat(c, BW(b, "attn.wo_b.weight"), ggml_cont_2d(c, oa, OL * OG, BS));
        ggml_tensor * h = ggml_dsv4_hc_post(c, attn, x, a_post, a_comb);
        // FFN: its mixes give the next block's pre-mix; its input uses the attention's pre-mix
        ggml_tensor * f_pre, * f_post, * f_comb;
        hc_mixes(h, BW(b, "hc_ffn_fn"), BW(b, "hc_ffn_scale"), BW(b, "hc_ffn_base"), &f_pre, &f_post, &f_comb);
        ggml_tensor * fn = rms_w(ggml_dsv4_hc_pre(c, h, a_pre), BW(b, "ffn_norm.weight"));   // [D, BS]
        // Gate: sqrt(softplus), top-NU on score + bias, weights from the raw scores
        ggml_tensor * rl = ggml_mul_mat(c, BW(b, "ffn.gate.weight"), fn);
        ggml_prec_set_acc(rl, GGML_PREC_F32);
        ggml_tensor * sc = ggml_sqrt(c, ggml_softplus(c, rl));                                  // [NE, BS]
        ggml_tensor * ids = ggml_top_k(c, ggml_add(c, sc, BW(b, "ffn.gate.bias")), (int) NU);   // [NU, BS]
        ggml_tensor * wts = ggml_reshape_2d(c, ggml_get_rows(c, ggml_reshape_3d(c, sc, 1, sc->ne[0], BS), ids), NU, BS);
        if (Gm.norm_topk && NU > 1) wts = ggml_div(c, wts, ggml_scale_bias(c, ggml_sum_rows(c, wts), 1.0f, 1e-20f));
        wts = ggml_scale(c, wts, (float) Gm.scaling);
        // routed experts (Expert.forward: clamped SwiGLU, the routing weight applied before w2)
        ggml_tensor * x3 = ggml_reshape_3d(c, fn, D, 1, BS);
        ggml_tensor * ge = ggml_mul_mat_id(c, BW(b, "ffn.experts.w1.weight"), x3, ids);         // [NFF, NU, BS]
        ggml_tensor * ue = ggml_mul_mat_id(c, BW(b, "ffn.experts.w3.weight"), x3, ids);
        if (lim > 0) { ue = ggml_clamp(c, ue, -lim, lim); ge = ggml_clamp(c, ge, -INFINITY, lim); }
        ggml_tensor * act = ggml_mul(c, ggml_swiglu_split(c, ge, ue), ggml_reshape_3d(c, wts, 1, NU, BS));
        ggml_tensor * de = ggml_mul_mat_id(c, BW(b, "ffn.experts.w2.weight"), act, ids);        // [D, NU, BS]
        ggml_tensor * routed = ggml_view_2d(c, de, D, BS, de->nb[2], 0);
        for (int64_t u = 1; u < NU; ++u) routed = ggml_add(c, routed, ggml_view_2d(c, de, D, BS, de->nb[2], (size_t) u * de->nb[1]));
        // shared expert
        ggml_tensor * up = ggml_mul_mat(c, BW(b, "ffn.shared_experts.w3.weight"), fn);
        ggml_tensor * gt = ggml_mul_mat(c, BW(b, "ffn.shared_experts.w1.weight"), fn);
        if (lim > 0) { up = ggml_clamp(c, up, -lim, lim); gt = ggml_clamp(c, gt, -INFINITY, lim); }
        ggml_tensor * se = ggml_mul_mat(c, BW(b, "ffn.shared_experts.w2.weight"), ggml_swiglu_split(c, gt, up));
        x = ggml_dsv4_hc_post(c, ggml_add(c, routed, se), h, f_post, f_comb);
        pre = f_pre;
    }
    // ---- forward_head: hc_pre -> norm -> trunk head; Markov bias left to right; confidence
    ggml_tensor * hn = ggml_dsv4_hc_pre(c, x, pre);                                             // [D, BS] (pre-norm)
    ggml_tensor * lg = ggml_mul_mat(c, W("output.weight"), rms_w(hn, W("mtp.2.norm.weight")));     // [V, BS]
    ggml_prec_set_acc(lg, GGML_PREC_F32);
    ggml_tensor * tok = i_bonus;
    for (int64_t i = 0; i < BS; ++i) {
        ggml_tensor * e = ggml_get_rows(c, W("mtp.2.markov_head.embed.weight"), tok);            // [256, 1]
        ggml_tensor * bias = ggml_mul_mat(c, W("mtp.2.markov_head.head.weight"), e);             // [V, 1]
        ggml_prec_set_acc(bias, GGML_PREC_F32);
        ggml_tensor * li = ggml_add(c, ggml_view_2d(c, lg, V, 1, lg->nb[1], (size_t) i * lg->nb[1]), bias);
        tok = ggml_argmax(c, li);                                                                 // I32 [1]
        ggml_tensor * hc = ggml_concat(c, ggml_view_2d(c, hn, D, 1, hn->nb[1], (size_t) i * hn->nb[1]), e, 0);
        ggml_tensor * cf = ggml_mul_mat(c, W("mtp.2.confidence_head.proj.weight"), hc);           // [1, 1]
        ggml_set_output(tok);
        ggml_set_output(cf);
        gr.ids.push_back(tok);
        gr.conf.push_back(cf);
        ggml_build_forward_expand(gf, tok);
        ggml_build_forward_expand(gf, cf);
    }
    return gr;
}

bool llama_dsv41_mtp::compute(int k,bool draft) {
    auto & g=build(k,draft);
    // One allocator serves all shapes. Cached graphs must not retain pointers
    // into a previous allocation after reserve() grows its buffer.
    for(auto * t=ggml_get_first_tensor(g.ctx.get());t;t=ggml_get_next_tensor(g.ctx.get(),t)){t->data=nullptr;t->buffer=nullptr;}
    const auto key=std::make_pair(k,draft);
    if(key!=allocated_key) {
        if(!ggml_gallocr_reserve(alloc.get(),g.gf))return false;
        allocated_key=key;
    }
    return ggml_gallocr_alloc_graph(alloc.get(),g.gf) && ggml_backend_graph_compute(backend.get(),g.gf)==GGML_STATUS_SUCCESS;
}
llama_dsv41_mtp * llama_dsv41_mtp_init(llama_context * target,const char * path,const char * manifest) {
    try {
        if(!target || !llama_dsv41_mtp_supported(llama_get_model(target)))throw std::runtime_error("target has no V4.1 DSpark configuration");
        return new llama_dsv41_mtp(target,path,manifest);
    } catch(const std::exception & e){std::fprintf(stderr,"[dsv4f-mtp] %s\n",e.what());return nullptr;}
}
void llama_dsv41_mtp_free(llama_dsv41_mtp * m){delete m;}
void llama_dsv41_mtp_reset(llama_dsv41_mtp * m){m->filled.assign(m->WIN,-1);}
int llama_dsv41_mtp_block_size(const llama_dsv41_mtp * m){return m->BS;}
int llama_dsv41_mtp_window_size(const llama_dsv41_mtp * m){return m->WIN;}
const int32_t * llama_dsv41_mtp_layers(const llama_dsv41_mtp * m){return m->model.dspark_layers.data();}
bool llama_dsv41_mtp_append(llama_dsv41_mtp * m,const float * taps,const int32_t * positions,int n) {
    try {
        if(!m || !taps || !positions || n<1 || n>m->WIN)return false;
        std::vector<int64_t> slots(n);
        for(int i=0;i<n;++i) {
            if(positions[i]<0 || (i && positions[i]!=positions[i-1]+1))return false;
            slots[i]=positions[i]%m->WIN;
        }
        ggml_backend_tensor_set(m->i_main,taps,0,size_t(n)*3*m->D*sizeof(float));
        ggml_backend_tensor_set(m->i_mpos,positions,0,n*sizeof(int32_t));
        ggml_backend_tensor_set(m->i_mslot,slots.data(),0,n*sizeof(int64_t));
        if(!m->compute(n,false))return false;
        for(int i=0;i<n;++i)m->filled[slots[i]]=positions[i];
        return true;
    } catch(const std::exception & e){std::fprintf(stderr,"[dsv4f-mtp] append: %s\n",e.what());return false;}
}
bool llama_dsv41_mtp_draft(llama_dsv41_mtp * m,int32_t pos,llama_token anchor,llama_token * ids,float * confidence) {
    try {
        if(!m || !ids || !confidence || pos<0 || anchor<0 || anchor>=m->V)return false;
        const auto * embedding=m->model.tok_embd;
        const auto * traits=ggml_get_type_traits(embedding->type);
        if(!traits->to_float && embedding->type!=GGML_TYPE_F32)return false;
        std::vector<float> emb(size_t(m->D)*m->BS);
        std::vector<char> row(embedding->nb[1]);std::vector<int32_t> bp(m->BS);
        std::vector<float> mask(size_t(m->MP)*m->BS,-INFINITY);
        for(int t=0;t<m->BS;++t) {
            const int token=t?m->model.dspark_noise:anchor;
            ggml_backend_tensor_get(embedding,row.data(),size_t(token)*embedding->nb[1],row.size());
            if(embedding->type==GGML_TYPE_F32)std::memcpy(emb.data()+t*m->D,row.data(),m->D*sizeof(float));
            else traits->to_float(row.data(),emb.data()+t*m->D,m->D);
            bp[t]=pos+t;
            for(int i=0;i<m->WIN;++i)if(m->filled[i]>=std::max(0,pos-m->WIN) && m->filled[i]<pos)mask[t*m->MP+i]=0.f;
            std::fill(mask.begin()+t*m->MP+m->WIN,mask.begin()+t*m->MP+m->WIN+m->BS,0.f);
        }
        ggml_backend_tensor_set(m->i_emb,emb.data(),0,emb.size()*sizeof(float));
        ggml_backend_tensor_set(m->i_bpos,bp.data(),0,bp.size()*sizeof(int32_t));
        ggml_backend_tensor_set(m->i_mask,mask.data(),0,mask.size()*sizeof(float));
        ggml_backend_tensor_set(m->i_bonus,&anchor,0,sizeof(anchor));
        if(!m->compute(0,true))return false;
        auto & g=m->build(0,true);
        for(int i=0;i<m->BS;++i) {
            float logit;
            ggml_backend_tensor_get(g.ids[i],ids+i,0,sizeof(int32_t));
            ggml_backend_tensor_get(g.conf[i],&logit,0,sizeof(float));
            confidence[i]=1.f/(1.f+std::exp(-logit));
            if(ids[i]<0 || ids[i]>=m->V || !std::isfinite(confidence[i]))return false;
        }
        return true;
    } catch(const std::exception & e){std::fprintf(stderr,"[dsv4f-mtp] draft: %s\n",e.what());return false;}
}
