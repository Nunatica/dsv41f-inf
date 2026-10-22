#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_can_fuse_hc_post_pre(const ggml_tensor * post, const ggml_tensor * pre);
void ggml_cuda_op_hc_post_pre(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * pre);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_can_fuse_hc_post_full(const ggml_tensor * post, const ggml_tensor * full);
void ggml_cuda_op_hc_post_full(ggml_backend_cuda_context & ctx, ggml_tensor * post, ggml_tensor * full);

void ggml_cuda_op_dsv41_fake_quant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_can_fuse_fq_store(const ggml_tensor * fq, const ggml_tensor * store);
void ggml_cuda_op_fq_store(ggml_backend_cuda_context & ctx, ggml_tensor * fq, ggml_tensor * store);

void ggml_cuda_op_dsv41_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
