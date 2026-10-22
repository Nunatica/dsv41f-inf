#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Fixed-route grouped Q2 MMQ, with both loaders in the same executable.
// Timing excludes input quantization and weight transfer; the expert test
// separately exercises PTQ transcode and the complete streamed pipeline.
int main(int argc, char ** argv) {
    const int K = argc>1 ? std::atoi(argv[1]) : 5120;
    const int M = argc>2 ? std::atoi(argv[2]) : 2304;
    const int tokens = argc>3 ? std::atoi(argv[3]) : 32;
    const int experts = argc>4 ? std::atoi(argv[4]) : 16;
    const int J = argc>5 ? std::atoi(argv[5]) : 32;
    const int mode = argc>6 ? std::atoi(argv[6]) : 0;
    const int repeats = argc>7 ? std::atoi(argv[7]) : 30;
    const bool skew = argc>8 && !std::strcmp(argv[8], "skew");
    if (K<256 || K%256 || M<1 || tokens<1 || experts<1 || repeats<1 || mode<0 || mode>1) return 2;
    ggml_backend_load_all();
    auto gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) return 77;
    auto & ctx = *static_cast<ggml_backend_cuda_context *>(gpu->context);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!GGML_CUDA_CC_IS_RDNA3_0(cc)) return 77;
    const bool fallback = ggml_cuda_mmq_needs_fallback(M);
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q2_0, J, fallback, cc, GGML_PREC_Q8);
    if (config.type == GGML_TYPE_COUNT) return 2;
    std::vector<int32_t> bounds(experts+1, 0);
    for (int e=0; e<experts; ++e) bounds[e+1] = bounds[e] + (skew && experts>2 ? e==0 ? 0 : std::max(1, tokens-e%5) : tokens);
    const int total = bounds.back();
    auto context = ggml_init({4<<20, nullptr, true});
    auto w = ggml_new_tensor_3d(context, GGML_TYPE_Q2_0, K, M, experts);
    auto x = ggml_new_tensor_2d(context, GGML_TYPE_F32, K, total);
    auto y = ggml_new_tensor_1d(context, GGML_TYPE_I8, int64_t(K/128)*total*sizeof(block_q8_1_mmq));
    auto ids = ggml_new_tensor_1d(context, GGML_TYPE_I32, total);
    auto eb = ggml_new_tensor_1d(context, GGML_TYPE_I32, experts+1);
    auto out = ggml_new_tensor_2d(context, GGML_TYPE_F32, M, total);
    auto buffer = ggml_backend_alloc_ctx_tensors(context, gpu);
    if (!buffer) return 3;
    std::mt19937 rng(4101);
    std::vector<block_q2_0> weights(ggml_nbytes(w)/sizeof(block_q2_0));
    const float scale_values[] = {.03125f, -.125f, 0.f, .0001f, .75f};
    size_t block_index = 0;
    for (auto & b: weights) {
        const auto scale = ggml_fp32_to_fp16(scale_values[block_index++%5]);
        std::memcpy(&b.d, &scale, 2);
        for (auto & q: b.qs) q=rng()%256;
    }
    std::vector<float> input(size_t(K)*total);
    for (auto & v: input) v=(int(rng()%65536)-32768)*(.5f/32768);
    std::vector<int32_t> identity(total);
    for (int i=0; i<total; ++i) identity[i]=total-1-i;
    ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids, identity.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(eb, bounds.data(), 0, ggml_nbytes(eb));
    const auto stream = ctx.stream();
    quantize_mmq_q8_1_cuda((float *)x->data, nullptr, y->data, GGML_TYPE_Q2_0, K, K, K*total, K*total, K, total, 1, 1, stream);
    mmq_args args = {(const char *)w->data, GGML_TYPE_Q2_0, (const int *)y->data, (const int32_t *)ids->data,
        (const int32_t *)eb->data, (float *)out->data, nullptr,
        K, M, total, K/64, total, M, experts, experts, int64_t(K/64)*M, 0, 0,
        1, 1, 0, 0, 0, tokens, J};
    auto launch = [&](bool tuned) {
        if (tuned && J==32) {
            if (fallback) launch_mul_mat_q<GGML_TYPE_Q2_0, 32, true, GGML_PREC_Q8, true>(ctx, args, stream);
            else launch_mul_mat_q<GGML_TYPE_Q2_0, 32, false, GGML_PREC_Q8, true>(ctx, args, stream);
        } else {
            if (fallback) mul_mat_q_switch_J<GGML_TYPE_Q2_0, true, GGML_PREC_Q8>(ctx, args, stream);
            else mul_mat_q_switch_J<GGML_TYPE_Q2_0, false, GGML_PREC_Q8>(ctx, args, stream);
        }
    };
    std::vector<float> reference(size_t(M)*total), actual(reference.size());
    launch(false); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
    ggml_backend_tensor_get(out, reference.data(), 0, ggml_nbytes(out));
    launch(true); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaStreamSynchronize(stream));
    ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
    bool ok = std::memcmp(reference.data(), actual.data(), ggml_nbytes(out))==0;
    for (float v: actual) ok &= std::isfinite(v);
    std::vector<block_q8_1_mmq> activations(ggml_nbytes(y)/sizeof(block_q8_1_mmq));
    ggml_backend_tensor_get(y, activations.data(), 0, ggml_nbytes(y));
    double err2=0, ref2=0, maximum=0;
    for (int e=0; e<experts; ++e) {
        if (bounds[e]==bounds[e+1]) continue;
        for (int c: {bounds[e], (bounds[e]+bounds[e+1]-1)/2, bounds[e+1]-1}) for (int r: {0, M/2, M-1}) {
            double expected=0;
            for (int k=0; k<K; ++k) {
                const auto & a=weights[(size_t(e)*M+r)*(K/64)+k/64];
                const auto & b=activations[size_t(k/128)*total+c];
                const int q=int((a.qs[(k%64)/4] >> (2*(k%4)))&3)-1;
                ggml_fp16_t d;
                std::memcpy(&d, &a.d, 2);
                expected+=double(q)*ggml_fp16_to_fp32(d)*int(b.qs[k%128])*b.d4[(k%128)/32];
            }
            const double delta=actual[size_t(identity[c])*M+r]-expected;
            err2+=delta*delta; ref2+=expected*expected; maximum=std::max(maximum,std::abs(delta));
        }
    }
    const double nmse=err2/std::max(ref2,1e-30);
    ok &= maximum<.001 && nmse<1e-8;
    cudaGraph_t graph; cudaGraphExec_t executable;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    launch(mode==1);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    std::vector<double> ms;
    for (int i=-5; i<repeats; ++i) {
        const auto start=std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphLaunch(executable, stream)); CUDA_CHECK(cudaStreamSynchronize(stream));
        if (i>=0) ms.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
    ok &= std::memcmp(reference.data(), actual.data(), ggml_nbytes(out))==0;
    std::sort(ms.begin(),ms.end());
    std::printf("mmq K=%d M=%d tokens=%d experts=%d J=%d mode=%d skew=%d I=%d threads=%d median_ms=%.6f p95_ms=%.6f cpu_max=%.9g cpu_nmse=%.9g bit_exact=%d %s\n",
        K,M,tokens,experts,J,mode,int(skew),config.I,config.nthreads,ms[ms.size()/2],ms[std::min(ms.size()-1,ms.size()*95/100)],maximum,nmse,int(ok),ok?"PASS":"FAIL");
    CUDA_CHECK(cudaGraphExecDestroy(executable)); CUDA_CHECK(cudaGraphDestroy(graph));
    ggml_backend_buffer_free(buffer); ggml_free(context); ggml_backend_free(gpu);
    return ok?0:1;
}
