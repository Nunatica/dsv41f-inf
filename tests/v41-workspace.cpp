#include "llama.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <fstream>
#include <cstdlib>

// Real-model geometry regression without loading its 175 GiB of weights.
// Skip on machines without the optional model or a GPU backend.
int main(int argc, char ** argv) {
    if ((argc != 2 && argc != 3) || !std::ifstream(argv[1]).good()) return 77;
    ggml_backend_load_all();
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) return 77;

    auto mp = llama_model_default_params();
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.n_gpu_layers = 999;
    mp.use_extra_bufts = false;
    llama_model_tensor_buft_override overrides[] = {
        {"\\.ffn_(gate|up|down)_exps", ggml_backend_cpu_buffer_type()},
        {nullptr, nullptr},
    };
    mp.tensor_buft_overrides = overrides;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;

    auto cp = llama_context_default_params();
    cp.n_ctx = argc == 3 ? std::strtoul(argv[2], nullptr, 10) : 32768;
    cp.n_batch = cp.n_ubatch = 4096;
    cp.n_seq_max = 1;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto * ctx = llama_init_from_model(model, cp);
    bool ok = ctx && llama_n_batch(ctx) == 4096 && llama_n_ubatch(ctx) == 4096;
    size_t compute = 0, kv = 0;
    if (ctx) {
        for (const auto & item : ctx->memory_breakdown()) {
            const auto dev = ggml_backend_buft_get_device(item.first);
            if (dev && !ggml_backend_buft_is_host(item.first) &&
                ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                compute += item.second.compute; kv += item.second.context;
            }
        }
        // The old per-layer FILL templates reserved 27,442 MiB. This leaves
        // room within the 12 GiB cache budget for the real prefill workspace.
        ok &= compute > 0 && compute <= (12ULL << 30);
        if (cp.n_ctx == 262144) ok &= kv < (1ULL << 30);
        llama_free(ctx);
    }
    llama_model_free(model);
    std::printf("[dsv4f-v41-test] ctx=%u batch=4096 ubatch=4096 GPU compute=%.2f MiB KV=%.2f MiB %s\n",
                cp.n_ctx, compute / 1048576., kv / 1048576., ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
