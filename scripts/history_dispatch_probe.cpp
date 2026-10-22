#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-dsv41-cache.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// Isolate the real-model 40 Q8 raw windows + six F32 pooling histories.
// Initializing a SET_ROWS view's existing GPU buffer changes only placement
// visibility: it models resolving view_src when checking HISTORY support.
static std::vector<unsigned char> run(bool initialize_views) {
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    auto * cpu = ggml_backend_cpu_init();
    if (!gpu || !cpu) throw std::runtime_error("backend");
    ggml_backend_cpu_set_n_threads(cpu, 12);
    auto * dev = ggml_backend_get_device(gpu);
    auto * ctx = ggml_init({512*ggml_tensor_overhead()+ggml_graph_overhead_custom(512, false), nullptr, true});
    auto * hc = ggml_init({64*ggml_tensor_overhead(), nullptr, true});
    auto * row = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    auto * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    std::vector<ggml_tensor *> caches;
    std::vector<ggml_dsv41_history> histories(46);
    size_t full_bytes = 0, row_bytes = 0;
    for (int i = 0; i < 46; ++i) {
        const auto type = i < 40 ? GGML_TYPE_Q8_0 : GGML_TYPE_F32;
        auto * cache = ggml_new_tensor_2d(ctx, type, 512, 4352);
        ggml_format_name(cache, "history_probe_cache_%d", i);
        caches.push_back(cache);
        histories[i].host = ggml_new_tensor_2d(hc, type, 512, 1);
        full_bytes += ggml_nbytes(cache);
        row_bytes += ggml_nbytes(histories[i].host);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_dev_buffer_type(dev));
    auto * hb = ggml_backend_alloc_ctx_tensors_from_buft(hc, ggml_backend_dev_host_buffer_type(dev));
    if (!buffer || !hb) throw std::runtime_error("buffer");
    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_buffer_clear(hb, 0);
    auto map = reinterpret_cast<ggml_dsv41_map_host_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_dsv41_map_host"));
    if (!map) throw std::runtime_error("host mapping");
    for (auto & h : histories) h.mapped = map(dev, h.host->data);
    std::vector<float> values(512);
    for (int i = 0; i < 512; ++i) values[i] = (i%29-14)*0.03125f;
    ggml_backend_tensor_set(row, values.data(), 0, values.size()*sizeof(float));
    const int64_t zero = 0;
    ggml_backend_tensor_set(position, &zero, 0, sizeof(zero));
    auto * graph = ggml_new_graph_custom(ctx, 512, false);
    std::vector<ggml_tensor *> archives;
    int supported_before = 0, supported_after = 0;
    for (int i = 0; i < 46; ++i) {
        auto * view = ggml_set_rows(ctx, caches[i], row, position);
        auto * archive = ggml_dsv41_archive(ctx, &histories[i], view, position, position);
        supported_before += ggml_backend_dev_supports_op(dev, archive);
        if (initialize_views) ggml_backend_view_init(view);
        supported_after += ggml_backend_dev_supports_op(dev, archive);
        archives.push_back(archive);
        ggml_build_forward_expand(graph, archive);
    }
    ggml_backend_t backends[] = {gpu, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 512, false, true);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) throw std::runtime_error("graph allocation");
    int cpu_archives = 0;
    for (auto * archive : archives) cpu_archives += ggml_backend_sched_get_tensor_backend(sched, archive) == cpu;
    std::vector<double> times;
    for (int i = 0; i < 35; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("compute");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count();
        if (i >= 5) times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    std::vector<unsigned char> output;
    for (auto & h : histories) {
        const auto * first = static_cast<const unsigned char *>(h.host->data);
        output.insert(output.end(), first, first+ggml_nbytes(h.host));
    }
    std::printf("{\"initialized_views\":%s,\"supported_before\":%d,\"supported_after\":%d,\"cpu_archives\":%d,\"full_window_mib\":%.6f,\"updated_row_kib\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f}\n",
        initialize_views?"true":"false", supported_before, supported_after, cpu_archives,
        full_bytes/1048576., row_bytes/1024., times[times.size()/2], times[size_t(.95*(times.size()-1))]);
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(hb);
    ggml_backend_buffer_free(buffer);
    ggml_free(hc);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    ggml_backend_free(gpu);
    return output;
}

int main() {
    try {
        ggml_backend_load_all();
        const auto original = run(false), initialized = run(true);
        if (original != initialized) throw std::runtime_error("history bytes differ");
        std::printf("history bytes identical: %zu bytes\n", original.size());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
