// CPU-only profiling of real Q3_K or PTQ1_0 miss calls, with fixed inputs/routes/hits.
#include "cpu-profile-trace.h"
#include "moe.h"
#include "ggml-cpu.h"
#include "ggml-backend-moe-cache.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

struct call {
    dsv4f_cpu_trace_record h;
    std::vector<int32_t> ids;
    std::vector<float> probabilities, x;
    ggml_tensor * t[3] = {};
    int misses = 0;
};
static double cpu_seconds(const rusage & r) {
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}
int main(int argc, char ** argv) try {
    if (argc != 8) throw std::runtime_error("usage: cpu-trace-replay MODEL TRACE SKIP COUNT REPEATS stream|hot MISSES(0=all)");
    const size_t skip = std::stoull(argv[3]), count = std::stoull(argv[4]);
    const int repeats = std::stoi(argv[5]), filter = std::stoi(argv[7]);
    const bool hot = std::string(argv[6]) == "hot";
    if ((!hot && std::string(argv[6]) != "stream") || !count || repeats < 1 || filter < 0 || filter > 64)
        throw std::runtime_error("invalid profiling options");
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input trace");
    std::vector<call> calls;
    size_t source_count = 0;
    for (;;) {
        call c;
        input.read(reinterpret_cast<char *>(&c.h), sizeof(c.h));
        if (!input && input.eof() && input.gcount() == 0) break;
        if (!input || (c.h.magic != DSV4F_TRACE_Q3 && c.h.magic != DSV4F_TRACE_PTQ) || !c.h.nk || c.h.nk > 64 || c.h.ne > 16384 || c.h.nf > 16384)
            throw std::runtime_error("invalid trace header");
        c.ids.resize(c.h.nk); c.probabilities.resize(c.h.nk); c.x.resize(c.h.ne);
        input.read(reinterpret_cast<char *>(c.ids.data()), c.ids.size()*sizeof(int32_t));
        input.read(reinterpret_cast<char *>(c.probabilities.data()), c.probabilities.size()*sizeof(float));
        input.read(reinterpret_cast<char *>(c.x.data()), c.x.size()*sizeof(float));
        if (!input) throw std::runtime_error("truncated trace payload");
        for (size_t i = 0; i < c.h.nk; ++i)
            c.misses += c.probabilities[i] != 0 && !(c.h.hits & (uint64_t(1) << i));
        if (source_count++ < skip || (filter && c.misses != filter)) continue;
        calls.push_back(std::move(c));
        if (calls.size() == count || hot) break;
    }
    if (calls.empty()) throw std::runtime_error("empty selected trace");
    ggml_context * metadata = nullptr;
    auto * gguf = gguf_init_from_file(argv[1], {true, &metadata});
    if (!gguf || !metadata) throw std::runtime_error("cannot read GGUF metadata");
    const int fd = open(argv[1], O_RDONLY);
    struct stat status;
    if (fd < 0 || fstat(fd, &status)) throw std::runtime_error("cannot open model");
    auto * mapped = static_cast<char *>(mmap(nullptr, status.st_size, PROT_READ, MAP_SHARED, fd, 0));
    if (mapped == MAP_FAILED) throw std::runtime_error("cannot mmap model");
    std::set<std::pair<std::string, int>> expert_weights;
    size_t unique_bytes = 0, stream_bytes = 0;
    int max_ne = 0;
    for (auto & c : calls) {
        max_ne = std::max(max_ne, int(c.h.ne));
        for (int j = 0; j < 3; ++j) {
            if (!std::memchr(c.h.names[j], 0, 64)) throw std::runtime_error("unterminated tensor name");
            c.t[j] = ggml_get_tensor(metadata, c.h.names[j]);
            const auto tid = gguf_find_tensor(gguf, c.h.names[j]);
            if (!c.t[j] || tid < 0 || c.t[j]->type != (c.h.magic == DSV4F_TRACE_Q3 ? GGML_TYPE_Q3_K : GGML_TYPE_PTQ1_0) ||
                c.t[j]->ne[0] != (j == 2 ? c.h.nf : c.h.ne) || c.t[j]->ne[1] != (j == 2 ? c.h.ne : c.h.nf))
                throw std::runtime_error("trace/model tensor mismatch");
            c.t[j]->data = mapped + gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tid);
            for (size_t i = 0; i < c.h.nk; ++i) if (c.probabilities[i] != 0 && !(c.h.hits & (uint64_t(1) << i))) {
                if (c.ids[i] < 0 || c.ids[i] >= c.t[j]->ne[2]) throw std::runtime_error("invalid expert index");
                stream_bytes += c.t[j]->nb[2];
                if (expert_weights.insert({c.h.names[j], c.ids[i]}).second) unique_bytes += c.t[j]->nb[2];
            }
        }
    }
    ggml_cpu_init();
    auto * init = ggml_init({1 << 20, nullptr, true});
    // Construct an unallocated operator to register the native CPU callback.
    auto * x = ggml_new_tensor_2d(init, GGML_TYPE_F32, calls[0].h.ne, 1);
    auto * ids = ggml_new_tensor_2d(init, GGML_TYPE_I32, calls[0].h.nk, 1);
    auto * probabilities = ggml_new_tensor_3d(init, GGML_TYPE_F32, 1, calls[0].h.nk, 1);
    dsv4f_moe(init, x, ids, probabilities, calls[0].t[0], calls[0].t[1], calls[0].t[2], calls[0].h.limit);
    if (!ggml_dsv4f_cpu_experts) throw std::runtime_error("CPU service not initialized");
    std::vector<float> result(max_ne);
    auto run = [&](const call & c) {
        ggml_dsv4f_cpu_experts(c.t[0], c.t[1], c.t[2], c.x.data(), c.ids.data(), c.probabilities.data(),
                              c.h.nk, c.h.hits, c.h.limit, result.data());
    };
    // Page faults and first-use thread creation are outside measurements.
    for (int pass = 0; pass < 2; ++pass) for (const auto & c : calls) run(c);
    std::vector<double> times;
    double checksum = 0;
    rusage before{}, after{};
    getrusage(RUSAGE_SELF, &before);
    for (int rep = 0; rep < repeats; ++rep) {
        const auto start = std::chrono::steady_clock::now();
        const size_t n = hot ? count : calls.size();
        for (size_t i = 0; i < n; ++i) {
            run(calls[hot ? 0 : i]);
            checksum += result[i % result.size()];
        }
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    getrusage(RUSAGE_SELF, &after);
    double total = 0; for (double ms : times) total += ms;
    std::sort(times.begin(), times.end());
    const size_t n = hot ? count : calls.size();
    const double bytes = hot ? double(stream_bytes)*count : stream_bytes;
    std::printf("{\"mode\":\"%s\",\"calls_per_repeat\":%zu,\"repeats\":%d,\"unique_weight_mib\":%.4f,\"nominal_weight_mib_per_repeat\":%.4f,\"mean_ms_per_call\":%.6f,\"median_repeat_ms\":%.6f,\"nominal_weight_GBps\":%.4f,\"process_cpu_seconds\":%.6f,\"major_faults\":%ld,\"minor_faults\":%ld,\"voluntary_context_switches\":%ld,\"involuntary_context_switches\":%ld,\"checksum\":%.12g}\n",
        hot ? "hot" : "stream", n, repeats, unique_bytes/1048576., bytes/1048576., total/(n*repeats), times[times.size()/2], bytes*repeats/total/1e6,
        cpu_seconds(after)-cpu_seconds(before), after.ru_majflt-before.ru_majflt, after.ru_minflt-before.ru_minflt,
        after.ru_nvcsw-before.ru_nvcsw, after.ru_nivcsw-before.ru_nivcsw, checksum);
    ggml_free(init); ggml_free(metadata); gguf_free(gguf); munmap(mapped, status.st_size); close(fd);
    return 0;
} catch (const std::exception & e) { std::fprintf(stderr, "[dsv4f-cpu-profile] %s\n", e.what()); return 1; }
