#pragma once
#include <cstdint>

// Local profiling artifact, little-endian/native IEEE-754. Followed by
// ids[nk] (int32), probabilities[nk] (float), activation[ne] (float).
// One selected format per file; all original routes/hit bits remain.
constexpr uint32_t DSV4F_TRACE_Q3 = 0x51335431; // Q3T1
constexpr uint32_t DSV4F_TRACE_PTQ = 0x50315431; // P1T1
struct dsv4f_cpu_trace_record {
    uint32_t magic = DSV4F_TRACE_Q3;
    uint32_t ne = 0, nf = 0, nk = 0;
    uint64_t hits = 0;
    float limit = 0;
    uint32_t reserved = 0;
    char names[3][64] = {};
    double phase_us[5] = {}; // prepare, gate/up workers, quantize, down workers, reduce
};
static_assert(sizeof(dsv4f_cpu_trace_record) == 264, "trace layout changed");
