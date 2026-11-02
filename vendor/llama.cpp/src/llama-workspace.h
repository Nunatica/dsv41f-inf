#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

// Size a prefill workspace in 1024-token buckets, bounded by the configured
// physical microbatch. Clamp before rounding and use 64 bits to avoid overflow.
inline uint32_t llama_prefill_workspace_tokens(size_t n_tokens, uint32_t max_ubatch, size_t n_remaining = 0) {
    const uint64_t tokens = std::min<uint64_t>(std::max(n_tokens, n_remaining), max_ubatch);
    return uint32_t(std::min<uint64_t>(max_ubatch, ((tokens + 1023) / 1024) * 1024));
}
