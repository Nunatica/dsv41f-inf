#include "llama-workspace.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    constexpr uint32_t limit = 10240;
    for (size_t tokens = 5; tokens <= 2 * limit; ++tokens) {
        const auto capacity = llama_prefill_workspace_tokens(tokens, limit);
        require(capacity >= std::min<size_t>(tokens, limit), "workspace cannot truncate input");
        require(capacity <= limit, "configured microbatch limit must be preserved");
        require(capacity % 1024 == 0, "workspace must use 1024-token buckets");
        require(capacity < tokens + 1024, "workspace must use the smallest fitting bucket");
    }

    // Alternating short and long turns, including exact bucket boundaries.
    const size_t inputs[] = {802, 1024, 1025, 1505, 10240, 16953, 43, 2048, 2049, 63};
    const uint32_t expected[] = {1024, 1024, 2048, 2048, 10240, 10240, 1024, 2048, 3072, 1024};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        require(llama_prefill_workspace_tokens(inputs[i], limit) == expected[i], "short/long turn capacity");
    }

    // Explicit ubatch settings can be smaller than a bucket or unaligned.
    require(llama_prefill_workspace_tokens(43, 256) == 256, "sub-bucket microbatch");
    require(llama_prefill_workspace_tokens(2049, 2500) == 2500, "unaligned microbatch ceiling");
    require(llama_prefill_workspace_tokens(3000, 2500) == 2500, "input spans multiple microbatches");
    require(llama_prefill_workspace_tokens(9, limit, 16953) == limit, "message boundary cannot shrink a long prompt's workspace");
    require(llama_prefill_workspace_tokens(9, limit, 1505) == 2048, "size from the uncached remainder");
    require(llama_prefill_workspace_tokens(1505, limit, 9) == 2048, "a stale small hint cannot truncate the current batch");
    const auto max = std::numeric_limits<uint32_t>::max();
    require(llama_prefill_workspace_tokens(max, max) == max, "rounding cannot overflow");
    require(llama_prefill_workspace_tokens(std::numeric_limits<size_t>::max(), limit) == limit, "large logical batch");
    std::puts("Prefill workspace sizing PASS");
}
