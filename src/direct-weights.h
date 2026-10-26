#pragma once
#include "ggml-backend.h"
#include <cstdint>
#include <cstddef>
#include <vector>

// Zero/unset keeps the original resident loader. Positive budgets enable Linux direct I/O.
uint64_t dsv4f_ram_budget();
bool dsv4f_direct_weights_enabled();
struct dsv4f_direct_tensor {
    ggml_tensor * tensor;
    int fd;
    uint64_t offset;
};
ggml_backend_buffer_t dsv4f_direct_weights_buffer(const std::vector<dsv4f_direct_tensor> & tensors);
// Aligned O_DIRECT payload read, including unaligned tensor/row offsets and the last file sector.
void dsv4f_direct_read(int fd, void * output, size_t bytes, uint64_t offset);
