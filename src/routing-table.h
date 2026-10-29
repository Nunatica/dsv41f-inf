#pragma once
#include <string>
#include <cstdint>
#include <utility>
#include <vector>

// Initialize before loading weights. start() seeds initialized GPU caches from
// loaded history at the quiescent startup boundary, then enables counting.
// Call only with the previous model and its services unloaded.
void dsv4f_routing_init(const std::string & model_path);
void dsv4f_routing_start();
// Drains decode telemetry and atomically replaces models/<model stem>.routing.json.
// Errors are logged; a completed response must still reach the client.
bool dsv4f_routing_save();
// Empty when disabled; otherwise 0 / -infinity in global expert ID order.
// Each group pairs its gate tensor name with a global-to-local expert map.
std::vector<float> dsv4f_pruning_mask(int layer, int n_experts, int n_used,
    const std::vector<std::pair<std::string, std::vector<int32_t>>> & groups);
