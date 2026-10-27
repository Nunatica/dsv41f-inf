#pragma once
#include <string>

// Initialize before loading weights; start counting after startup warmup.
// Call only with the previous model and its services unloaded.
void dsv4f_routing_init(const std::string & model_path);
void dsv4f_routing_start();
// Drains decode telemetry and atomically replaces models/<model stem>.routing.json.
// Errors are logged; a completed response must still reach the client.
bool dsv4f_routing_save();
