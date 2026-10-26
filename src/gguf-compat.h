#pragma once
struct gguf_context;
// Add engine aliases to in-memory metadata; never modify a GGUF file/payload.
void dsv4f_native_metadata(gguf_context * ctx);
