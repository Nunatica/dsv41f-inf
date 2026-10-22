#pragma once
#include "ggml.h"
// Gather format-local IDs and mask route probabilities in one operation.
// The returned views share one backing allocation; IDs store I32 bits.
void dsv4f_route_plan(ggml_context * ctx, ggml_tensor * mapping, ggml_tensor * ids,
                     ggml_tensor * probabilities, ggml_tensor ** local_ids,
                     ggml_tensor ** local_probabilities);
// Fused sqrt-softplus routing with bias-only selection and optional normalization.
void dsv4f_router(ggml_context * ctx, ggml_tensor * logits, ggml_tensor * bias,
                 int nk, bool normalize, float scale, ggml_tensor ** ids,
                 ggml_tensor ** probabilities);
ggml_tensor * dsv4f_moe(ggml_context * ctx, ggml_tensor * x, ggml_tensor * ids,
                       ggml_tensor * weights, ggml_tensor * gate, ggml_tensor * up,
                       ggml_tensor * down, float limit, ggml_tensor ** prefill_scratch = nullptr);
ggml_tensor * dsv4f_moe_join(ggml_context * ctx, ggml_tensor * routed, ggml_tensor * shared);
// Expand the routed GPU/CPU submission before unrelated work, deferring its
// completion until a subsequent dsv4f_moe_join. CPU fallback remains synchronous.
void dsv4f_moe_submit(ggml_cgraph * graph, ggml_tensor * routed);

// Publish this format's CPU requests before expanding routed GPU work.
// The returned plan is also attached as a live dependency of routed.
ggml_tensor * dsv4f_moe_prepare(ggml_context * ctx, ggml_tensor * routed);
