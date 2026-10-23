This document describes the optimizations used in this project to run DeepSeek V4.1 Flash with large system RAM and roughly 24 GiB of VRAM. Since all expert weights cannot remain on the GPU, the engine caches frequently used experts in VRAM and overlaps CPU computation, GPU computation, weight transfers, and Engram I/O.

The defaults below apply when launching through `run-server.sh`. Some defaults differ when running the binary directly. The primary validation platform is a Ryzen 9900X with an RX 7900 XTX using ROCm/HIP. In particular, the V4.1 graph overlap and asynchronous Engram paths have ROCm-specific conditions. CUDA support and actual performance validation are separate considerations.

For performance comparisons, **disable MTP with `DSV4F_MTP=0`**. Enable it only when testing or optimizing MTP itself. Use `Explain story of the romance of three kingdoms.` for short inputs and the contents of `PROMPT_TEST.md` for long inputs. The descriptions below explain the implementation; they do not claim individual speedups for each feature.

# MoE Expert Cache

## Caching complete experts on the GPU

The original expert weights remain in system RAM. Each VRAM slot holds a selected expert's complete `gate`, `up`, and `down` weight set. During decode, a GPU routing kernel checks a `(layer, expert)` lookup table to divide work between experts that can run immediately on the GPU and experts that must run on the CPU. V4.1 mixed-quantization models use separate services and caches for `Q3_K`, `Q2_0`, and `PTQ1_0`.

A cache miss does not make the current token wait for a weight transfer. The CPU computes that expert using its RAM weights, while experts selected by the cache policy are uploaded asynchronously by a separate loader thread and copy stream. Uploads therefore serve future requests; the current request completes using the hit/miss plan established during routing. Even if no cache slots fit in VRAM, all routed experts can run on the CPU.

Each slot tracks GPU readers and exclusive loader ownership. The loader cannot overwrite a slot in use. It invalidates the old lookup entry and publishes the new entry only after all three weight transfers finish. The GPU releases its reader references after consuming all down-projection results. This allows computation and cache replacement to overlap without exposing partially transferred weights.

## Adapting replacement to recent usage

The default `aged` policy counts expert selections from actual routing and periodically halves the scores. This reduces the tendency for experts popular in an earlier prompt to occupy the cache indefinitely. Entries with zero routing weight in a format-specific route are excluded from usage counts, and resident or pending experts are not uploaded again.

The admission candidate is the highest-scoring expert among the current misses. The replacement victim is the lowest-scoring expert among eligible slots. Unless the victim slot is empty, the candidate's score must exceed the resident expert's score by a configurable factor. This margin reduces repeated transfers between experts with similar usage frequencies.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DSV4F_EXPERT_CACHE_MIB` | `14336` | Aggregate expert-cache ceiling across all formats: 14 GiB |
| `DSV4F_V41_CACHE_Q3_MIB` | `8192` | Q3_K cache ceiling |
| `DSV4F_V41_CACHE_Q2_MIB` | `1024` | Q2_0 cache ceiling |
| `DSV4F_V41_CACHE_PTQ_MIB` | `5120` | PTQ1_0 cache ceiling |
| `DSV4F_CACHE_POLICY` | `aged` | Selects `legacy`, `scored`, or `aged` |
| `DSV4F_CACHE_DECAY_TOKENS` | `256` | Decay interval parameter; each service decays scores every `43 × configured value` routing requests |
| `DSV4F_CACHE_REPLACE_RATIO` | `1.25` | Replacement score multiplier for the aged policy |

The default format ceilings divide the aggregate budget in an **8:1:5** ratio. Changing the total budget also changes format ceilings that have not been explicitly overridden. These values are ceilings; actual capacity depends on available VRAM and workspace requirements. A format ceiling can also be set to `0`. The launcher sets `GGML_CUDA_MOE_CACHE=0` to avoid enabling the generic llama.cpp MoE cache alongside this implementation.

# Dynamic VRAM allocation for prefill / decode switching

## Redistributing workspace and cache capacity by phase

Prefill processes many tokens at once and needs large activations, matrix-multiplication buffers, and expert staging space. Decode processes only a few tokens, allowing smaller workspaces and more VRAM for the expert cache. `DSV4F_PHASE_WORKSPACE=1` switches graph reservation sizes between these phases.

The launcher defaults to a prefill batch and ubatch of 4096 each. Decode lowers the batch ceiling to 512 and the ubatch ceiling to 256, but **actual decode graph workspace reservation is limited to four tokens**. Ordinary token decode with at most four input tokens and small inputs in an MTP context use this path. Larger inputs return to the prefill workspace. Embedding inputs in an ordinary context have an additional eligibility condition.

A phase transition synchronizes outstanding computation, releases the previous graphs and scheduler, trims backend temporary-memory pools, and reserves the graph needed for the new phase. V4.1 manages its cache as reclaimable blocks, so returning to prefill does not require evicting every expert first. The allocator reclaims enough capacity for the new workspace and then grows the cache into the remaining space.

Cache blocks target roughly 64 MiB, adjusted to expert size and capped at 64 slots per block. Before shrinking, useful experts with higher scores are moved into the retained region through GPU copies, and the lookup table is rebuilt. This reduces the cost of discarding a cache populated during decode when the next prefill begins.

## Accounting for physical VRAM and library allocations

Cache growth considers the aggregate and format ceilings, current free VRAM, headroom, and guard space. On HIP/Linux, the engine checks both the runtime's free-memory value and AMDGPU's physical VRAM usage, using the more conservative value. It also handles delays between the two counters after memory is freed.

When an engine workspace allocation needs more room, expert-cache blocks can be reclaimed. Headroom provides spare capacity for ordinary allocations, while guard space covers library workspaces and GPU code loading that bypass the engine allocator. Prefill cache slots in use cannot be reclaimed before their projections finish.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DSV4F_PHASE_WORKSPACE` | `1` | Enables workspace switching by phase |
| `BATCH_SIZE` / `UBATCH_SIZE` | `4096` / `4096` | Prefill batch and ubatch sizes |
| `DSV4F_DECODE_BATCH` / `DSV4F_DECODE_UBATCH` | `512` / `256` | Decode batch and ubatch ceilings |
| `DSV4F_VRAM_HEADROOM_MIB` | `256` | VRAM headroom; the implementation enforces a minimum of 128 MiB |
| `DSV4F_VRAM_GUARD_MIB` | `256` | Additional allowance for library and similar allocations |

## Streaming expert weights during prefill

Prefill groups routed token rows by expert for GPU matrix multiplication. Rather than uploading every required expert at once, it processes groups of 16 experts by default using two alternating staging slots. The copy stream uploads a subsequent group while the compute stream processes a ready group. Events control when each staging slot can be reused.

This path is configured through `DSV4F_PREFILL_STREAM=1` and `DSV4F_PREFILL_GROUP=16`. PTQ1_0 weights are converted on the GPU to a Q2_0 staging representation for the MMQ path. Staging and conversion scratch are graph operands, so their reservations are included when budgeting VRAM alongside the expert cache. The V4.1 graph reuses scratch across layers.

The optional `DSV4F_PREFILL_CACHE=1` path reuses resident experts and borrows cache slots. It defaults to `0`. When enabled, it is attempted only for inputs meeting `DSV4F_PREFILL_CACHE_MIN_TOKENS` (default 4096). Borrowed slots are published back to the decode cache after the layer finishes computing.

Implementation: [phase-specific graph reservation](vendor/llama.cpp/src/llama-context.cpp), [VRAM budgeting](vendor/llama.cpp/ggml/src/ggml-cuda/dsv4f-memory.cuh), [cache reclamation and growth](vendor/llama.cpp/ggml/src/ggml-cuda/dsv4f-service.cu), [prefill streaming](vendor/llama.cpp/ggml/src/ggml-cuda/dsv4f-prefill.cu), [scratch layout](vendor/llama.cpp/ggml/src/ggml-dsv4f-prefill.h).

# Parallel computation on cpu offloading

## Computing cache hits and misses concurrently on CPU and GPU

During decode, expert computation within a layer is divided according to routing results. The GPU routing kernel writes the hit mask, expert IDs, weights, and required input activations into a request ring in mapped pinned host memory. A persistent service thread dispatches only misses to the CPU worker pool while the GPU continues computing cache hits. This reduces waits caused by synchronously reading routing results back to the host for every layer.

The CPU computes each missed expert through the complete `gate → SwiGLU with up → down → routing weight` sequence. Intermediate expert activations stay on the CPU, and the final contributions are combined with GPU results. With the default `DSV4F_GPU_WAIT=callback`, a host callback on the GPU stream waits for CPU completion before a join kernel adds the CPU result to the GPU result. The comparison mode, `poll`, has the GPU poll for completion.

For paths with at most four tokens, V4.1's `DSV4F_V41_MOE_OVERLAP=1` and `DSV4F_V41_MOE_PIPELINE=1` arrange the graph to submit format-specific MoE work first, execute other formats' GPU work and the shared expert, and then join CPU results. On ROCm, `DSV4F_V41_ROUTE_FIRST=1` prepares format-specific routing requests first so CPU work can begin earlier. Accumulation preserves the existing shared-expert and tier order.

The format-specific services share one CPU worker pool, so CPU expert computation is serialized through one lane. Each CPU job overlaps other GPU work; CPU computation from multiple formats does not run concurrently without bounds.

## Persistent workers and quantized computation

CPU workers persist across layers. Output rows from missed experts are split into chunks, allowing multiple cores to contribute even when there are few misses. The implementation synchronizes between gate/up and down computation and reuses quantized activations and temporary result buffers across layers. When gate and up use the same activation quantization format, they share the converted input.

The defaults are 12 workers and dynamic scheduling in chunks of 32 rows. Workers briefly spin for the next job before sleeping on a condition variable. An atomic epoch and completion counter reduce synchronization overhead for repeated jobs. Physical-core affinity and static scheduling are optional.

V4.1 CPU dot products select AVX2 or AVX-512 paths according to CPU support. PTQ has an F16C path for FP16 scale conversion, and the Q3_K AVX-512 path can use an unsigned-offset VNNI kernel. These kernels compute directly with the quantized formats instead of repeatedly expanding weights into large FP32 matrices.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DSV4F_GPU_SERVICE` | `1` | Enables GPU routing and the persistent CPU service |
| `DSV4F_GPU_WAIT` | `callback` | CPU completion wait mode: `callback` or `poll` |
| `DSV4F_SKIP_NO_MISS` | `1` | Omit CPU answer/completion work for all-hit or zero-weight routes; keep ordered cache observations. Captured callback nodes remain and return immediately when the batch has no CPU job. Set `0` for the preceding path. |
| `DSV4F_CPU_THREADS` | `12` | CPU computation parallelism, including the calling thread |
| `DSV4F_CPU_SCHEDULE` / `DSV4F_CPU_ROW_CHUNK` | `dynamic` / `32` | Row scheduling policy and chunk size |
| `DSV4F_CPU_SPIN_US` / `DSV4F_CPU_FAST_SYNC` | `300` / `1` | Worker wait and synchronization settings |
| `DSV4F_CPU_AFFINITY` | `none` | Set to `physical` to select physical-core pinning |
| `DSV4F_CPU_PTQ_ISA` / `DSV4F_CPU_QUANT_ISA` | `auto` / `auto` | Selects supported CPU computation paths |
| `DSV4F_CPU_PTQ_KERNEL` / `DSV4F_CPU_Q3_K_KERNEL` | `auto` / `auto` | Selects PTQ and Q3_K kernel variants |
| `DSV4F_PINNED_WEIGHTS` | `1` | Uses pinned host weight buffers to reduce upload costs for large prefills |

Even with a higher cache hit rate, total latency depends on the slower CPU/GPU branch and the remaining join wait. Compare actual prefill/decode throughput alongside hit rates, CPU time, and admission traffic in the `[dsv4f-service]` logs.

Implementation: [CPU workers, expert computation, and submit/join graph operations](src/moe.cpp), [GPU request ring and join](vendor/llama.cpp/ggml/src/ggml-cuda/dsv4f-service.cu), [routing plans](src/route.cpp), [V4.1 graph scheduling](vendor/llama.cpp/src/models/deepseek41.cpp), [CPU ISA selection](vendor/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c).

# Async engram loading

## Reading and converting only the required FP8 rows

Engram embeddings remain in their original FP8 `.safetensors` shards. The reader derives row IDs from token n-gram history and uses `pread` to fetch the required weights and scales. Weights use E4M3, with one E8M0 scale per 32 elements. The selected rows are converted to FP32 and passed to the model.

Repeated rows are served by a per-table four-way set-associative cache in RAM. It stores **FP8 weights and scales**, and evicts the least recently used entry within each set. The default total budget is 32 MiB, divided among the tables. Set `DSV4F_ENGRAM_CACHE_MIB=0` to disable it.

The default `lut` conversion uses lookup tables for the 256 FP8 values and ordinary scale encodings, reducing repeated per-element conversion and `ldexp` calls. Exceptional scale encodings retain the original operation. The optional `DSV4F_ENGRAM_DEDUP=1` path copies existing FP32 results for duplicate rows within a batch; it defaults to `0`.

## Overlapping I/O with GPU computation within a batch

The synchronous path reads and uploads rows for each Engram table before graph execution. In the asynchronous path, a dedicated reader thread processes tables in order, performs FP8 conversion, and publishes each completed table into a pinned host buffer. A layer can consume its rows as soon as that table is ready.

The ROCm graph includes a custom node anchored to the layer's incoming hidden state. When the GPU stream reaches that layer, a host callback waits for the corresponding table to become ready, followed by an asynchronous H2D copy. The reader prepares later Engram data while earlier GPU layers compute. If I/O takes longer than computation, the stream still waits at the point of consumption.

The callback only waits for readiness and does not call HIP APIs. The actual copy is submitted to the stream outside the callback. The reader inherits the caller's floating-point environment to preserve conversion behavior relative to the synchronous path.

## Prefetching the next prefill batch

Prefill can use known prompt tokens to read Engram rows **one batch ahead**. After preparing the current batch's tables, it queues the next batch so its I/O can run while the current graph executes. This uses known prompt input rather than predicting future decode tokens.

Prefetch results are reused only when token history, position, sequence, batch size, and floating-point environment match the actual execution. The preceding n-gram in the prompt hint is also checked against actual KV-derived history. Changed or mismatched inputs cancel the old work and trigger reads for the current input. Hints do not modify KV state. Next-batch prefetch applies when a prompt hint is available and batches contain more than four tokens.

| Setting | Default | Purpose |
| --- | --- | --- |
| `DSV4F_ENGRAM_OVERLAP` | `1` | Asynchronous reads within the current batch for paths with at most four tokens |
| `DSV4F_ENGRAM_PREFILL_OVERLAP` | `1` | Asynchronous reads within the current prefill batch |
| `DSV4F_ENGRAM_PREFETCH` | `1` | Prefetches the next batch of a known prompt |
| `DSV4F_ENGRAM_DECODE` | `lut` | FP8 conversion method: `baseline` or `lut` |
| `DSV4F_ENGRAM_CACHE_MIB` | `32` | Total RAM budget for the FP8 row cache |
| `DSV4F_ENGRAM_DEDUP` | `0` | Reuses results for duplicate rows within a batch |

Both `OVERLAP` settings accept `0` for the synchronous path, `1` for asynchronous reads at all Engram layers, or `2` for asynchronous reads excluding the first Engram table. The V4.1 graph connects next-batch prefetch only when all tables use the asynchronous path.

`DSV4F_ENGRAM_PROFILE=1` records row reads, cache hits, FP8 conversion, and related timings. `DSV4F_ENGRAM_ASYNC_PROFILE=1` records asynchronous read/wait times and prefetch reuse. Account for warm versus cold row caches and OS page-cache versus SSD I/O behavior when checking whether GPU waits actually decrease.

Implementation: [row reads and cache](src/engram.cpp), [FP8 LUT](src/engram-fp8.h), [asynchronous reader for the current batch](src/engram-async.cpp), [next-batch prefetch](src/engram-prefetch.cpp), [GPU stream transfers](vendor/llama.cpp/ggml/src/ggml-cuda/dsv4f-service.cu).
