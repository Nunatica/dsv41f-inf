# dsv41f-inf

Highly experimental inference engine for DeepSeek V4.1 Flash.

## Target system

9900x (or any zen5 cpus such as 9800x3d) + 192GiB ddr5 5600mhz + 7900xtx (or nvidia gpus such 3090, 4090 which has at least 24gb vram, not tested), Also you need decent SSD for gguf and engram weights.

## Dependencies and setup

The launcher requires Linux (`/proc` and cgroup v2 are used for memory/swap checks), Bash, Python 3, and a HIP or CUDA GPU backend. Run the commands below from the repository root.

### Common build tools

- A C/C++ compiler with C++17 support (GCC/G++ is used on the target system).
- CMake **3.24 or newer** and Ninja.
- OpenMP headers/runtime for the CPU backend (provided by GCC/G++ on Ubuntu/Debian; Clang builds need the matching OpenMP development package).
- Python 3 for model inspection and the server launcher. These scripts use only the Python standard library; no `pip install` or virtual environment is required for serving.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build python3
cmake --version  # Must be 3.24 or newer; upgrade if your distro ships an older version.
ninja --version
python3 --version
```

The llama.cpp source is included under `vendor/llama.cpp`. The provided build disables OpenSSL and uses the included `web/` assets, so no OpenSSL development package, Node.js, or npm install is needed for serving.

### AMD GPU (ROCm / HIP)

Install an AMD GPU driver compatible with your GPU and ROCm release, plus **ROCm/HIP 6.1 or newer**. The build requires the HIP compiler/device libraries and the **hipBLAS and rocBLAS development libraries**, including their CMake package files. HIP was tested on RX 7900 XTX with ROCm 7.14.1.

Configure AMD's package repository for your distribution and install the driver and HIP SDK. On Ubuntu/Debian, after configuring the repository:

```bash
sudo apt install rocm-hip-sdk
```

For the default `/opt/rocm` installation, expose the tools and check that ROCm can see your GPU:

```bash
export ROCM_PATH=/opt/rocm
export PATH="$ROCM_PATH/bin:$PATH"
hipcc --version
rocminfo
BACKEND=hip GPU_ARCH=gfx1100 ./build-engine.sh
```

`gfx1100` is the default target for the RX 7900 XTX. For another GPU, set `GPU_ARCH` to its ROCm architecture. If ROCm is installed elsewhere, set `ROCM_PATH`, add its `bin` directory to `PATH`, and set `HIP_COMPILER` to its `llvm/bin/clang++` path. Ensure your user has access to `/dev/kfd` and the GPU render devices if `rocminfo` cannot access the GPU.

### NVIDIA GPU (CUDA, untested)

Install a compatible NVIDIA driver and the full CUDA Toolkit, including **nvcc, CUDA runtime development files, and cuBLAS**. A driver-only installation does not provide the compiler or development libraries. Use a host GCC/G++ version supported by your CUDA Toolkit. NVIDIA execution has not been tested on this host.

For a toolkit installed at `/usr/local/cuda`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvidia-smi
nvcc --version
BACKEND=cuda GPU_ARCH=native ./build-engine.sh
```

The CUDA build defaults to `GPU_ARCH=native`, so the GPU should be visible during compilation. Set `GPU_ARCH` explicitly when needed. `build-engine.sh` otherwise selects HIP first if `hipcc` is on `PATH`, then CUDA if `nvcc` is available; use `BACKEND` to choose when both are installed. Use separate `BUILD_DIR` values when building both backends (and pass the same `BUILD_DIR` to `run-server.sh`).

### Model files and runtime requirements

Get the model from [icecubetr/DeepSeek-V4.1-Flash-1017-dsx-2.6bit](https://huggingface.co/icecubetr/DeepSeek-V4.1-Flash-1017-dsx-2.6bit) and place the GGUF and its original FP8 Engram `.safetensors` shards in `models/`. Keep the included `models/dsv41f.jinja` chat template in place.
Alternately, you can use high quality model near original from [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF). In this case, do not download 8th shard (DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00008-of-00009.gguf), because it is engram shard.

The launcher defaults to `MODEL=models/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf` and `DSV4F_ENGRAM_DIR=models/`, both resolved relative to the repository root. Format selection requires an exact first filename: `DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf` or `DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf`. Set `MODEL` and `DSV4F_ENGRAM_DIR` before launching if your files are elsewhere, keeping these filenames. For UD-Q4_K_XL, supply shards 1–7 and 9; the missing Engram shard 8 is replaced by the original FP8 Engram `.safetensors` in `DSV4F_ENGRAM_DIR`. Missing other shards or incompatible original Engram tables are rejected. Use a fast SSD with enough space for both the GGUF and Engram weights; Engram rows are read during inference.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `DSV4F_RAM_BUDGET_MIB` | `184320` | `0`: original resident loader (Engram excluded); positive: total process-tree RAM ceiling. |
| `DSV4F_ADAPTIVE_ROUTING` | `0` (launcher) | `1`: persist decode expert counters and use history for RAM/VRAM preload and eviction. Mutually exclusive with pruning. The server binary defaults to `1` when unset. |
| `DSV4F_ADAPTIVE_PRUNING` | `0` | `0`: off; `1..50`: target percentage of routed expert weight bytes to exclude, with per-layer selections based on historical usage per byte. Mutually exclusive with routing. |
| `DSV4F_RAM_PRELOAD` | `0` | `1`: preload experts at startup when the RAM budget is positive; `0`: disabled. |
| `DSV4F_LOAD_PIPELINE_MIB` | `1024` | Overlap host allocation with disk reads for resident loading and RAM preload. `0`: sequential loading; range `0..65536` MiB. |
| `DSV4F_RAM_RESERVE_MIB` | `1024` | Headroom left outside the expert cache for runtime growth. |
| `DSV4F_RAM_CACHE_MIB` | automatic | Optional additional ceiling for the expert RAM cache. |
| `DSV4F_EXPERT_CACHE_MIB` | `14336` | Existing aggregate VRAM expert-cache ceiling. |

```bash
./run-server.sh                          # Tiered mode, 180 GiB RAM ceiling
DSV4F_RAM_BUDGET_MIB=0 ./run-server.sh     # Resident model weights; Engram stays on SSD
DSV4F_RAM_BUDGET_MIB=131072 ./run-server.sh # Tiered mode, 128 GiB RAM ceiling
```
Resident loading (`DSV4F_RAM_BUDGET_MIB=0`) allocates host weights in batches and reads the previous batch while allocating the next. GPU uploads finish before this host pipeline starts. Each tensor remains contiguous; a tensor larger than the configured batch size is loaded as one batch. Pruned resident expert planes remain omitted.

With `DSV4F_RAM_PRELOAD=1`, the same batch setting controls a two-stage preload pipeline: allocate/populate the next cache batch while the previous batch reads with O_DIRECT. Both resident and pending batches count against the RAM cache budget. Preload preserves historical priority/physical tie order and stops at the first plane that does not fit, without evicting the loaded prefix. Setting `DSV4F_LOAD_PIPELINE_MIB=0` restores sequential behavior for both paths.

Optionally, you can get MTP drafter from [here](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF/tree/main/dspark). Place `dspark-xs2.bin` and `dspark-xs2.json` in `models/`, then run:

```bash
DSV4F_MTP=1 ./run-server.sh
```

`MTP_MODEL` overrides the binary path; `MTP_CONFIG` overrides the JSON manifest
(default: the same basename with `.json`). The engine loads text MTP tensors
only and excludes vision, aligner, image, and vision router biases. The target
embedding/output head are shared. The additional weights use about 4.09 GiB
of VRAM, reducing the space available for the target expert cache.
MTP defaults to off (`DSV4F_MTP=0`); keep it off for benchmarks unrelated to MTP.

`DSV4F_DRAFT_N_MAX=1..5` limits proposals (default 1), and
`DSV4F_DRAFT_P_MIN=0..1` truncates them at the confidence threshold (default 0.25).

MTP is disabled by default because it does not improve any speed for target system.

### Adaptive expert pruning

This model is too large to load entirely into RAM on many systems. The high-quality Unsloth Q4 model requires approximately 277.5 GiB for its weights alone, excluding Engram, exceeding even this project's 192 GiB target system. Adaptive expert pruning helps address this by recording expert usage on representative workloads and excluding the least-used experts, substantially reducing resident memory requirements at a potential cost in output quality.

To use this feature, first collect representative decode history with `DSV4F_ADAPTIVE_ROUTING=1 DSV4F_ADAPTIVE_PRUNING=0`. Then restart with pruning enabled, for example:

```bash
DSV4F_ADAPTIVE_ROUTING=0 DSV4F_ADAPTIVE_PRUNING=30 DSV4F_RAM_BUDGET_MIB=0 MODEL=models/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf ./run-server.sh
DSV4F_ADAPTIVE_ROUTING=0 DSV4F_ADAPTIVE_PRUNING=25 DSV4F_RAM_BUDGET_MIB=184320 MODEL=models/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf ./run-server.sh
```

`DSV4F_ADAPTIVE_PRUNING=N` accepts integers from `0` to `50`; `0` disables pruning. **N is a target percentage of the main model's total routed expert weight bytes to remove**, rather than a fixed percentage of experts in every layer. For example, `25` targets a 25% reduction in the combined gate/up/down weights across all main-model layers and quantization tiers. Shared experts, attention, Engram, KV caches, compute buffers and MTP drafter weights are excluded from this target. This replaces the earlier uniform per-layer expert-count interpretation; the same setting can now select a different set of experts.

At startup, pruning reads `models/<model stem>.routing.json`, maps format-local history IDs to global expert IDs, and computes each candidate's cost as:

```text
cost = (expert historical calls / total historical calls in its layer)
       / (expert gate + up + down weight bytes)
```

Candidates with lower cost are excluded first across all layers until the memory target is reached. Normalizing within each layer accounts for differing history sample sizes, while dividing by exact weight bytes accounts for mixed quantization formats. Unobserved experts have zero cost. Equal costs are resolved by smaller layer index, then smaller global expert ID. Whole experts are removed, so savings may exceed the target by less than the size of the last excluded expert. Every layer retains at least its original top-k count. A candidate is skipped if removing it would leave too little removable byte capacity to reach the target while retaining top-k; startup fails if no feasible target exists under that constraint. A layer may exclude more than N% of its experts, including more than 50%, while another layer excludes much less. Layer 0 has no hardcoded exemption: its usage history determines how much it is pruned.

The mask applies before top-k selection in both prefill and decode: pruned top-1/top-2/etc. candidates are skipped, the original top-k count is filled from active experts, and weights are normalized as configured by the model. The MTP drafter's own experts are neither recorded nor pruned; main-model verification of drafted tokens uses the pruned main model.

The history file is read-only during pruning, and changing it requires a server restart. Missing, malformed, mismatched or incomplete history, invalid percentages, and simultaneous routing/pruning cause startup failure. With `DSV4F_RAM_BUDGET_MIB=0` and `--load-mode none`, resident weight tensors (including pinned buffers) contain only active expert planes: excluded planes are neither allocated nor read. Global routing IDs and history IDs stay unchanged; format-local IDs are packed for execution, and empty quantization tiers are omitted. The original GGUF stays unchanged. Tiered O_DIRECT and mmap loading retain their file layout and apply the routing mask; N describes the excluded weight set, not a guaranteed reduction in process RSS or a positive RAM-cache budget. The launcher uses the same selection rule for resident memory estimates. Startup logs report per-layer exclusions, historical call shares, and target/actual skipped bytes.

For the differential policy, use **`DSV4F_ADAPTIVE_PRUNING=30` for dsx 2.6 bpw** and **`25` for UD-Q4_K_XL** as history-based starting points. If you want one setting for both models, start with **`25`**. The earlier recommendation to treat every value of `30` or higher as experimental was based on uniform per-layer pruning and no longer describes the differential selection rule.

| Model | More conservative | Recommended starting point | Memory-priority candidate | Experimental range |
| --- | ---: | ---: | ---: | ---: |
| dsx 2.6 bpw | `25` | **`30`** | `35` | `40`–`50` |
| UD-Q4_K_XL | `20` | **`25`** | `30` | `35`–`50` |

These ranges are judgments based on the currently collected histories, not measured quality thresholds. Use the memory-priority candidate when the additional savings matter, and compare output quality with pruning disabled before adopting it. Different workloads or more complete histories can change the appropriate setting; the difference between the two models' recommendations does not establish that one model inherently tolerates pruning better.

Re-analysis of the locally collected routing histories on 2026-10-30 using the differential byte-target policy produced the following results. Both models have 40 layers and 384 routed experts per layer. The historical call share is the percentage of recorded expert selections belonging to the excluded set; the maximum per-layer share measures the same percentage within the most affected layer. These values replace the earlier uniform-pruning analysis.

| Model | Memory target | Historical call share excluded | Maximum per-layer share excluded | Resident expert memory saved |
| --- | ---: | ---: | ---: | ---: |
| dsx 2.6 bpw | 20% | 0.005% | 0.032% | 33.07 GiB |
| dsx 2.6 bpw | 25% | 0.047% | 0.091% | 41.33 GiB |
| dsx 2.6 bpw | 30% | 0.129% | 0.198% | 49.60 GiB |
| dsx 2.6 bpw | 35% | 0.275% | 0.395% | 57.87 GiB |
| dsx 2.6 bpw | 40% | 0.517% | 0.765% | 66.14 GiB |
| UD-Q4_K_XL | 20% | 0.145% | 0.283% | 53.79 GiB |
| UD-Q4_K_XL | 25% | 0.304% | 0.502% | 67.24 GiB |
| UD-Q4_K_XL | 30% | 0.543% | 0.813% | 80.68 GiB |
| UD-Q4_K_XL | 35% | 0.884% | 1.258% | 94.13 GiB |
| UD-Q4_K_XL | 40% | 1.352% | 1.847% | 107.58 GiB |

For **dsx 2.6 bpw**, `30` saves 49.60 GiB while excluding 0.129% of historical selections overall and at most 0.198% within any layer. Both call shares are below those of the earlier uniform `25` policy (0.162% overall and 1.598% in its most affected layer), despite saving more memory. Moving from `30` to `35` saves another 8.27 GiB, but the overall excluded share more than doubles to 0.275%. At `40`, it rises to 0.517%, which is why `30` is the balanced starting point and `35` is the memory-priority candidate.

For **UD-Q4_K_XL**, `25` saves 67.24 GiB with a 0.304% overall excluded share and a 0.502% maximum per-layer share. Moving to `30` adds 13.45 GiB of savings, with those shares increasing to 0.543% and 0.813%. At `35`, the maximum per-layer share exceeds 1% (1.258%). This supports starting at `25`, considering `30` for memory savings, and reserving `35` or higher for explicit quality comparisons. The 1% figure is a history-coverage heuristic, not a proven quality boundary.

At `25`, the 2.6 bpw history selects between 20 and 183 excluded experts per layer; layer 0 loses 20 experts rather than the previous 96. Compared with uniform 25% pruning, the new policy saves slightly more memory (41.33 vs. 40.82 GiB) while reducing the excluded overall historical call share from 0.162% to 0.047%. The selection rule is a greedy usage-per-byte heuristic, not a measured estimate of layer sensitivity or a guarantee of optimal output quality.

The memory savings above apply to resident expert weight tensors only. A positive RAM budget remains a configured ceiling. For UD-Q4_K_XL on the 192 GiB target system, `25` leaves 201.71 GiB of routed expert weights and `30` still leaves 188.26 GiB before other weights and runtime buffers, so use budgeted loading as shown above rather than assuming the recommended values allow full resident loading.

The existing 2.6 bpw resident/pinned smoke test used `25` with MTP disabled and completed 64-token generations for both the short prompt and `PROMPT_TEST.md`. The other settings above were evaluated against the saved routing histories; these calculations and the smoke test do not establish output-quality parity with pruning disabled.

These call shares are **not quality-loss percentages or percentages of affected tokens**: pruned selections are replaced by other experts, and each token passes through many layers. The histories do not record selection rank, routing weight, prompt domain or quality labels; pruning can also change later hidden states and routing decisions. Rare or unobserved experts may matter on unseen tasks. Collect representative history for your workload and validate quality before increasing pruning; the recommendation is based on historical usage coverage, not a measured quality guarantee.

## Build and run

After installing the dependencies and placing the model files:

```bash
./build-engine.sh
./run-server.sh
```

## Performance

| Implementation | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: |
| dsv41f-inf (2.6bpw, full weights on ram) | 226.65 | 20.87 |
| dsv41f-inf (2.6bpw, 128 GiB budget, cold test) | 74.03 | 14.39 |
| llama.cpp | - | - |

## Credits
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Xolotl](https://github.com/Indras-Mirror/Xolotl)
- [dsx](https://github.com/lxe/dsx)

## License
Project itself is under MIT License. included llama.cpp source follows its own license.

## Disclaimer
This project is not intended to use for serious tasks. Use it at your own risk.
