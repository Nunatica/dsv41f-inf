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
| `DSV4F_ADAPTIVE_ROUTING` | `0` | `0`: off; `1`: persist decode expert counters and use history for RAM/VRAM preload and eviction. Mutually exclusive with pruning. |
| `DSV4F_ADAPTIVE_PRUNING` | `0` | `0`: off; `1..50`: target percentage of routed expert weight bytes to exclude, with per-layer selections based on historical usage per byte. Mutually exclusive with routing. |
| `DSV4F_RAM_PRELOAD` | `0` | `1`: preload experts at startup when the RAM budget is positive; `0`: disabled. |
| `DSV4F_EXPERT_CACHE_MIB` | `14336` | Existing aggregate VRAM expert-cache ceiling. |

```bash
./run-server.sh                          # Tiered mode, 180 GiB RAM ceiling
DSV4F_RAM_BUDGET_MIB=0 ./run-server.sh     # Resident model weights; Engram stays on SSD
DSV4F_RAM_BUDGET_MIB=131072 ./run-server.sh # Tiered mode, 128 GiB RAM ceiling
```
Resident loading (`DSV4F_RAM_BUDGET_MIB=0`) allocates host weights in batches and reads the previous batch while allocating the next. GPU uploads finish before this host pipeline starts. Each tensor remains contiguous; a tensor larger than the configured batch size is loaded as one batch. Pruned resident expert planes remain omitted.

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

Target model is too large to load entirely into RAM on many systems. The high-quality Unsloth Q4 model requires approximately 277.5 GiB for its weights alone, excluding Engram, exceeding even this project's 192 GiB target system. Adaptive expert pruning helps address this by recording expert usage on representative workloads and excluding the least-used experts, substantially reducing resident memory requirements at a potential cost in output quality.

See [Adaptive expert pruning](./ADAPTIVE_EXPERT_PRUNING.md) for setup, routing and loading behavior, memory savings, and history-based setting recommendations.

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
