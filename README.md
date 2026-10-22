# dsv41f-inf

Highly experimental inference engine for DeepSeek V4.1 Flash.

## Target system

9900x (or any zen5 cpus such as 9800x3d) + 192GiB ddr5 5600mhz + 7900xtx (or nvidia gpus such 3090, 4090 which has at least 24gb vram, not tested), Also you need decent SSD for gguf and engram weights.

## Build and run

Linux, CMake >= 3.24, Ninja, a C++17 compiler, Python 3, and a ROCm/HIP or CUDA development toolkit are required. No Python packages are needed for serving.

```bash
./build-engine.sh
./run-server.sh
```

For CUDA use `BACKEND=cuda GPU_ARCH=native ./build-engine.sh`. HIP was tested on RX 7900 XTX with ROCm 7.14.1; NVIDIA execution has not been tested on this host.

Get models from [here](https://huggingface.co/icecubetr/DeepSeek-V4.1-Flash-1017-dsx-2.6bit) and place GGUF and its original FP8 Engram `.safetensors` shards in `models/`.

## Performance

| Implementation | Prefill (tok/s) | Decode (tok/s) |
| --- | ---: | ---: |
| dsv41f-inf | 200.65 | 19.72 |
| llama.cpp | - | - |

Data was measured at target system. test was done with same prompt with PROMPT_TEST.md

## Credits
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Xolotl](https://github.com/Indras-Mirror/Xolotl)
- [dsx](https://github.com/lxe/dsx)

## License
Project itself is under MIT License. included llama.cpp source follows its own license.

## Disclaimer
This project is not intended to use for serious tasks. Use it at your own risk.
