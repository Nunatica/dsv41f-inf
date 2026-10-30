# Adaptive expert pruning

To use this feature, first collect representative decode history with `DSV4F_ADAPTIVE_ROUTING=1 DSV4F_ADAPTIVE_PRUNING=0`. Then restart with pruning enabled, for example:

```bash
DSV4F_ADAPTIVE_PRUNING=30 DSV4F_RAM_BUDGET_MIB=0 MODEL=models/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf ./run-server.sh
DSV4F_ADAPTIVE_PRUNING=20 DSV4F_RAM_BUDGET_MIB=184320 MODEL=models/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf ./run-server.sh
```

`DSV4F_ADAPTIVE_PRUNING=N` accepts integers from `0` to `50`; `0` disables pruning. **N is a target percentage of the main model's total routed expert weight bytes to remove**. For example, `25` targets a 25% reduction in the combined gate/up/down weights across all main-model layers and quantization tiers. Shared experts, attention, Engram, KV caches, compute buffers and MTP drafter weights are excluded from this target.

At startup, pruning reads `models/<model>.routing.json`, maps format-local history IDs to global expert IDs, and computes each candidate's cost as:

```text
cost = (expert historical calls / total historical calls in its layer)
       / (expert gate + up + down weight bytes)
```

Candidates with lower cost are excluded first across all layers until the memory target is reached. Normalizing within each layer accounts for differing history sample sizes, while dividing by exact weight bytes accounts for mixed quantization formats. Unobserved experts have zero cost. Equal costs are resolved by smaller layer index, then smaller global expert ID. Whole experts are removed, so savings may exceed the target by less than the size of the last excluded expert. Every layer retains at least its original top-k count. A candidate is skipped if removing it would leave too little removable byte capacity to reach the target while retaining top-k; startup fails if no feasible target exists under that constraint. A layer may exclude more than N% of its experts, including more than 50%, while another layer excludes much less.

The mask applies before top-k selection in both prefill and decode: pruned top-1/top-2/etc. candidates are skipped, the original top-k count is filled from active experts, and weights are normalized as configured by the model. The MTP drafter's own experts are neither recorded nor pruned; main-model verification of drafted tokens uses the pruned main model.

The history file is read-only during pruning, and changing it requires a server restart. Missing, malformed, mismatched or incomplete history, invalid percentages, and simultaneous routing/pruning cause startup failure. With `DSV4F_RAM_BUDGET_MIB=0` and `--load-mode none`, resident weight tensors (including pinned buffers) contain only active expert planes: excluded planes are neither allocated nor read. Global routing IDs and history IDs stay unchanged; format-local IDs are packed for execution, and empty quantization tiers are omitted. The original GGUF stays unchanged. Tiered O_DIRECT and mmap loading retain their file layout and apply the routing mask; N describes the excluded weight set, not a guaranteed reduction in process RSS or a positive RAM-cache budget. The launcher uses the same selection rule for resident memory estimates. Startup logs report per-layer exclusions, historical call shares, and target/actual skipped bytes.

For the differential policy, use **`DSV4F_ADAPTIVE_PRUNING=30` for dsx 2.6 bpw** and **`20` for Unsloth UD-Q4_K_XL** as history-based starting points. If you want one setting for both models, start with **`20`**.

| Model | More conservative | Recommended starting point | Memory-priority candidate | Experimental range |
| --- | ---: | ---: | ---: | ---: |
| dsx 2.6 bpw | `25` | **`30`** | `35` | `40`–`50` |
| UD-Q4_K_XL | `15` | **`20`** | `25` | `30`–`50` |

These ranges are judgments based on the currently collected histories, not measured quality thresholds. Use the memory-priority candidate when the additional savings matter, and compare output quality with pruning disabled before adopting it. Different workloads or more complete histories can change the appropriate setting; the difference between the two models' recommendations does not establish that one model inherently tolerates pruning better.

Analysis of the locally collected routing histories on specific workloads using the differential byte-target policy produced the following results. Both models have 40 layers and 384 routed experts per layer. The historical call share is the percentage of recorded expert selections belonging to the excluded set; the maximum per-layer share measures the same percentage within the most affected layer. These routing statistics may vary depending on the workload, and the information presented is for reference only.

| Model | Memory target | Historical call share excluded | Maximum per-layer share excluded | Resident expert memory saved |
| --- | ---: | ---: | ---: | ---: |
| dsx 2.6 bpw | 5% | 0.000% | 0.000% | 8.27 GiB |
| dsx 2.6 bpw | 10% | 0.000% | 0.000% | 16.54 GiB |
| dsx 2.6 bpw | 15% | 0.000% | 0.000% | 24.81 GiB |
| dsx 2.6 bpw | 20% | 0.005% | 0.032% | 33.07 GiB |
| dsx 2.6 bpw | 25% | 0.047% | 0.091% | 41.33 GiB |
| dsx 2.6 bpw | 30% | 0.129% | 0.198% | 49.60 GiB |
| dsx 2.6 bpw | 35% | 0.275% | 0.395% | 57.87 GiB |
| dsx 2.6 bpw | 40% | 0.517% | 0.765% | 66.14 GiB |
| UD-Q4_K_XL | 5% | 0.000% | 0.000% | 13.45 GiB |
| UD-Q4_K_XL | 10% | 0.009% | 0.026% | 26.89 GiB |
| UD-Q4_K_XL | 15% | 0.053% | 0.097% | 40.34 GiB |
| UD-Q4_K_XL | 20% | 0.145% | 0.283% | 53.79 GiB |
| UD-Q4_K_XL | 25% | 0.304% | 0.502% | 67.24 GiB |
| UD-Q4_K_XL | 30% | 0.543% | 0.813% | 80.68 GiB |
| UD-Q4_K_XL | 35% | 0.884% | 1.258% | 94.13 GiB |
| UD-Q4_K_XL | 40% | 1.352% | 1.847% | 107.58 GiB |

At `5`, `10` and `15` for dsx 2.6 bpw, and at `5` for UD-Q4_K_XL, the excluded experts have exactly zero recorded calls in the current histories. This does not establish that those experts are unnecessary on other workloads or that pruning has no quality impact.

For **dsx 2.6 bpw**, `30` saves 49.60 GiB while excluding 0.129% of historical selections overall and at most 0.198% within any layer. Moving from `30` to `35` saves another 8.27 GiB, but the overall excluded share more than doubles to 0.275%. At `40`, it rises to 0.517%, which is why `30` is the balanced starting point and `35` is the memory-priority candidate.

For **Unsloth UD-Q4_K_XL**, `20` saves 53.79 GiB with a 0.145% overall excluded share and a 0.283% maximum per-layer share. Moving to `25` adds 13.45 GiB of savings, but the overall excluded share more than doubles to 0.304% and the maximum per-layer share rises to 0.502%. At `30`, those shares reach 0.543% and 0.813%; at `35`, the maximum per-layer share exceeds 1% (1.258%). This supports starting at `20`, considering `25` for memory savings, and reserving `30` or higher for explicit quality comparisons. The 1% figure is a history-coverage heuristic, not a proven quality boundary.

At `25`, the 2.6 bpw history selects between 20 and 183 excluded experts per layer, including 20 in layer 0. The selection rule is a greedy usage-per-byte heuristic, not a measured estimate of layer sensitivity or a guarantee of optimal output quality.

The memory savings above apply to resident expert weight tensors only. A positive RAM budget remains a configured ceiling. For UD-Q4_K_XL on the 192 GiB target system, the recommended `20` leaves 215.16 GiB of routed expert weights, `25` leaves 201.71 GiB, and `30` still leaves 188.26 GiB before other weights and runtime buffers, so use budgeted loading as shown above rather than assuming the recommended values allow full resident loading.

The existing 2.6 bpw resident/pinned smoke test used `25` with MTP disabled and completed 64-token generations for both the short prompt and `PROMPT_TEST.md`. The other settings above were evaluated against the saved routing histories; these calculations and the smoke test do not establish output-quality parity with pruning disabled.

These call shares are **not quality-loss percentages or percentages of affected tokens**: pruned selections are replaced by other experts, and each token passes through many layers. The histories do not record selection rank, routing weight, prompt domain or quality labels; pruning can also change later hidden states and routing decisions. Rare or unobserved experts may matter on unseen tasks. Collect representative history for your workload and validate quality before increasing pruning; the recommendation is based on historical usage coverage, not a measured quality guarantee.
