#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# Zero keeps the original resident loader; a positive process-tree budget enables tiering.
export DSV4F_RAM_BUDGET_MIB=${DSV4F_RAM_BUDGET_MIB:-0}
export DSV4F_RAM_RESERVE_MIB=${DSV4F_RAM_RESERVE_MIB:-4096}
for name in DSV4F_RAM_BUDGET_MIB DSV4F_RAM_RESERVE_MIB; do
    value=${!name}
    if [[ ! $value =~ ^[0-9]{1,10}$ ]] || ((10#$value > 16777216)); then
        echo "$name must be an integer in 0..16777216 MiB." >&2; exit 1
    fi
done
ram_budget_mib=$((10#$DSV4F_RAM_BUDGET_MIB))
if ((ram_budget_mib && ram_budget_mib <= 10#$DSV4F_RAM_RESERVE_MIB + 64)); then
    echo 'RAM budget must exceed RAM reserve by more than 64 MiB.' >&2; exit 1
fi
if [[ -v DSV4F_RAM_CACHE_MIB ]]; then
    if [[ ! $DSV4F_RAM_CACHE_MIB =~ ^[0-9]{1,10}$ ]] || ((10#$DSV4F_RAM_CACHE_MIB < 64 || 10#$DSV4F_RAM_CACHE_MIB > 16777216)); then
        echo 'DSV4F_RAM_CACHE_MIB must be an integer in 64..16777216 MiB.' >&2; exit 1
    fi
    export DSV4F_RAM_CACHE_MIB
fi
export MODEL="${MODEL:-$ROOT/models/DeepSeek-V4.1-Flash-1017-dsx-2.6bit.gguf}"
export DSV4F_ENGRAM_DIR="${DSV4F_ENGRAM_DIR:-$ROOT/models}"
export DSV4F_ENGRAM_DECODE=${DSV4F_ENGRAM_DECODE:-lut}
export DSV4F_ENGRAM_CACHE_MIB=${DSV4F_ENGRAM_CACHE_MIB:-32}
export DSV4F_ENGRAM_DEDUP=${DSV4F_ENGRAM_DEDUP:-0}
case "$DSV4F_ENGRAM_DECODE" in baseline|lut) ;; *) echo 'DSV4F_ENGRAM_DECODE must be baseline or lut.' >&2; exit 1;; esac
case "$DSV4F_ENGRAM_DEDUP" in 0|1) ;; *) echo 'DSV4F_ENGRAM_DEDUP must be 0 or 1.' >&2; exit 1;; esac
if [[ ! $DSV4F_ENGRAM_CACHE_MIB =~ ^[0-9]{1,4}$ ]] || ((10#$DSV4F_ENGRAM_CACHE_MIB > 4096)); then
    echo 'DSV4F_ENGRAM_CACHE_MIB must be an integer in 0..4096.' >&2; exit 1
fi
export CTX_SIZE=${CTX_SIZE:-262144}
export DSV4F_KV_RESIDENT=${DSV4F_KV_RESIDENT:-65536}
#export DSV4F_KV_RESIDENT=${DSV4F_KV_RESIDENT:-0}
if [[ ! $DSV4F_KV_RESIDENT =~ ^[0-9]{1,10}$ ]] ||
   ((10#$DSV4F_KV_RESIDENT > 2147483647 || (10#$DSV4F_KV_RESIDENT != 0 && 10#$DSV4F_KV_RESIDENT < 2048))); then
    echo 'DSV4F_KV_RESIDENT must be 0 or an integer in 2048..2147483647.' >&2; exit 1
fi
export GGML_HIP_SPARSE_FA=${GGML_HIP_SPARSE_FA:-1}
case "$GGML_HIP_SPARSE_FA" in 0|1) ;; *) echo 'GGML_HIP_SPARSE_FA must be 0 or 1.' >&2; exit 1;; esac
export GGML_HIP_PREFILL_FA=${GGML_HIP_PREFILL_FA:-2}
case "$GGML_HIP_PREFILL_FA" in 0|1|2) ;; *) echo 'GGML_HIP_PREFILL_FA must be 0, 1 or 2.' >&2; exit 1;; esac
export GGML_HIP_PREFILL_FA_SPILL=${GGML_HIP_PREFILL_FA_SPILL:-1}
case "$GGML_HIP_PREFILL_FA_SPILL" in 0|1) ;; *) echo 'GGML_HIP_PREFILL_FA_SPILL must be 0 or 1.' >&2; exit 1;; esac
export GGML_HIP_INDEXER_WMMA=${GGML_HIP_INDEXER_WMMA:-1}
case "$GGML_HIP_INDEXER_WMMA" in 0|1) ;; *) echo 'GGML_HIP_INDEXER_WMMA must be 0 or 1.' >&2; exit 1;; esac
export GGML_HIP_Q2_MMQ_TUNE=${GGML_HIP_Q2_MMQ_TUNE:-1}
case "$GGML_HIP_Q2_MMQ_TUNE" in 0|1) ;; *) echo 'GGML_HIP_Q2_MMQ_TUNE must be 0 or 1.' >&2; exit 1;; esac
export GGML_CUDA_MOE_CACHE=0
export DSV4F_CPU_THREADS=${DSV4F_CPU_THREADS:-12}
# Experimental model-owned routing constants; keep the original path for A/B.
export DSV4F_STATIC_EXPERT_MAPS=${DSV4F_STATIC_EXPERT_MAPS:-1}
export DSV4F_V41_HISTORY_GPU=${DSV4F_V41_HISTORY_GPU:-1}
export DSV4F_V41_DIRECT_KV=${DSV4F_V41_DIRECT_KV:-1}
export DSV4F_V41_COMPACT_KV=${DSV4F_V41_COMPACT_KV:-1}
export DSV4F_V41_MOE_OVERLAP=${DSV4F_V41_MOE_OVERLAP:-1}
export DSV4F_V41_MOE_PIPELINE=${DSV4F_V41_MOE_PIPELINE:-1}
export DSV4F_V41_PTQ_DOT=${DSV4F_V41_PTQ_DOT:-1}
export DSV4F_V41_SHARED_FUSED=${DSV4F_V41_SHARED_FUSED:-1}
export GGML_HIP_Q8_TILE=${GGML_HIP_Q8_TILE:-1}
export DSV4F_ENGRAM_PREFILL_OVERLAP=${DSV4F_ENGRAM_PREFILL_OVERLAP:-1}
export DSV4F_ENGRAM_PREFETCH=${DSV4F_ENGRAM_PREFETCH:-1}
case "$DSV4F_ENGRAM_PREFETCH" in 0|1) ;; *) echo 'DSV4F_ENGRAM_PREFETCH must be 0 or 1.' >&2; exit 1;; esac
export DSV4F_ENGRAM_OVERLAP=${DSV4F_ENGRAM_OVERLAP:-1}
export DSV4F_V41_ROUTE_FUSED=${DSV4F_V41_ROUTE_FUSED:-1}
export DSV4F_V41_ROUTER_FUSED=${DSV4F_V41_ROUTER_FUSED:-1}
export DSV4F_V41_ROUTE_FIRST=${DSV4F_V41_ROUTE_FIRST:-1}
# Fuse format-local routing with cache lookup/request publication.
export DSV4F_V41_ROUTE_PUBLISH_FUSED=${DSV4F_V41_ROUTE_PUBLISH_FUSED:-1}
case "$DSV4F_V41_ROUTE_PUBLISH_FUSED" in 0|1) ;; *) echo 'DSV4F_V41_ROUTE_PUBLISH_FUSED must be 0 or 1.' >&2; exit 1;; esac
export DSV4F_V41_FQ_STORE_FUSED=${DSV4F_V41_FQ_STORE_FUSED:-1}
export DSV4F_V41_HC_MAT_SPLIT=${DSV4F_V41_HC_MAT_SPLIT:-1}
export DSV4F_V41_HC_POST_PRE_FUSED=${DSV4F_V41_HC_POST_PRE_FUSED:-1}
case "$DSV4F_ENGRAM_PREFILL_OVERLAP" in 0|1|2) ;; *) echo 'DSV4F_ENGRAM_PREFILL_OVERLAP must be 0, 1, or 2.' >&2; exit 1;; esac
case "$DSV4F_ENGRAM_OVERLAP" in 0|1|2) ;; *) echo 'DSV4F_ENGRAM_OVERLAP must be 0, 1, or 2.' >&2; exit 1;; esac
for name in DSV4F_V41_FQ_STORE_FUSED DSV4F_V41_HC_MAT_SPLIT DSV4F_V41_DIRECT_KV DSV4F_V41_HISTORY_GPU DSV4F_V41_COMPACT_KV DSV4F_V41_MOE_OVERLAP DSV4F_V41_MOE_PIPELINE DSV4F_V41_PTQ_DOT DSV4F_V41_SHARED_FUSED GGML_HIP_Q8_TILE DSV4F_V41_ROUTE_FUSED DSV4F_V41_ROUTER_FUSED DSV4F_V41_ROUTE_FIRST DSV4F_V41_HC_POST_PRE_FUSED; do
    case "${!name}" in 0|1) ;; *) echo "$name must be 0 or 1." >&2; exit 1;; esac
done
case "$DSV4F_STATIC_EXPERT_MAPS" in 0|1) ;; *) echo 'DSV4F_STATIC_EXPERT_MAPS must be 0 or 1.' >&2; exit 1;; esac
export DSV4F_CPU_PTQ_ISA=${DSV4F_CPU_PTQ_ISA:-auto}
# auto enables hardware FP16 scale conversion on compatible CPU backends.
export DSV4F_CPU_PTQ_KERNEL=${DSV4F_CPU_PTQ_KERNEL:-auto}
case "$DSV4F_CPU_PTQ_KERNEL" in auto|baseline|f16c|threshold) ;; *) echo 'DSV4F_CPU_PTQ_KERNEL must be auto, baseline, f16c or threshold.' >&2; exit 1;; esac
export DSV4F_CPU_QUANT_ISA=${DSV4F_CPU_QUANT_ISA:-auto}
# auto selects the unsigned-offset VNNI kernel on the AVX-512 quant path.
export DSV4F_CPU_Q3_K_KERNEL=${DSV4F_CPU_Q3_K_KERNEL:-auto}
case "$DSV4F_CPU_Q3_K_KERNEL" in auto|signed|offset) ;; *) echo 'DSV4F_CPU_Q3_K_KERNEL must be auto, signed or offset.' >&2; exit 1;; esac
case "$DSV4F_CPU_QUANT_ISA" in auto|legacy|avx512) ;; *) echo 'DSV4F_CPU_QUANT_ISA must be auto, legacy or avx512.' >&2; exit 1;; esac
case "$DSV4F_CPU_PTQ_ISA" in auto|avx2|avx512) ;; *) echo 'DSV4F_CPU_PTQ_ISA must be auto, avx2 or avx512.' >&2; exit 1;; esac
export DSV4F_EXPERT_CACHE_MIB=${DSV4F_EXPERT_CACHE_MIB:-14336}
export DSV4F_VRAM_HEADROOM_MIB=${DSV4F_VRAM_HEADROOM_MIB:-256}
export DSV4F_VRAM_GUARD_MIB=${DSV4F_VRAM_GUARD_MIB:-256}
for name in DSV4F_EXPERT_CACHE_MIB DSV4F_VRAM_HEADROOM_MIB DSV4F_VRAM_GUARD_MIB; do
    value=${!name}
    if [[ ! $value =~ ^[0-9]{1,14}$ ]] || ((10#$value > 17592186044415)); then
        echo "$name must be a nonnegative integer in MiB." >&2; exit 1
    fi
done
# Q3/Q2/PTQ default to the measured 8:1:5 split of the aggregate ceiling.
# Keep explicit overrides (including zero); unused capacity is still subject
# to the aggregate budget and physical VRAM/headroom checks in the allocator.
cache_budget_mib=$((10#$DSV4F_EXPERT_CACHE_MIB))
cache_q3_mib=$((cache_budget_mib * 8 / 14))
cache_q2_mib=$((cache_budget_mib / 14))
export DSV4F_V41_CACHE_Q3_MIB=${DSV4F_V41_CACHE_Q3_MIB-$cache_q3_mib}
export DSV4F_V41_CACHE_Q2_MIB=${DSV4F_V41_CACHE_Q2_MIB-$cache_q2_mib}
export DSV4F_V41_CACHE_PTQ_MIB=${DSV4F_V41_CACHE_PTQ_MIB-$((cache_budget_mib - cache_q3_mib - cache_q2_mib))}
for name in DSV4F_V41_CACHE_Q3_MIB DSV4F_V41_CACHE_Q2_MIB DSV4F_V41_CACHE_PTQ_MIB; do
    if [[ -v $name ]]; then
        value=${!name}
        if [[ ! $value =~ ^[0-9]{1,14}$ ]] || ((10#$value > 17592186044415)); then
            echo "$name must be a nonnegative integer in MiB." >&2; exit 1
        fi
        export "$name"
    fi
done
export DSV4F_WHOLE_EXPERT=${DSV4F_WHOLE_EXPERT:-1}
export DSV4F_BATCH_TRANSFERS=${DSV4F_BATCH_TRANSFERS:-1}
export DSV4F_GPU_JOIN=${DSV4F_GPU_JOIN:-1}
export DSV4F_PINNED_COMPUTE=${DSV4F_PINNED_COMPUTE:-1}
export DSV4F_PINNED_WEIGHTS=${DSV4F_PINNED_WEIGHTS:-$DSV4F_PINNED_COMPUTE}
if ((ram_budget_mib)); then export DSV4F_PINNED_WEIGHTS=0; fi
export DSV4F_EXPERT_GRAPHS=${DSV4F_EXPERT_GRAPHS:-1}
# Experimental: instantiate each captured HIP graph instead of updating an old executable.
export DSV4F_HIP_GRAPH_REBUILD=${DSV4F_HIP_GRAPH_REBUILD:-0}
case "$DSV4F_HIP_GRAPH_REBUILD" in 0|1) ;; *) echo 'DSV4F_HIP_GRAPH_REBUILD must be 0 or 1.' >&2; exit 1;; esac
export DSV4F_FUSED_HC_MIX=${DSV4F_FUSED_HC_MIX:-1}
export DSV4F_GPU_SERVICE=${DSV4F_GPU_SERVICE:-1}
export DSV4F_GPU_WAIT=${DSV4F_GPU_WAIT:-callback}
export DSV4F_SKIP_NO_MISS=${DSV4F_SKIP_NO_MISS:-1}
case "$DSV4F_SKIP_NO_MISS" in 0|1) ;; *) echo 'DSV4F_SKIP_NO_MISS must be 0 or 1.' >&2; exit 1;; esac
export DSV4F_SERVICE_WAIT=${DSV4F_SERVICE_WAIT:-legacy}
export DSV4F_SERVICE_CALLBACK_SPIN_US=${DSV4F_SERVICE_CALLBACK_SPIN_US:-0}
export DSV4F_SERVICE_IDLE_SPINS=${DSV4F_SERVICE_IDLE_SPINS:-200000}
case "$DSV4F_SERVICE_WAIT" in legacy|notify) ;; *) echo 'DSV4F_SERVICE_WAIT must be legacy or notify.' >&2; exit 1;; esac
if [[ ! $DSV4F_SERVICE_CALLBACK_SPIN_US =~ ^[0-9]{1,4}$ ]] || ((10#$DSV4F_SERVICE_CALLBACK_SPIN_US > 1000)); then
    echo 'DSV4F_SERVICE_CALLBACK_SPIN_US must be an integer in 0..1000.' >&2; exit 1
fi
if [[ ! $DSV4F_SERVICE_IDLE_SPINS =~ ^[0-9]{1,7}$ ]] || ((10#$DSV4F_SERVICE_IDLE_SPINS > 2000000)); then
    echo 'DSV4F_SERVICE_IDLE_SPINS must be an integer in 0..2000000.' >&2; exit 1
fi
export DSV4F_CPU_SPIN_US=${DSV4F_CPU_SPIN_US:-300}
export DSV4F_CPU_FAST_SYNC=${DSV4F_CPU_FAST_SYNC:-1}
export DSV4F_CPU_AFFINITY=${DSV4F_CPU_AFFINITY:-none}
export DSV4F_CPU_SCHEDULE=${DSV4F_CPU_SCHEDULE:-dynamic}
export DSV4F_CPU_ROW_CHUNK=${DSV4F_CPU_ROW_CHUNK:-32}
export DSV4F_CPU_MXFP4_ROWS=${DSV4F_CPU_MXFP4_ROWS:-1}
export DSV4F_CPU_MXFP4_ISA=${DSV4F_CPU_MXFP4_ISA:-auto}
export DSV4F_CPU_EXPERT_LEASE=${DSV4F_CPU_EXPERT_LEASE:-1}
case "$DSV4F_CPU_AFFINITY" in none|physical) ;; *) echo 'DSV4F_CPU_AFFINITY must be none or physical.' >&2; exit 1;; esac
case "$DSV4F_CPU_SCHEDULE" in dynamic|static) ;; *) echo 'DSV4F_CPU_SCHEDULE must be dynamic or static.' >&2; exit 1;; esac
case "$DSV4F_CPU_EXPERT_LEASE" in 0|1) ;; *) echo 'DSV4F_CPU_EXPERT_LEASE must be 0 or 1.' >&2; exit 1;; esac
export DSV4F_HC_WARP=${DSV4F_HC_WARP:-1}
export DSV4F_HC_FULL=${DSV4F_HC_FULL:-1}
export DSV4F_HC_POST_FULL=${DSV4F_HC_POST_FULL:-1}
export DSV4F_PREFILL_STREAM=${DSV4F_PREFILL_STREAM:-1}
export DSV4F_PREFILL_GROUP=${DSV4F_PREFILL_GROUP:-16}
export DSV4F_PREFILL_CACHE=${DSV4F_PREFILL_CACHE:-0}
export DSV4F_PREFILL_CACHE_MIN_TOKENS=${DSV4F_PREFILL_CACHE_MIN_TOKENS:-4096}
case "$DSV4F_PREFILL_CACHE" in 0|1) ;; *) echo 'DSV4F_PREFILL_CACHE must be 0 or 1.' >&2; exit 1;; esac
if [[ ! $DSV4F_PREFILL_CACHE_MIN_TOKENS =~ ^[0-9]{1,10}$ ]] || ((10#$DSV4F_PREFILL_CACHE_MIN_TOKENS < 1 || 10#$DSV4F_PREFILL_CACHE_MIN_TOKENS > 2147483647)); then
    echo 'DSV4F_PREFILL_CACHE_MIN_TOKENS must be an integer in 1..2147483647.' >&2; exit 1
fi
case "$DSV4F_PREFILL_STREAM" in 0|1) ;; *) echo 'DSV4F_PREFILL_STREAM must be 0 or 1.' >&2; exit 1;; esac
if [[ ! $DSV4F_PREFILL_GROUP =~ ^[0-9]{1,3}$ ]] || ((10#$DSV4F_PREFILL_GROUP < 1 || 10#$DSV4F_PREFILL_GROUP > 256)); then
    echo 'DSV4F_PREFILL_GROUP must be an integer in 1..256.' >&2; exit 1
fi
export DSV4F_PHASE_WORKSPACE=${DSV4F_PHASE_WORKSPACE:-1}
export DSV4F_DECODE_BATCH=${DSV4F_DECODE_BATCH:-512}
export DSV4F_DECODE_UBATCH=${DSV4F_DECODE_UBATCH:-256}
# Expert lookahead: 0 off, 1/2 temporal, 3/4 early-router shadow/copy, 5 cache protection.
export DSV4F_EXPERT_LOOKAHEAD=${DSV4F_EXPERT_LOOKAHEAD:-0}
case "$DSV4F_EXPERT_LOOKAHEAD" in 0|1|2|3|4|5) ;; *) echo 'DSV4F_EXPERT_LOOKAHEAD must be 0..5.' >&2; exit 1;; esac
export DSV4F_EXPERT_LOOKAHEAD_DEPTH=${DSV4F_EXPERT_LOOKAHEAD_DEPTH:-8}
export DSV4F_EXPERT_LOOKAHEAD_SCORE=${DSV4F_EXPERT_LOOKAHEAD_SCORE:-raw}
case "$DSV4F_EXPERT_LOOKAHEAD_SCORE" in raw|native) ;; *) echo 'DSV4F_EXPERT_LOOKAHEAD_SCORE must be raw or native.' >&2; exit 1;; esac
if [[ ! $DSV4F_EXPERT_LOOKAHEAD_DEPTH =~ ^([1-9]|1[0-6])$ ]]; then
    echo 'DSV4F_EXPERT_LOOKAHEAD_DEPTH must be 1..16.' >&2; exit 1
fi
export DSV4F_EXPERT_LOOKAHEAD_READ=${DSV4F_EXPERT_LOOKAHEAD_READ:-$DSV4F_EXPERT_LOOKAHEAD_DEPTH}
if [[ ! $DSV4F_EXPERT_LOOKAHEAD_READ =~ ^([1-9]|1[0-6])$ ]] || ((DSV4F_EXPERT_LOOKAHEAD_READ > DSV4F_EXPERT_LOOKAHEAD_DEPTH)); then
    echo 'DSV4F_EXPERT_LOOKAHEAD_READ must be 1..DSV4F_EXPERT_LOOKAHEAD_DEPTH.' >&2; exit 1
fi
export DSV4F_CACHE_POLICY=${DSV4F_CACHE_POLICY:-aged}
case "$DSV4F_CACHE_POLICY" in legacy|scored|aged) ;; *) echo 'DSV4F_CACHE_POLICY must be legacy, scored or aged.' >&2; exit 1;; esac
export DSV4F_MXFP4_FUSED=${DSV4F_MXFP4_FUSED:-1}
case "$DSV4F_GPU_WAIT" in poll|callback) ;; *) echo 'DSV4F_GPU_WAIT must be poll or callback.' >&2; exit 1;; esac
# Pinned host weights reduce the first large prefill's expert-upload cost.
# Decode cache misses stay on CPU. Both host buffer kinds share one allocator.
case "$DSV4F_PINNED_WEIGHTS" in 0|1) ;; *) echo 'DSV4F_PINNED_WEIGHTS must be 0 or 1.' >&2; exit 1;; esac
# The bounded V4.1 KV cache writes rollback history to mapped host memory,
# independently of whether expert weights and compute buffers are pinned.
unset GGML_CUDA_NO_PINNED
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$DSV4F_CPU_THREADS}
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8080}
export API_KEY=${API_KEY:-dummykey}
# Disabled by default, MTP makes it slower for vram limited system.
export DSV4F_MTP=${DSV4F_MTP:-0}
case "$DSV4F_MTP" in 0|1) ;; *) echo 'DSV4F_MTP must be 0 or 1.' >&2; exit 1;; esac
if [[ $DSV4F_MTP == 1 ]]; then
    export MTP_MODEL="${MTP_MODEL:-$ROOT/models/dspark-xs2.bin}"
    export MTP_CONFIG="${MTP_CONFIG:-${MTP_MODEL%.bin}.json}"
    export DSV4F_DRAFT_N_MAX=${DSV4F_DRAFT_N_MAX:-1}
    export DSV4F_DRAFT_P_MIN=${DSV4F_DRAFT_P_MIN:-0.25}
    if [[ ! $DSV4F_DRAFT_N_MAX =~ ^[1-5]$ ]]; then
        echo 'DSV4F_DRAFT_N_MAX must be 1..5.' >&2; exit 1
    fi
    if ! python3 - "$DSV4F_DRAFT_P_MIN" <<'CHECK_MTP_THRESHOLD'
import math, sys
try:
    value = float(sys.argv[1])
    assert math.isfinite(value) and 0 <= value <= 1
except (ValueError, AssertionError):
    sys.exit(1)
CHECK_MTP_THRESHOLD
    then echo 'DSV4F_DRAFT_P_MIN must be a finite probability in 0..1.' >&2; exit 1; fi
    python3 "$ROOT/scripts/model_info.py" --mtp "$MTP_MODEL" "$MTP_CONFIG"
fi
BUILD_DIR=${BUILD_DIR:-$ROOT/build}

#--jinja --chat-template-file "$ROOT/models/dsv41f.jinja" --reasoning off --reasoning-budget 0 --no-context-shift
load_mode=none
if ((ram_budget_mib)); then load_mode=dio; fi
args=(--model "$MODEL" --ctx-size "$CTX_SIZE" --cache-type-k q8_0 --cache-type-v q8_0
      --load-mode "$load_mode" --lazy-mode off --no-repack --cpu-moe --n-gpu-layers 999 --fit off
      --threads "$DSV4F_CPU_THREADS" --threads-batch "$DSV4F_CPU_THREADS"
      --batch-size "${BATCH_SIZE:-4096}" --ubatch-size "${UBATCH_SIZE:-4096}"
      --flash-attn on --parallel 1 --host "$HOST" --port "$PORT" --api-key "$API_KEY"
      --jinja --chat-template-file "$ROOT/models/dsv41f.jinja" --reasoning on --reasoning-budget 4096 --no-context-shift
      --alias dsv4f --ui --path "$ROOT/web")
if [[ $DSV4F_MTP == 1 ]]; then
    args+=(--spec-type draft-mtp --spec-draft-n-max "$DSV4F_DRAFT_N_MAX"
           --spec-draft-n-min 0 --spec-draft-p-min "$DSV4F_DRAFT_P_MIN")
fi
if [[ $DSV4F_PINNED_WEIGHTS == 0 ]]; then args+=(--no-host); fi
if [[ -n ${MMPROJ:-} ]]; then
    echo 'DeepSeek V4.1 vision is not integrated in this engine.' >&2; exit 1
fi
if [[ ${1:-} == --dry-run ]]; then
    python3 "$ROOT/scripts/model_info.py" "$MODEL"
    echo "RAM budget: $ram_budget_mib MiB; SSD payload I/O: $([[ $ram_budget_mib == 0 ]] && echo resident || echo O_DIRECT)"
    printf '%q ' "$BUILD_DIR/bin/dsv4f-server" "${args[@]}" "${@:2}"
    printf '\n'
    exit 0
fi
[[ -x "$BUILD_DIR/bin/dsv4f-server" ]] || "$ROOT/build-engine.sh"
if ! python3 "$ROOT/scripts/memory_budget.py" --check; then
    if [[ ${DSV4F_SCOPE_REQUESTED:-0} == 1 ]]; then
        echo 'Requested cgroup RAM/swap limits were not applied; refusing an unbounded launch.' >&2; exit 1
    fi
    if command -v systemd-run >/dev/null; then
        scope_args=(-p MemorySwapMax=0)
        if ((ram_budget_mib)); then
            scope_args+=(-p "MemoryMax=$((ram_budget_mib * 1048576))"
                         -p "MemoryHigh=$(((ram_budget_mib - 10#$DSV4F_RAM_RESERVE_MIB / 2) * 1048576))")
        fi
        export DSV4F_SCOPE_REQUESTED=1
        exec systemd-run --user --scope --quiet --unit="dsv4f-$$" "${scope_args[@]}" "$ROOT/run-server.sh" "$@"
    fi
    echo 'Cannot enforce RAM/swap limits. A systemd user session with cgroup v2 memory control is required.' >&2
    exit 1
fi
state_dir=${DSV4F_STATE_DIR:-$ROOT/state}
log_dir=${DSV4F_LOG_DIR:-$ROOT/logs}
mkdir -p "$log_dir" "$state_dir"
exec python3 "$ROOT/scripts/serve.py" --state-dir "$state_dir" --log-dir "$log_dir" -- \
    "$BUILD_DIR/bin/dsv4f-server" "${args[@]}" "$@"
