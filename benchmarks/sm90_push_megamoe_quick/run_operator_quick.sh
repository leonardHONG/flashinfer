#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Short, counterbalanced operator/protocol A/B.  Defaults intentionally cover
# one decode-like and one prefill-like token count without running the full
# acceptance matrix.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

NGPU="${SM90_QUICK_NGPU:-4}"
CONFIG="${SM90_QUICK_CONFIG:-SMALL}"
TOKENS="${SM90_QUICK_TOKENS:-64 2048}"
ROUTINGS="${SM90_QUICK_ROUTINGS:-random hot}"
WARMUP="${SM90_QUICK_WARMUP:-3}"
ITERS="${SM90_QUICK_ITERS:-10}"
PASSES="${SM90_QUICK_PASSES:-2}"
USE_NCCL="${SM90_QUICK_NCCL:-1}"
RUN_GRAPH="${SM90_QUICK_GRAPH:-0}"

OUT="${1:-sm90_push_quick_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/logs" "$OUT/deep_gemm_cache"
OUT="$(cd "$OUT" && pwd)"
RESULTS="$OUT/operator_results.jsonl"
: >"$RESULTS"

if ! [[ "$NGPU" =~ ^[0-9]+$ ]] || ((NGPU < 1)); then
  echo "SM90_QUICK_NGPU must be a positive integer, got '$NGPU'" >&2
  exit 2
fi
if ! [[ "$PASSES" =~ ^[0-9]+$ ]] || ((PASSES < 1)); then
  echo "SM90_QUICK_PASSES must be a positive integer, got '$PASSES'" >&2
  exit 2
fi

python - "$NGPU" <<'PY'
import sys
import torch

required = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() < required:
    raise SystemExit(
        f"need {required} visible GPUs, found {torch.cuda.device_count()}"
    )
for device in range(required):
    major, _ = torch.cuda.get_device_capability(device)
    if major != 9:
        raise SystemExit(
            f"device {device} is {torch.cuda.get_device_name(device)} "
            f"(SM{major}x), need SM90/Hopper"
        )
PY

if ((NGPU == 1)); then
  LAUNCH=(python)
else
  LAUNCH=(torchrun --standalone "--nproc-per-node=$NGPU")
fi

run_variant() {
  local routing=$1 tokens=$2 pass=$3 variant=$4 case_id=$5
  local -a flags=(--skip-baseline)
  local -a nccl=()
  if [[ "$variant" == "optimized" ]]; then
    flags+=(--dedup --grouped-combine --fuse-fc1)
  fi
  if ((NGPU > 1)) && [[ "$USE_NCCL" == "1" ]]; then
    nccl+=(--nccl-baseline)
  fi

  local log="$OUT/logs/${routing}_t${tokens}_p${pass}_${variant}.log"
  local cache="$OUT/deep_gemm_cache/$variant"
  mkdir -p "$cache"
  echo "[$(date +%T)] routing=$routing tokens=$tokens pass=$pass variant=$variant"
  TRTLLM_DG_CACHE_DIR="$cache" "${LAUNCH[@]}" \
    benchmarks/bench_sm90_push_megamoe.py \
    --config "$CONFIG" \
    --tokens "$tokens" \
    --routing "$routing" \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --case-id "$case_id" \
    --json "$RESULTS" \
    --assert-cos-min 0.997 \
    --assert-growth-max 1.25 \
    "${flags[@]}" \
    "${nccl[@]}" 2>&1 | tee "$log"
}

case_index=0
for routing in $ROUTINGS; do
  for tokens in $TOKENS; do
    case_id="quick_ep${NGPU}_${CONFIG}_${routing}_t${tokens}"
    python benchmarks/sm90_push_megamoe_quick/route_balance.py \
      --config "$CONFIG" \
      --tokens "$tokens" \
      --ep-size "$NGPU" \
      --routing "$routing" \
      --output "$OUT/route_${routing}_t${tokens}.json" \
      >"$OUT/logs/route_${routing}_t${tokens}.log"

    for ((pass = 1; pass <= PASSES; ++pass)); do
      # Mirrored pairs: A,B then B,A.  A one-pass smoke remains available
      # through SM90_QUICK_PASSES=1.
      if (((pass + case_index) % 2 == 1)); then
        order=(baseline optimized)
      else
        order=(optimized baseline)
      fi
      for variant in "${order[@]}"; do
        run_variant "$routing" "$tokens" "$pass" "$variant" "$case_id"
      done
    done

    if [[ "$RUN_GRAPH" == "1" ]]; then
      graph_log="$OUT/logs/${routing}_t${tokens}_graph.log"
      graph_cache="$OUT/deep_gemm_cache/optimized"
      TRTLLM_DG_CACHE_DIR="$graph_cache" "${LAUNCH[@]}" \
        benchmarks/bench_sm90_push_megamoe.py \
        --config "$CONFIG" \
        --tokens "$tokens" \
        --routing "$routing" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --case-id "${case_id}_graph" \
        --json "$RESULTS" \
        --skip-baseline \
        --dedup --grouped-combine --fuse-fc1 \
        --graph \
        --assert-cos-min 0.997 \
        --assert-growth-max 1.25 2>&1 | tee "$graph_log"
    fi
    case_index=$((case_index + 1))
  done
done

python benchmarks/sm90_push_megamoe_quick/summarize.py "$OUT" \
  --output "$OUT/summary.md" \
  --ttft-slo-ms "${SM90_QUICK_TTFT_SLO_MS:-500}" \
  --tpot-slo-ms "${SM90_QUICK_TPOT_SLO_MS:-15}"

echo "Quick operator benchmark complete: $OUT"
