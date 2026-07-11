#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# One short stage/kernel profile for dispatch, compact, FC1/FC2, combine,
# reduce, and ack.  This is diagnostic output, not a throughput benchmark.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

NGPU="${SM90_QUICK_NGPU:-4}"
CONFIG="${SM90_QUICK_CONFIG:-SMALL}"
TOKENS="${SM90_PROFILE_TOKENS:-2048}"
ROUTING="${SM90_PROFILE_ROUTING:-random}"
OUT="${1:-sm90_push_profile_quick_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT/deep_gemm_cache"
OUT="$(cd "$OUT" && pwd)"

if ((NGPU == 1)); then
  LAUNCH=(python)
else
  LAUNCH=(torchrun --standalone "--nproc-per-node=$NGPU")
fi

TRTLLM_DG_CACHE_DIR="$OUT/deep_gemm_cache" "${LAUNCH[@]}" \
  benchmarks/bench_sm90_push_megamoe.py \
  --config "$CONFIG" \
  --tokens "$TOKENS" \
  --routing "$ROUTING" \
  --dedup --grouped-combine --fuse-fc1 \
  --skip-baseline \
  --nvtx \
  --torch-profile \
  --warmup "${SM90_PROFILE_WARMUP:-2}" \
  --iters "${SM90_PROFILE_ITERS:-5}" 2>&1 | tee "$OUT/profile.log"

echo "Stage/kernel profile written to $OUT/profile.log"
