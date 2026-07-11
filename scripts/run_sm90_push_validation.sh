#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# SM90 push MegaMoE validation suite. Requires a CUDA-visible SM90 device;
# fails non-zero if any step fails. EP sizes above the visible GPU count are
# skipped unless SM90_PUSH_GATE_REQUIRE_GPUS forbids it.
#   bash scripts/run_sm90_push_validation.sh [output_dir]
# Knobs: SM90_PUSH_GATE_{REQUIRE_GPUS,SPEEDUP_MIN,SOAK_ROUNDS}.
set -uo pipefail

DIR="${1:-sm90_push_validation_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$DIR"
SUMMARY="$DIR/summary.txt"
: > "$SUMMARY"

if ! python -c "import sys, torch; ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 9; print('SM90 precondition:', 'OK ' + torch.cuda.get_device_name(0) if ok else 'FAIL (need a CUDA-visible Hopper/SM90 device)'); sys.exit(0 if ok else 1)" 2>&1 | tee -a "$SUMMARY"; then
  echo "FATAL: SM90 precondition failed -- refusing to run" | tee -a "$SUMMARY"
  exit 1
fi

NGPU="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
echo "GPUs visible: $NGPU" | tee -a "$SUMMARY"

if [ -n "${SM90_PUSH_GATE_REQUIRE_GPUS:-}" ] && [ "$NGPU" -lt "$SM90_PUSH_GATE_REQUIRE_GPUS" ]; then
  echo "FATAL: SM90_PUSH_GATE_REQUIRE_GPUS=$SM90_PUSH_GATE_REQUIRE_GPUS but only $NGPU GPU(s) visible" | tee -a "$SUMMARY"
  exit 1
fi

failures=0
step() {
  local name=$1
  shift
  local out="$DIR/${name}.txt"
  echo "==== $name: $*" | tee -a "$SUMMARY"
  if "$@" >"$out" 2>&1; then
    echo "PASS $name" | tee -a "$SUMMARY"
  else
    echo "FAIL $name (see $out)" | tee -a "$SUMMARY"
    failures=$((failures + 1))
  fi
}

# a skipped test would silently shrink the gate, so skips fail the step
pystep() {
  local name=$1
  shift
  local out="$DIR/${name}.txt"
  echo "==== $name: $*" | tee -a "$SUMMARY"
  if ! "$@" >"$out" 2>&1; then
    echo "FAIL $name (see $out)" | tee -a "$SUMMARY"
    failures=$((failures + 1))
  elif grep -Eq '[0-9]+ skipped' "$out"; then
    echo "FAIL $name (tests skipped; skips are failures here -- see $out)" | tee -a "$SUMMARY"
    failures=$((failures + 1))
  else
    echo "PASS $name" | tee -a "$SUMMARY"
  fi
}

skip() {
  echo "SKIP $1 ($2)" | tee -a "$SUMMARY"
}

pystep ep1_pytest python -m pytest tests/moe/test_sm90_push_megamoe.py -v -x \
  -k "not dist and not dsv3_shape"
pystep ep1_pytest_dsv3_shape env SM90_PUSH_HEAVY=1 \
  python -m pytest tests/moe/test_sm90_push_megamoe.py -v -x -k dsv3_shape
pystep ep1_pytest_moe_ep python -m pytest tests/moe/test_moe_ep_sm90_push.py \
  -v -x -k "not dist"

for routing in random hot; do
  step "ep1_bench_${routing}" python benchmarks/bench_sm90_push_megamoe.py \
    --config SMALL --routing "$routing" --json "$DIR/results.jsonl" \
    --assert-cos-min 0.997 --assert-growth-max 1.25 --assert-p99-jitter-max 2.0 \
    --assert-speedup-min "${SM90_PUSH_GATE_SPEEDUP_MIN:-1.0}"
done
# --skip-baseline: no local baseline round, so no speedup gate
step ep1_bench_graph python benchmarks/bench_sm90_push_megamoe.py \
  --config SMALL --graph --skip-baseline --json "$DIR/results.jsonl" \
  --assert-cos-min 0.997 --assert-growth-max 1.25
step ep1_preflight_ab python benchmarks/bench_sm90_push_megamoe.py \
  --config SMALL --preflight-ab --warmup 2 --iters 5 --skip-baseline \
  --assert-cos-min 0.997 --assert-growth-max 1.25

run_dist() {
  local n=$1
  shift
  torchrun --standalone --nproc-per-node="$n" "$@"
}

for n in 2 4 8; do
  if [ "$NGPU" -lt "$n" ]; then
    skip "ep${n}_gates" "need $n GPUs, have $NGPU"
    continue
  fi
  pystep "ep${n}_pytest" run_dist "$n" -m pytest \
    tests/moe/test_sm90_push_megamoe.py -v -x -k dist
  pystep "ep${n}_pytest_moe_ep" run_dist "$n" -m pytest \
    tests/moe/test_moe_ep_sm90_push.py -v -x -k dist
  pystep "ep${n}_soak" env SM90_PUSH_SOAK_ROUNDS="${SM90_PUSH_GATE_SOAK_ROUNDS:-200}" \
    torchrun --standalone --nproc-per-node="$n" -m pytest \
    tests/moe/test_sm90_push_megamoe.py -v -x -k dist_soak
  pystep "ep${n}_soak_dedup_grouped" env SM90_PUSH_SOAK_ROUNDS="${SM90_PUSH_GATE_SOAK_ROUNDS:-200}" \
    torchrun --standalone --nproc-per-node="$n" -m pytest \
    tests/moe/test_sm90_push_megamoe.py -v -x -k dist_dedup_grouped_soak
  pystep "ep${n}_soak_dedup_grouped_fused" env SM90_PUSH_SOAK_ROUNDS="${SM90_PUSH_GATE_SOAK_ROUNDS:-200}" \
    torchrun --standalone --nproc-per-node="$n" -m pytest \
    tests/moe/test_sm90_push_megamoe.py -v -x -k dist_dedup_grouped_fused_soak
  for routing in random hot all_remote; do
    step "ep${n}_bench_${routing}" run_dist "$n" \
      benchmarks/bench_sm90_push_megamoe.py --config SMALL --routing "$routing" \
      --nccl-baseline --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25 --assert-p99-jitter-max 2.0
  done
  step "ep${n}_bench_tok2048_graph" run_dist "$n" \
    benchmarks/bench_sm90_push_megamoe.py --config SMALL --tokens 2048 --graph \
    --nccl-baseline --json "$DIR/results.jsonl" \
    --assert-cos-min 0.997 --assert-growth-max 1.25
  for routing in random hot all_remote; do
    step "ep${n}_bench_dedup_${routing}" run_dist "$n" \
      benchmarks/bench_sm90_push_megamoe.py --config SMALL --routing "$routing" \
      --dedup --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25 --assert-p99-jitter-max 2.0
    step "ep${n}_bench_grouped_${routing}" run_dist "$n" \
      benchmarks/bench_sm90_push_megamoe.py --config SMALL --routing "$routing" \
      --grouped-combine --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25 --assert-p99-jitter-max 2.0
    step "ep${n}_bench_dedup_grouped_${routing}" run_dist "$n" \
      benchmarks/bench_sm90_push_megamoe.py --config SMALL --routing "$routing" \
      --dedup --grouped-combine --nccl-baseline --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25 --assert-p99-jitter-max 2.0
    step "ep${n}_bench_dedup_grouped_fused_${routing}" run_dist "$n" \
      benchmarks/bench_sm90_push_megamoe.py --config SMALL --routing "$routing" \
      --dedup --grouped-combine --fuse-fc1 --nccl-baseline \
      --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25 --assert-p99-jitter-max 2.0
  done
  if [ "$n" -eq 8 ]; then
    step "ep8_bench_dsv3_random" run_dist 8 \
      benchmarks/bench_sm90_push_megamoe.py --config DSV3 --routing random \
      --nccl-baseline --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25
    step "ep8_bench_dsv3_hot" run_dist 8 \
      benchmarks/bench_sm90_push_megamoe.py --config DSV3 --routing hot \
      --nccl-baseline --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25
    step "ep8_bench_dsv3_dedup_grouped_random" run_dist 8 \
      benchmarks/bench_sm90_push_megamoe.py --config DSV3 --routing random \
      --dedup --grouped-combine --nccl-baseline --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25
    step "ep8_bench_dsv3_dedup_grouped_hot" run_dist 8 \
      benchmarks/bench_sm90_push_megamoe.py --config DSV3 --routing hot \
      --dedup --grouped-combine --nccl-baseline --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25
    step "ep8_bench_dsv3_dedup_grouped_fused_random" run_dist 8 \
      benchmarks/bench_sm90_push_megamoe.py --config DSV3 --routing random \
      --dedup --grouped-combine --fuse-fc1 --nccl-baseline \
      --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25
    step "ep8_bench_dsv3_dedup_grouped_fused_hot" run_dist 8 \
      benchmarks/bench_sm90_push_megamoe.py --config DSV3 --routing hot \
      --dedup --grouped-combine --fuse-fc1 --nccl-baseline \
      --json "$DIR/results.jsonl" \
      --assert-cos-min 0.997 --assert-growth-max 1.25
  fi
done

step trace_regen bash -c \
  "rm -rf tests/trace/fi_trace_out && python tests/trace/example.py \
   && ls tests/trace/fi_trace_out/transform_weights_for_sm90_push*.json"

echo "" | tee -a "$SUMMARY"
if (( failures != 0 )); then
  echo "$failures gate(s) FAILED -- see $DIR" | tee -a "$SUMMARY"
  exit 1
fi
echo "all gates passed -- logs in $DIR" | tee -a "$SUMMARY"
