#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Correctness + performance on/off comparison for the FC1 fused epilogue,
# with a small-token break-even sweep. Requires a Linux SM90/Hopper node.
#   bash scripts/run_sm90_push_fc1_epilogue_comparison.sh [output_dir]
# Knobs: SM90_PUSH_FC1_{NGPU,WARMUP,ITERS,ROUTINGS,TOKEN_SWEEP,SWEEP_ITERS,
#                       BUDGET_SECONDS}.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FEATURE_LABEL="FC1 fused SwiGLU+quant epilogue"
FEATURE_FIELD="fuse_fc1"
FEATURE_ARGS=(--fuse-fc1)
TEST_NODE="tests/moe/test_sm90_push_megamoe.py::test_ep1_fc1_fused_bitwise_equal"

SM90_PUSH_FC1_NGPU="${SM90_PUSH_FC1_NGPU:-4}"
SM90_PUSH_FC1_WARMUP="${SM90_PUSH_FC1_WARMUP:-5}"
SM90_PUSH_FC1_ITERS="${SM90_PUSH_FC1_ITERS:-30}"
SM90_PUSH_FC1_ROUTINGS="${SM90_PUSH_FC1_ROUTINGS:-random hot}"
SM90_PUSH_FC1_TOKEN_SWEEP="${SM90_PUSH_FC1_TOKEN_SWEEP:-64 512 2048}"
SM90_PUSH_FC1_SWEEP_ITERS="${SM90_PUSH_FC1_SWEEP_ITERS:-15}"
SM90_PUSH_FC1_BUDGET_SECONDS="${SM90_PUSH_FC1_BUDGET_SECONDS:-1500}"

OUT="${1:-sm90_push_fc1_epilogue_comparison_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
RESULTS="$OUT/results.jsonl"
REPORT="$OUT/comparison_summary.md"
: >"$SUMMARY"
: >"$RESULTS"

if ! command -v timeout >/dev/null 2>&1; then
  echo "FATAL: GNU timeout is required to enforce SM90_PUSH_FC1_BUDGET_SECONDS" | tee -a "$SUMMARY"
  exit 1
fi

if ! [[ "$SM90_PUSH_FC1_NGPU" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$SM90_PUSH_FC1_WARMUP" =~ ^[0-9]+$ ]] ||
   ! [[ "$SM90_PUSH_FC1_ITERS" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$SM90_PUSH_FC1_BUDGET_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: SM90_PUSH_FC1_NGPU/SM90_PUSH_FC1_WARMUP/SM90_PUSH_FC1_ITERS/SM90_PUSH_FC1_BUDGET_SECONDS must be integers" |
    tee -a "$SUMMARY"
  exit 1
fi

if ! python - "$SM90_PUSH_FC1_NGPU" <<'PY' 2>&1 | tee -a "$SUMMARY"
import sys
import torch

n = int(sys.argv[1])
available = torch.cuda.device_count()
if available < n:
    raise SystemExit(f"need {n} visible GPUs, found {available}")
bad = [
    f"cuda:{i}={torch.cuda.get_device_name(i)} cc={torch.cuda.get_device_capability(i)}"
    for i in range(n)
    if torch.cuda.get_device_capability(i)[0] != 9
]
if bad:
    raise SystemExit("all selected GPUs must be SM90 Hopper: " + ", ".join(bad))
print(f"SM90 precondition: OK ({n}/{available} visible GPUs selected)")
PY
then
  echo "FATAL: GPU precondition failed" | tee -a "$SUMMARY"
  exit 1
fi

{
  echo "feature: $FEATURE_LABEL"
  echo "git_head: $(git rev-parse HEAD)"
  echo "utc_start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ngpu: $SM90_PUSH_FC1_NGPU"
  echo "warmup: $SM90_PUSH_FC1_WARMUP"
  echo "iters: $SM90_PUSH_FC1_ITERS"
  echo "routings: $SM90_PUSH_FC1_ROUTINGS"
  echo "budget_seconds: $SM90_PUSH_FC1_BUDGET_SECONDS"
  python - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"cuda: {torch.version.cuda}")
for i in range(torch.cuda.device_count()):
    print(f"gpu_{i}: {torch.cuda.get_device_name(i)}")
PY
} >"$OUT/environment.txt"

START_SECONDS="$(date +%s)"

run_step() {
  local name=$1
  shift
  local now remaining rc log_file
  now="$(date +%s)"
  remaining=$((SM90_PUSH_FC1_BUDGET_SECONDS - (now - START_SECONDS)))
  if ((remaining <= 0)); then
    echo "FATAL: time budget exhausted before $name" | tee -a "$SUMMARY"
    exit 124
  fi
  log_file="$OUT/${name}.txt"
  {
    printf '==== %s (remaining %ss):' "$name" "$remaining"
    printf ' %q' "$@"
    printf '\n'
  } | tee -a "$SUMMARY"
  if timeout --signal=INT --kill-after=30s "${remaining}s" "$@" >"$log_file" 2>&1; then
    echo "PASS $name" | tee -a "$SUMMARY"
  else
    rc=$?
    echo "FAIL $name (exit $rc; see $log_file)" | tee -a "$SUMMARY"
    exit "$rc"
  fi
}

rm -rf ~/.cache/flashinfer ~/.tensorrt_llm/cache "${TRTLLM_DG_CACHE_DIR:-/nonexistent-unset}" || true
echo "cleared flashinfer + DeepGEMM kernel caches" | tee -a "$SUMMARY"

run_step correctness python -m pytest "$TEST_NODE" -q -x

read -r -a ROUTES <<<"$SM90_PUSH_FC1_ROUTINGS"
for routing in "${ROUTES[@]}"; do
  case "$routing" in
  random | hot | hot1 | all_remote) ;;
  *)
    echo "FATAL: unsupported routing '$routing'" | tee -a "$SUMMARY"
    exit 2
    ;;
  esac

  common=(
    benchmarks/bench_sm90_push_megamoe.py
    --config SMALL
    --routing "$routing"
    --warmup "$SM90_PUSH_FC1_WARMUP"
    --iters "$SM90_PUSH_FC1_ITERS"
    --case-id "ab_${routing}"
    --skip-baseline
    --json "$RESULTS"
    --assert-cos-min 0.997
    --assert-growth-max 1.25
  )
  run_step "bench_${routing}_baseline" torchrun --standalone \
    --nproc-per-node="$SM90_PUSH_FC1_NGPU" "${common[@]}"
  run_step "bench_${routing}_optimized" torchrun --standalone \
    --nproc-per-node="$SM90_PUSH_FC1_NGPU" "${common[@]}" "${FEATURE_ARGS[@]}"
done

if [[ -n "$SM90_PUSH_FC1_TOKEN_SWEEP" ]]; then
  read -r -a SWEEP_TOKENS <<<"$SM90_PUSH_FC1_TOKEN_SWEEP"
  for tok in "${SWEEP_TOKENS[@]}"; do
    if ! [[ "$tok" =~ ^[1-9][0-9]*$ ]]; then
      echo "FATAL: bad token count '$tok' in SM90_PUSH_FC1_TOKEN_SWEEP" | tee -a "$SUMMARY"
      exit 2
    fi
    sweep=(
      benchmarks/bench_sm90_push_megamoe.py
      --config SMALL
      --tokens "$tok"
      --routing random
      --warmup "$SM90_PUSH_FC1_WARMUP"
      --iters "$SM90_PUSH_FC1_SWEEP_ITERS"
      --case-id "sweep_t${tok}"
      --skip-baseline
      --json "$RESULTS"
      --assert-cos-min 0.997
    )
    run_step "sweep_t${tok}_baseline" torchrun --standalone \
      --nproc-per-node="$SM90_PUSH_FC1_NGPU" "${sweep[@]}"
    run_step "sweep_t${tok}_optimized" torchrun --standalone \
      --nproc-per-node="$SM90_PUSH_FC1_NGPU" "${sweep[@]}" "${FEATURE_ARGS[@]}"
  done
fi

python - "$RESULTS" "$FEATURE_FIELD" "$FEATURE_LABEL" "$SM90_PUSH_FC1_ITERS" \
  >"$REPORT" <<'PY'
import json
import math
import sys
from pathlib import Path

path, field, label, iters = sys.argv[1:]
records = [
    json.loads(line)
    for line in Path(path).read_text().splitlines()
    if line.strip()
]


def selected(case, mode, enabled):
    return [
        row
        for row in records
        if row.get("case_id") == case
        and row.get("mode") == mode
        and bool(row.get(field, False)) is enabled
    ]


def worst(rows, key):
    # the bench serializes NaN/Inf as JSON null; treat null as absent
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
    ]
    return max(values) if values else float("nan")


def fmt(value):
    return "n/a" if not math.isfinite(value) else f"{value:.4f}"


def case_meta(case):
    row = next(r for r in records if r.get("case_id") == case)
    return (
        row.get("routing"),
        row.get("config", {}).get("tokens"),
        row.get("iters"),
    )


cases = sorted({row.get("case_id") for row in records if row.get("mode") == "e2e"})

print(f"# Quick A/B: {label}")
print()
print(f"- Timing samples per full run: {iters} (token sweep uses fewer)")
print("- Reported p99 is only the sampled tail; it is not a release-grade p99.")
print("- Group latency uses the benchmark's barrier-aligned cross-rank timing.")
print("- The small-token rows are the break-even scan: the fused kernel has")
print("  no small-M swapAB tactic, so speedup < 1 at small T is expected;")
print("  keep --fuse-fc1 off below the crossover.")
for case in cases:
    route, tokens, case_iters = case_meta(case)
    print()
    print(f"## case={case} (routing={route} tokens={tokens} iters={case_iters})")
    for mode in ("e2e", "transport"):
        base_rows = selected(case, mode, False)
        opt_rows = selected(case, mode, True)
        if not base_rows and not opt_rows:
            continue
        base = worst(base_rows, "p50_ms")
        opt = worst(opt_rows, "p50_ms")
        speedup = base / opt if math.isfinite(base) and opt > 0 else float("nan")
        p99 = worst(opt_rows, "p99_ms")
        print(
            f"- {mode}: baseline p50 {fmt(base)} ms; optimized p50 {fmt(opt)} ms; "
            f"speedup {fmt(speedup)}x; optimized sampled-p99 {fmt(p99)} ms"
        )
    opt_e2e = selected(case, "e2e", True)
    cos_values = [
        float(row["cos"])
        for row in opt_e2e
        if isinstance(row.get("cos"), (int, float)) and math.isfinite(float(row["cos"]))
    ]
    growth = worst(opt_e2e, "growth")
    cos = min(cos_values) if cos_values else float("nan")
    print(f"- optimized correctness: min cosine {fmt(cos)}; max growth {fmt(growth)}")
PY

ELAPSED=$(( $(date +%s) - START_SECONDS ))
echo "PASS all steps in ${ELAPSED}s; report: $REPORT" | tee -a "$SUMMARY"
python - "$REPORT" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).read_text())
PY
