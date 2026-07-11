#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Counterbalanced feature-matrix benchmark for the SM90 push MegaMoE flags
# (tiers: baseline / dedup / grouped / dedup+grouped / dedup+grouped+fused).
# Passes run in mirrored order pairs (asc,desc,desc,asc) so linear drift
# cancels within each pair; the reporter medians per-pair ratios and refuses
# mixed-provenance or incomplete cells. See docs/sm90_push_megamoe.md.
#   bash scripts/run_sm90_push_feature_matrix.sh [output_dir]
# Knobs: SM90_PUSH_MATRIX_{NGPU,CONFIG,ROUTINGS,WARMUP,ITERS,TOKENS,PASSES}.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SM90_PUSH_MATRIX_NGPU="${SM90_PUSH_MATRIX_NGPU:-4}"
SM90_PUSH_MATRIX_CONFIG="${SM90_PUSH_MATRIX_CONFIG:-SMALL}"
SM90_PUSH_MATRIX_ROUTINGS="${SM90_PUSH_MATRIX_ROUTINGS:-random hot all_remote}"
SM90_PUSH_MATRIX_WARMUP="${SM90_PUSH_MATRIX_WARMUP:-10}"
SM90_PUSH_MATRIX_ITERS="${SM90_PUSH_MATRIX_ITERS:-100}"
SM90_PUSH_MATRIX_TOKENS="${SM90_PUSH_MATRIX_TOKENS:-}"
SM90_PUSH_MATRIX_PASSES="${SM90_PUSH_MATRIX_PASSES:-4}"

if ! [[ "$SM90_PUSH_MATRIX_PASSES" =~ ^[0-9]+$ ]] || ((SM90_PUSH_MATRIX_PASSES < 2)) || ((SM90_PUSH_MATRIX_PASSES % 2 != 0)); then
  echo "FATAL: SM90_PUSH_MATRIX_PASSES must be an even integer >= 2 (mirrored pass pairs)," \
    "got '$SM90_PUSH_MATRIX_PASSES'" >&2
  exit 2
fi

OUT="${1:-sm90_push_feature_matrix_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.txt"
RESULTS="$OUT/results.jsonl"
REPORT="$OUT/feature_matrix_report.md"
: >"$SUMMARY"
: >"$RESULTS"

TIER_FLAGS=(
  ""
  "--dedup"
  "--grouped-combine"
  "--dedup --grouped-combine"
  "--dedup --grouped-combine --fuse-fc1"
)
TIER_NAMES=(baseline dedup grouped dedup_grouped dedup_grouped_fused)

run_tier() {
  local routing=$1 tier=$2 pass=$3
  local name="r_${routing}_t${tier}_p${pass}"
  local log_file="$OUT/${name}.txt"
  local -a flags=()
  # shellcheck disable=SC2206
  flags=(${TIER_FLAGS[$tier]})
  local -a extra=()
  if [[ "$tier" == 0 || "$tier" == 4 ]]; then
    extra+=(--nccl-baseline)
  fi
  if [[ -n "$SM90_PUSH_MATRIX_TOKENS" ]]; then
    extra+=(--tokens "$SM90_PUSH_MATRIX_TOKENS")
  fi
  echo "==== $name (${TIER_NAMES[$tier]})" | tee -a "$SUMMARY"
  if TRTLLM_DG_CACHE_DIR="$OUT/dgcache_t${tier}" \
    torchrun --standalone --nproc-per-node="$SM90_PUSH_MATRIX_NGPU" \
    benchmarks/bench_sm90_push_megamoe.py \
    --config "$SM90_PUSH_MATRIX_CONFIG" --routing "$routing" \
    --warmup "$SM90_PUSH_MATRIX_WARMUP" --iters "$SM90_PUSH_MATRIX_ITERS" \
    --case-id "tier_${routing}" --skip-baseline \
    --json "$RESULTS" \
    --assert-cos-min 0.997 --assert-growth-max 1.25 \
    "${flags[@]}" "${extra[@]}" >"$log_file" 2>&1; then
    echo "PASS $name" | tee -a "$SUMMARY"
  else
    echo "FAIL $name (see $log_file)" | tee -a "$SUMMARY"
    exit 1
  fi
}

read -r -a ROUTES <<<"$SM90_PUSH_MATRIX_ROUTINGS"
for routing in "${ROUTES[@]}"; do
  for ((p = 1; p <= SM90_PUSH_MATRIX_PASSES; p++)); do
    pair=$(((p - 1) / 2))
    first_of_pair=$((p % 2)) # 1 for the pair's first pass, 0 for its second
    if (((pair % 2) == (1 - first_of_pair))); then
      order="asc"
    else
      order="desc"
    fi
    if [[ "$order" == "asc" ]]; then
      for tier in 0 1 2 3 4; do run_tier "$routing" "$tier" "$p"; done
    else
      for tier in 4 3 2 1 0; do run_tier "$routing" "$tier" "$p"; done
    fi
  done
done

python - "$RESULTS" "$SM90_PUSH_MATRIX_PASSES" >"$REPORT" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

records = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text().splitlines()
    if line.strip()
]
expected_passes = int(sys.argv[2])


def num(v):
    return float(v) if isinstance(v, (int, float)) else float("nan")

prov = {(r.get("git_commit"), r.get("source_hash")) for r in records}
if len(prov) > 1:
    raise SystemExit(f"refusing to report: mixed provenance in results {prov}")
if any(r.get("git_dirty") for r in records):
    print("> WARNING: records were produced from a dirty tree; "
          "treat numbers as provisional.")
commit = next(iter(prov))[0] if prov else "?"

TIERS = [
    ("baseline", (False, False, False)),
    ("dedup", (True, False, False)),
    ("grouped", (False, True, False)),
    ("dedup+grouped", (True, True, False)),
    ("dedup+grouped+fused", (True, True, True)),
]


def flags_of(r):
    return (
        bool(r.get("dedup")),
        bool(r.get("grouped_combine")),
        bool(r.get("fuse_fc1")),
    )


def cell_rows(case, mode, flags):
    return [
        r
        for r in records
        if r.get("case_id") == case and r.get("mode") == mode and flags_of(r) == flags
    ]


def median_p50(case, mode, flags):
    vals = [
        num(r.get("p50_ms"))
        for r in cell_rows(case, mode, flags)
        if math.isfinite(num(r.get("p50_ms")))
    ]
    return statistics.median(vals) if vals else float("nan")


_RUN_ORDER = {}
for _idx, _r in enumerate(records):
    _RUN_ORDER.setdefault(_r.get("run_id"), _idx)


def run_series(case, mode, flags):
    by_run = {}
    for r in cell_rows(case, mode, flags):
        v = num(r.get("p50_ms"))
        if math.isfinite(v):
            by_run.setdefault(r.get("run_id"), []).append(v)
    runs = sorted(by_run, key=lambda rid: _RUN_ORDER.get(rid, 1 << 60))
    return [statistics.median(by_run[rid]) for rid in runs]


def pair_means(series):
    return [
        (series[2 * k] + series[2 * k + 1]) / 2.0 for k in range(len(series) // 2)
    ]


def paired_ratio(num_series, den_series):
    if len(num_series) != len(den_series) or len(num_series) < 2:
        return float("nan")
    ratios = [
        n / d for n, d in zip(pair_means(num_series), pair_means(den_series)) if d > 0
    ]
    return statistics.median(ratios) if ratios else float("nan")


def cell_runs(case, mode, flags):
    return len({r.get("run_id") for r in cell_rows(case, mode, flags)})


def fmt(v):
    return "n/a" if not (isinstance(v, float) and math.isfinite(v)) else f"{v:.4f}"


incomplete = []
cases = sorted({r.get("case_id") for r in records if r.get("mode") == "e2e"})
print(f"# SM90 push feature matrix (commit {commit[:12]})")
print()
print(f"- Tier medians are cross-run over {expected_passes} counterbalanced passes")
print("  per (case, tier, mode); each record's per-iteration time is already")
print("  the barrier-aligned cross-rank max (group round latency).")
print("- Headline ratios are computed per mirrored pass-pair (pair value = mean")
print("  of the pair's mirrored-order runs) and reported as the median across")
print("  pairs.")
print("- Cells whose distinct run_id count != the expected pass count are")
print("  flagged incomplete and fail the script.")
print("- push-vs-NCCL uses the same-run wire-only lower-bound NCCL twin (it")
print("  moves the bytes but does none of the compact/combine/reduce work);")
print("  with the flags on the push path also moves fewer bytes by design.")
for case in cases:
    meta = next(r for r in records if r.get("case_id") == case)
    print()
    print(f"## case={case} (routing={meta.get('routing')} "
          f"tokens={meta.get('config', {}).get('tokens')} ep={meta.get('ep')})")
    series = {}
    for mode in ("e2e", "transport"):
        print(f"- {mode} median p50 (ms):", end="")
        for name, flags in TIERS:
            m = median_p50(case, mode, flags)
            n_runs = cell_runs(case, mode, flags)
            if n_runs != expected_passes:
                incomplete.append(f"{case}/{mode}/{name}: {n_runs}/{expected_passes} runs")
                series[(mode, name)] = []
                print(f"  {name}=INCOMPLETE({n_runs}/{expected_passes})", end="")
                continue
            series[(mode, name)] = run_series(case, mode, flags)
            print(f"  {name}={fmt(m)}", end="")
        print()
    for mode in ("e2e", "transport"):
        inc = paired_ratio(series[(mode, "dedup+grouped")], series[(mode, "dedup+grouped+fused")])
        overall = paired_ratio(series[(mode, "baseline")], series[(mode, "dedup+grouped+fused")])
        print(f"- {mode}: fused-FC1 increment (dedup+grouped / dedup+grouped+fused) {fmt(inc)}x; "
              f"overall (baseline / dedup+grouped+fused) {fmt(overall)}x "
              f"(median of per-pair ratios)")
    nccl_series = run_series(case, "transport_nccl", (True, True, True))
    ratio = paired_ratio(nccl_series, series.get(("transport", "dedup+grouped+fused"), []))
    if math.isfinite(ratio):
        print(f"- transport dedup+grouped+fused vs same-run wire-only NCCL lower bound: "
              f"{fmt(ratio)}x")

if incomplete:
    print()
    print("INCOMPLETE CELLS (report is not publication-grade):")
    for line in incomplete:
        print(f"- {line}")
    raise SystemExit(1)
PY

echo "PASS all tiers; report: $REPORT" | tee -a "$SUMMARY"
cat "$REPORT"
