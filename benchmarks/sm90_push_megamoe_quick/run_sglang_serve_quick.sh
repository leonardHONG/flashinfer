#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Short SGLang online-serving benchmark.  The server must already be running.
# Detailed request arrays are saved so summarize.py can calculate p50/p90/p99
# and goodput with the same SLOs used by the vLLM wrapper.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_URL="${SM90_SGLANG_BASE_URL:-http://127.0.0.1:30000}"
BACKEND="${SM90_SGLANG_BENCH_BACKEND:-sglang}"
BENCH_MODULE="${SM90_SGLANG_BENCH_MODULE:-sglang.benchmark.serving}"
MODEL="${SM90_SERVE_MODEL:-}"
TOKENIZER="${SM90_SERVE_TOKENIZER:-}"
VARIANT="${SM90_SERVE_VARIANT:-sm90_push}"
SCENARIOS="${SM90_SERVE_SCENARIOS:-prefill decode}"
WARMUPS="${SM90_SERVE_WARMUPS:-4}"
TTFT_SLO_MS="${SM90_SERVE_TTFT_SLO_MS:-500}"
TPOT_SLO_MS="${SM90_SERVE_TPOT_SLO_MS:-15}"
OUT="${1:-sm90_push_serving_quick}"
mkdir -p "$OUT/logs"
OUT="$(cd "$OUT" && pwd)"

MODEL_ARGS=()
if [[ -n "$MODEL" ]]; then
  MODEL_ARGS+=(--model "$MODEL")
fi
if [[ -n "$TOKENIZER" ]]; then
  MODEL_ARGS+=(--tokenizer "$TOKENIZER")
fi

for scenario in $SCENARIOS; do
  case "$scenario" in
    prefill)
      input_len="${SM90_SERVE_PREFILL_INPUT_LEN:-2048}"
      output_len="${SM90_SERVE_PREFILL_OUTPUT_LEN:-32}"
      concurrency="${SM90_SERVE_PREFILL_CONCURRENCY:-8}"
      prompts="${SM90_SERVE_PREFILL_PROMPTS:-40}"
      request_rate="${SM90_SERVE_PREFILL_REQUEST_RATE:-4}"
      ;;
    decode)
      input_len="${SM90_SERVE_DECODE_INPUT_LEN:-256}"
      output_len="${SM90_SERVE_DECODE_OUTPUT_LEN:-64}"
      concurrency="${SM90_SERVE_DECODE_CONCURRENCY:-16}"
      prompts="${SM90_SERVE_DECODE_PROMPTS:-80}"
      request_rate="${SM90_SERVE_DECODE_REQUEST_RATE:-16}"
      ;;
    *)
      echo "unknown scenario '$scenario' (expected: prefill or decode)" >&2
      exit 2
      ;;
  esac

  result="$OUT/sglang_${VARIANT}_${scenario}.jsonl"
  log="$OUT/logs/sglang_${VARIANT}_${scenario}.log"
  rm -f "$result"
  echo "SGLang $VARIANT/$scenario: input=$input_len output=$output_len " \
    "concurrency=$concurrency prompts=$prompts rate=$request_rate"

  python -m "$BENCH_MODULE" \
    --backend "$BACKEND" \
    --base-url "$BASE_URL" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --random-range-ratio 1.0 \
    --num-prompts "$prompts" \
    --warmup-requests "$WARMUPS" \
    --request-rate "$request_rate" \
    --max-concurrency "$concurrency" \
    --output-file "$result" \
    --output-details \
    "${MODEL_ARGS[@]}" 2>&1 | tee "$log"
done

python benchmarks/sm90_push_megamoe_quick/summarize.py "$OUT" \
  --output "$OUT/summary.md" \
  --ttft-slo-ms "$TTFT_SLO_MS" \
  --tpot-slo-ms "$TPOT_SLO_MS"

echo "SGLang serving quick benchmark complete: $OUT"
