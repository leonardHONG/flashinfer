#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Short vLLM online-serving benchmark.  The server must already be running.
# Run once against the baseline server and once against the SM90-push server,
# reusing the same output directory and changing SM90_SERVE_VARIANT.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASE_URL="${SM90_VLLM_BASE_URL:-http://127.0.0.1:8000}"
BACKEND="${SM90_VLLM_BENCH_BACKEND:-openai}"
ENDPOINT="${SM90_VLLM_ENDPOINT:-/v1/completions}"
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
EOS_ARGS=()
if [[ "${SM90_SERVE_IGNORE_EOS:-1}" == "1" ]]; then
  EOS_ARGS+=(--ignore-eos)
fi

git_commit="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

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

  result_name="vllm_${VARIANT}_${scenario}.json"
  log="$OUT/logs/vllm_${VARIANT}_${scenario}.log"
  rm -f "$OUT/$result_name"
  echo "vLLM $VARIANT/$scenario: input=$input_len output=$output_len " \
    "concurrency=$concurrency prompts=$prompts rate=$request_rate"

  vllm bench serve \
    --backend "$BACKEND" \
    --base-url "$BASE_URL" \
    --endpoint "$ENDPOINT" \
    --dataset-name random \
    --random-input-len "$input_len" \
    --random-output-len "$output_len" \
    --num-prompts "$prompts" \
    --num-warmups "$WARMUPS" \
    --request-rate "$request_rate" \
    --burstiness 1.0 \
    --max-concurrency "$concurrency" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --goodput "ttft:$TTFT_SLO_MS" "tpot:$TPOT_SLO_MS" \
    --save-result \
    --save-detailed \
    --result-dir "$OUT" \
    --result-filename "$result_name" \
    --label "vllm_${VARIANT}_${scenario}" \
    --metadata "variant=$VARIANT" "scenario=$scenario" "git_commit=$git_commit" \
    --disable-tqdm \
    "${MODEL_ARGS[@]}" \
    "${EOS_ARGS[@]}" 2>&1 | tee "$log"
done

python benchmarks/sm90_push_megamoe_quick/summarize.py "$OUT" \
  --output "$OUT/summary.md" \
  --ttft-slo-ms "$TTFT_SLO_MS" \
  --tpot-slo-ms "$TPOT_SLO_MS"

echo "vLLM serving quick benchmark complete: $OUT"
