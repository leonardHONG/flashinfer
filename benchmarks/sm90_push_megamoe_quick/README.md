# SM90 Push MegaMoE Quick Benchmarks

Short, isolated benchmarks for checking the SM90 push backend before running
the full 8-GPU acceptance gate.  Nothing in this directory changes the kernel
or public API.

## File map

The benchmark entry points are:

- `benchmarks/bench_sm90_push_megamoe.py`: authoritative single-run
  whole-layer benchmark. It measures BF16 input through dispatch, activation
  quantization, FC1, SwiGLU, FC2, combine, and BF16 output.
- `benchmarks/sm90_push_megamoe_quick/run_operator_quick.sh`: short
  counterbalanced baseline/all-features A/B.
- `benchmarks/sm90_push_megamoe_quick/run_profile_quick.sh`: short stage and
  CUDA-kernel profile.
- `benchmarks/sm90_push_megamoe_quick/run_vllm_serve_quick.sh` and
  `run_sglang_serve_quick.sh`: clients for servers that are already running.
- `benchmarks/sm90_push_megamoe_quick/route_balance.py`: synthetic routing,
  traffic, and symmetric-window estimator.
- `benchmarks/sm90_push_megamoe_quick/sample_gpu_memory.py`: `nvidia-smi`
  memory sampler.
- `benchmarks/sm90_push_megamoe_quick/summarize.py`: JSON/JSONL report
  generator.

Longer acceptance scripts live under `scripts/`:

- `scripts/run_sm90_push_validation.sh`: complete correctness, graph, soak,
  benchmark, and trace gate.
- `scripts/run_sm90_push_feature_matrix.sh`: five-tier counterbalanced
  baseline/dedup/grouped/fused performance matrix.
- `scripts/run_sm90_push_fc1_epilogue_comparison.sh`: focused fused-FC1
  on/off and small-token break-even scan.

The backend design and correctness contracts are documented in
`docs/sm90_push_megamoe.md`.

## Requirements

- Linux with one or more visible Hopper/SM90 GPUs;
- the FlashInfer checkout installed in development mode;
- initialized submodules and a working CUDA/nvcc toolchain;
- `torchrun` plus NCCL for EP > 1;
- enough free memory for the selected shape (`DSV3` is the largest preset).

Run all commands from the repository root. The scripts create their output
directory and isolate the DeepGEMM cache where appropriate.

The suite has three levels:

1. `run_operator_quick.sh`: baseline versus
   `dedup + grouped combine + fused FC1` at decode-like and prefill-like token
   counts.
2. `run_profile_quick.sh`: one short PyTorch/NVTX stage profile.
3. `run_vllm_serve_quick.sh` / `run_sglang_serve_quick.sh`: online serving
   TTFT, TPOT, ITL, E2E latency, throughput, and goodput against an already
   running server.

`route_balance.py` reports synthetic tokens-per-expert imbalance, estimated
remote wire bytes, and symmetric-window size. `summarize.py` combines all
recognized JSON/JSONL results into `summary.md`.

## Recommended first run

Start with the EP4 operator E2E smoke:

```bash
SM90_QUICK_NGPU=4 \
SM90_QUICK_TOKENS="64 2048" \
SM90_QUICK_ROUTINGS="random hot all_remote" \
bash benchmarks/sm90_push_megamoe_quick/run_operator_quick.sh \
  results/sm90_push_ep4_quick
```

Inspect:

- `results/sm90_push_ep4_quick/summary.md`;
- `operator_results.jsonl` for machine-readable records;
- `logs/` for each individual invocation;
- `route_*.json` for estimated load balance and wire volume.

The headline merge comparison should use eager SM90 push versus the eager
same-run NCCL E2E twin. CUDA Graph rows are correctness and graph-latency
checks; do not present graph-push versus eager-NCCL as an apples-to-apples
headline speedup.

## Direct whole-layer E2E

For one explicit EP4 run with all three features:

```bash
torchrun --standalone --nproc-per-node=4 \
  benchmarks/bench_sm90_push_megamoe.py \
  --config SMALL --tokens 2048 --routing random \
  --dedup --grouped-combine --fuse-fc1 --nccl-baseline \
  --warmup 10 --iters 50 \
  --assert-cos-min 0.997 --assert-growth-max 1.25 \
  --json results/sm90_push_ep4.jsonl
```

Repeat with `--tokens 64` and routing `hot` / `all_remote`. For the final
target-shape check:

```bash
torchrun --standalone --nproc-per-node=8 \
  benchmarks/bench_sm90_push_megamoe.py \
  --config DSV3 --tokens 2048 --routing random \
  --dedup --grouped-combine --fuse-fc1 --nccl-baseline \
  --warmup 10 --iters 50 \
  --assert-cos-min 0.997 --assert-growth-max 1.25 \
  --json results/sm90_push_ep8_dsv3.jsonl
```

Use `--preflight-ab` to append a 30-iteration checked-versus-trusted offsets
micro-measurement. It verifies that the internal trusted path avoids the
public checked-path preflight launch without changing output.

## 1. Operator/protocol quick A/B

On a Linux Hopper node:

```bash
SM90_QUICK_NGPU=4 \
bash benchmarks/sm90_push_megamoe_quick/run_operator_quick.sh results/sm90_quick
```

Defaults:

- EP4, `SMALL`;
- T=64 and T=2048;
- random and full-hot routing;
- 3 warmups, 10 timed rounds;
- two mirrored passes (`A,B` then `B,A`);
- NCCL e2e/transport twins enabled when EP > 1.

Useful overrides:

```bash
# Fastest smoke: one order, one token count, one routing.
SM90_QUICK_PASSES=1 \
SM90_QUICK_TOKENS="64" \
SM90_QUICK_ROUTINGS="random" \
bash benchmarks/sm90_push_megamoe_quick/run_operator_quick.sh results/smoke

# Include all-remote and CUDA Graph replay.
SM90_QUICK_ROUTINGS="random hot all_remote" \
SM90_QUICK_GRAPH=1 \
bash benchmarks/sm90_push_megamoe_quick/run_operator_quick.sh results/graph

# Final topology spot-check. This is still shorter than the full validation.
SM90_QUICK_NGPU=8 \
SM90_QUICK_CONFIG=DSV3 \
SM90_QUICK_TOKENS="64 2048" \
bash benchmarks/sm90_push_megamoe_quick/run_operator_quick.sh results/ep8_dsv3
```

The generated `summary.md` reports:

- baseline/optimized MoE-layer p50 speedup;
- optimized p99 and correctness cosine/growth;
- push and NCCL transport ratios;
- routing-estimated aggregate payload GB/s;
- expert/rank load imbalance;
- estimated dedup/grouped wire and window reductions.

Ten timed rounds make p99 a smoke signal (effectively near the maximum), not
a publication-quality tail estimate. Use the full feature matrix and hundreds
of samples for final numbers.

## 2. Short stage profile

```bash
SM90_QUICK_NGPU=4 \
bash benchmarks/sm90_push_megamoe_quick/run_profile_quick.sh results/profile
```

`profile.log` contains GPU time attributed to:

`begin_round`, `dispatch`, `wait_prefix`, `compact`, `fc1`, `act_quant`,
`fc2`, `combine`, `wait_combine`, `reduce`, and `ack`.

For compute/communication overlap, run the same benchmark under Nsight Systems
with CUDA and NVTX tracing. The current backend is a single-stream stage chain,
so do not infer overlap from aggregate transport time alone.

## 3. Peak GPU-memory sampling

Sample an already running server for 30 seconds while a load client runs in a
different terminal:

```bash
python benchmarks/sm90_push_megamoe_quick/sample_gpu_memory.py \
  --duration-s 30 --interval-ms 100 \
  --output results/serving/memory_sm90_push.json
```

Or wrap a diagnostic command:

```bash
python benchmarks/sm90_push_megamoe_quick/sample_gpu_memory.py \
  --output results/profile_memory.json -- \
  bash benchmarks/sm90_push_megamoe_quick/run_profile_quick.sh results/profile
```

This uses whole-GPU `nvidia-smi` memory, including the model, KV cache, and
other processes. Run it separately from publication timing. The synthetic
route report gives the protocol symmetric-window component; neither value is
a DeepGEMM-workspace-only measurement.

## 4. vLLM serving quick benchmark

Start the model server separately with either the baseline EP backend or the
SM90 push backend. Use identical model, scheduler, TP/EP, CUDA Graph, memory,
and sampling settings for both runs.

These wrappers do not enable or select the SM90 backend. Confirm from the
server configuration/logs that requests actually use `sm90_push`; otherwise
the serving result does not validate this implementation.

Run against the baseline server:

```bash
SM90_SERVE_MODEL=/path/to/model \
SM90_SERVE_VARIANT=nccl_ep \
SM90_VLLM_BASE_URL=http://127.0.0.1:8000 \
bash benchmarks/sm90_push_megamoe_quick/run_vllm_serve_quick.sh \
  results/serving
```

Restart with SM90 push enabled, then reuse the directory:

```bash
SM90_SERVE_MODEL=/path/to/model \
SM90_SERVE_VARIANT=sm90_push \
SM90_VLLM_BASE_URL=http://127.0.0.1:8000 \
bash benchmarks/sm90_push_megamoe_quick/run_vllm_serve_quick.sh \
  results/serving
```

The wrapper uses `vllm bench serve`, requests p50/p90/p99 for TTFT, TPOT,
ITL, and request E2E latency, and records goodput with default SLOs
TTFT < 500 ms and TPOT < 15 ms.

## 5. SGLang serving quick benchmark

For a native SGLang server:

```bash
SM90_SERVE_MODEL=/path/to/model \
SM90_SERVE_VARIANT=nccl_ep \
SM90_SGLANG_BASE_URL=http://127.0.0.1:30000 \
bash benchmarks/sm90_push_megamoe_quick/run_sglang_serve_quick.sh \
  results/serving

# Restart the server with SM90 push enabled.
SM90_SERVE_VARIANT=sm90_push \
SM90_SGLANG_BASE_URL=http://127.0.0.1:30000 \
bash benchmarks/sm90_push_megamoe_quick/run_sglang_serve_quick.sh \
  results/serving
```

Detailed per-request arrays are saved. `summarize.py` derives p50/p90/p99 and,
when SGLang does not emit goodput directly, computes it from the same TTFT and
TPOT SLOs.

## Serving defaults and overrides

The wrappers run two short scenarios:

- prefill: input 2048, output 32, concurrency 8, 40 prompts, 4 req/s;
- decode: input 256, output 64, concurrency 16, 80 prompts, 16 req/s.

Common overrides:

```bash
SM90_SERVE_SCENARIOS="decode"
SM90_SERVE_DECODE_CONCURRENCY=32
SM90_SERVE_DECODE_PROMPTS=160
SM90_SERVE_DECODE_REQUEST_RATE=32
SM90_SERVE_TTFT_SLO_MS=500
SM90_SERVE_TPOT_SLO_MS=15
```

For stable serving p99/goodput, use at least five times as many prompts as the
maximum concurrency and repeat at several offered request rates. These quick
defaults are intended to detect regressions, not establish capacity curves.

## Standalone route/load estimator

No GPU is required:

```bash
python benchmarks/sm90_push_megamoe_quick/route_balance.py \
  --config DSV3 --tokens 2048 --ep-size 8 --routing all_remote \
  --output results/route_ep8.json
```

The byte calculation intentionally excludes protocol control cells, GEMM
workspace, weights, and framework/KV-cache memory. Treat it as a routing-aware
traffic estimate; GPU measurements remain authoritative.

## Full acceptance

Quick results do not replace:

```bash
SM90_PUSH_GATE_REQUIRE_GPUS=8 \
bash scripts/run_sm90_push_validation.sh results/sm90_push_acceptance
```

The full gate must pass EP1/2/4/8 correctness, graph replay, failure checks,
soak, DSV3 random/hot/all-three, and the NCCL comparison without skips.
