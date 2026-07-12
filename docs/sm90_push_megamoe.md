# SM90 Push-Based MegaMoE Expert Parallelism

Design document for the `sm90_push` whole-layer EP backend (single-node
NVLink, Hopper): the push protocol, its three switchable optimizations
(dispatch dedup, grouped combine, fused FC1 epilogue), the correctness
contracts they are held to, and how to validate them.

## 1. What this is

A push-based MoE expert-parallel pipeline for SM90. Each rank owns a
symmetric window (peer-mapped VMM allocation) and one forward runs, in
stream order with zero host synchronization:

```
bump_tag -> wait_acks -> [dedup dispatch: count -> reserve -> store]
  -> wait_prefix -> compact -> FC1 (+ fused SwiGLU/1x128-quant epilogue)
  -> FC2 -> grouped combine -> wait_combine -> reduce (BF16 out) -> ack
```

- Dispatch quantizes BF16 activations 1x128 to FP8 while writing into the
  destination's window; there is no staging copy. Compute is DeepGEMM
  grouped FP8-blockscale GEMM. The whole forward is CUDA-graph capturable.
- The protocol uses 8-byte `{tag<<32|value}` cells with
  `st.release.sys` / `ld.acquire.sys` publish/acquire chains. The actual
  row count of a round lives only in a device scalar (`m_dev`); the host
  never learns it.

## 2. Public surface

The one supported integration path (everything else is internal):

```python
from flashinfer.moe_ep import (
    BootstrapConfig, FleetParams, MoEEpLayer, MoEEpTensors, Sm90PushEpConfig,
)
from flashinfer.fused_moe import MoEConfig, MoEWeightPack, ...

pack = MoEWeightPack()
pack.prepare_for(                       # load time, like other backends
    Sm90PushEpConfig.WEIGHT_VIEW_KEY,   # "sm90_push_fp8_block"
    Sm90PushEpConfig.prepare_weights(w13_bf16_local, w2_bf16_local),
)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(world_size=ep, rank=rank),
    fleet_params=FleetParams(num_experts=E, max_tokens_per_rank=T_cap,
                             token_hidden_size=H),
    backend=Sm90PushEpConfig(),         # whole-layer backend
    compute_config=MoEConfig(routing=..., quant=QuantConfig(DeepSeekFp8),
                             experts=ExpertConfig(intermediate_size=I)),
    weights=pack,
)
out_bf16 = layer(MoEEpTensors(hidden_states=x_bf16, topk_ids=ids_i32,
                              topk_weights=w_f32))
```

`Sm90PushEpConfig` selects a whole-layer backend: `MoEEpLayer` creates a
private `_Sm90PushEpBackend` and never builds a Fleet/Handle; the split
backends (nccl_ep / nixl_ep) keep their dispatch -> compute -> combine path
untouched. `_Sm90PushPipe` / `_Sm90PushMoERunner` are internal executors and
not exported; the low-level test suite and benchmark import them from
`flashinfer.fused_moe.sm90_push_a2a` directly.

Hard constraints raise instead of falling back, since the backend is
selected explicitly: SM90 only (MIG slices rejected), single-node NVLink
peer group with verified peer access and native system-scope atomics
(`cudaDeviceGetP2PAttribute`, queried through the A2A module),
`ep_size <= 32`, FP8-blockscale quant config, SwiGLU activation, weight
view present with a layout tag matching `fuse_fc1_epilogue`. A split
backend given `compute_config`/`weights` also raises; they would otherwise
be silently ignored.

Defaults: `Sm90PushEpConfig()` enables all three optimizations
(`dedup_dispatch`, `grouped_combine`, `fuse_fc1_epilogue`), the
configuration the validation suite gates hardest. Each switch can be
disabled independently for comparison runs and debugging.

Placement and execution knobs:

- `ep_group`: an explicit `torch.distributed` process group for DP x EP
  deployments. Every collective the backend issues (fingerprint, staged
  readiness reports, window handle exchange, destroy barrier) runs on this
  group; `bootstrap.world_size` / `bootstrap.rank` are the group-local
  size/rank. `None` uses an independent single-rank backend when
  `bootstrap.world_size == 1`, and the default group otherwise.
- `init_timeout_s`: timeout applied to torch-distributed initialization
  collectives (default 600 seconds).
- `device_index`: CUDA device for this rank
  (`None` = `torch.cuda.current_device()`).
- Streams: the pipeline launches on the current torch stream; a non-default
  `bootstrap.stream` handle is rejected rather than silently ignored. Run
  the layer inside `with torch.cuda.stream(...)` to place it. Explicit
  raw-stream plumbing is a documented deferral.
- `zero_copy_output`: `forward` returns a fresh tensor by default (safe to
  hold across forwards); opt in to the persistent-buffer view for
  graph/perf paths, where the next forward overwrites it.

nn.Module lifecycle contract (enforced by
`tests/moe/test_moe_ep_sm90_push.py`):

- Eager, collective construction: `MoEEpLayer.__init__` runs the whole
  staged-readiness protocol of section 5. Construct on every EP rank
  together; the first forward is CUDA-graph capturable (no lazy compile
  remains).
- One in-flight forward per layer (single-stream protocol); a concurrent
  second call raises.
- Prepared weight views are registered as module buffers
  (`sm90_push_w13_fp8`, ..., visible to `state_dict()`); the w13 layout tag
  is a persistent scalar bool buffer. A checkpoint whose tag is absent,
  malformed, or mismatches the frozen buffer is rejected before any
  weight is copied. `.to()` / `_apply` calls that would change device,
  dtype, or storage are rejected before any registered buffer is changed.
- Inputs must be finite: non-finite activations/weights propagate through
  the FP8 quantization (`FLASHINFER_VALIDATE_INPUTS=1` adds a host-sync
  debug check at the boundary). Combine slots nobody wrote in a round carry
  scale 0 and are skipped by the reduce, so stale bytes cannot pollute live
  output.
- `destroy()` quiesces first (device synchronize + EP-group barrier, so no
  local kernel and no peer round can still touch the window) before
  releasing references; idempotent, collective like construction.

## 3. Dtype policy

| Stage | Dtype | Nature |
|---|---|---|
| Public input activations | BF16 | native; FP8 1x128 quant happens inside dispatch, writing directly into the symmetric window |
| Runtime compute | FP8 block-scale (128x128 weights, 1x128 activations) | native (DeepGEMM) |
| BF16 weights | converted at load time (`prepare_weights`) | the only public checkpoint format |
| MXFP8 / NVFP4 | unsupported | no packed-checkpoint loader or native SM90 compute exists; future support requires load-time conversion or emulation |
| Public output | BF16, written by the reduce kernel directly (RN of the fp32 accumulation; no extra cast pass) | native output contract |
| Debug/reference output | FP32 (`out_dtype=torch.float32`, the low-level default) | anchors every bit gate |

The reduce-output dtype only affects each rank's own inbox reduction, but
the construction handshake still enforces it, like every other constructor
argument, to be identical across ranks -- one supported configuration per
job. The public BF16 output equals `RN_bf16(fp32_result)` bit for bit
(enforced by `test_moe_ep_sm90_ep1_forward_bf16`).

## 4. The three optimizations (default-on, independently switchable)

**Dispatch dedup** (`dedup_dispatch`). Payload+scale rows are stored once
per (source token, destination rank); meta stays one record per route
(`SlotMeta {src_token, packed(src_rank, k), weight, payload_slot}`, rank
<= 31, k <= 7). Dual pools: payload `ep*token_capacity` rows, meta
`ep*token_capacity*top_k` records, reserved by one packed 64-bit system
atomic. Compact still emits one output row per route, gathered through
`payload_slot`. Bit-exact with the per-route path (hard-gated
`torch.equal`). Cuts dispatch P2P bytes by `1 - E[distinct ranks]/top_k`
(~43% at K=6/EP4 random routing).

**Grouped combine** (`grouped_combine`, FP8 combine only). The owner rank
pre-reduces all of a token's routes that landed on it in fp32, then
quantizes the sum once and pushes one row per (token, owner); the combine
inbox re-keys from `[token_capacity][top_k]` to `[token_capacity][ep_size]`
and the completion chain counts groups. Not bit-comparable with per-route
(the quantization point moves), so it is gated oracle-relative:
`err(grouped) <= 1.05 * err(per-route)` plus cosine >= 0.997; grouped
eager-vs-graph-replay stays bitwise (in-group rows are sorted by route k
before the fp32 chain).

**Fused FC1 epilogue** (`fuse_fc1_epilogue`). A DeepGEMM variant
(`GroupedWithOffsetFc1Fused`) runs gate/up back-to-back K sweeps per
(m_block, pair) over gate/up-interleaved weights and applies SwiGLU + the
1x128 quant in the epilogue, emitting `a2`/`sfa2` directly; the bf16 `h`
tensor is never written, and never allocated in fused mode (~1 GiB/rank at
DSV3/EP8). Bit-exact with the unfused path by contract, including the
cross-TU division-instruction contract: the fused-activation kernel's
translation unit is compiled with `-use_fast_math` (`div.approx.f32`), so
the epilogue's three divisions use `__fdividef` explicitly. Residual `.ftz`
caveat: bf16 has subnormals, so FTZ-independence is not claimed; the
enforced contract is the a2/sfa2 + e2e bit gates. Known trade-off: no
small-M swapAB tactic, so at small token counts per rank the two-pass K
sweep can lose to the unfused path; measure the break-even token count for
the deployment shape and keep the switch off below it.

Orthogonality: dedup changes only where dispatch payload bytes are stored
(compact output bit-identical); fusion changes only the FC1->activation
boundary (a2/sfa2 bit-identical); grouped changes only combine/reduce
(consumes y/meta, both invariant under the other two). Each flag's own gate
therefore covers the compositions; composition gates additionally run all 8
combinations bitwise along the fusion axis and the dedup+grouped+fused
configuration collectively.

## 5. Layout, fingerprint, lifecycle

- 21 layout scalars (`LAYOUT_PARAMS` == `_Sm90PushPipe._layout_args()`),
  window regions: dual pools, base/count/cdone/ack cells, combine/cfp8/csc.
- Staged-readiness construction, the distributed failure model. The
  communicator is built first, because it is the failure-reporting channel;
  the only errors that may raise before the first status collective are
  comm construction/topology mismatches themselves (that rank never entered
  a collective) and pure-argument validation (every rank holds the same
  arguments, so those raise identically everywhere). Every rank-asymmetric
  hazard then runs in a phase: execute locally under try/except, then an
  unconditional status allgather. Every rank executes the same collective
  sequence regardless of local success, so a rank failing JIT/OOM/probing
  cannot strand peers inside a collective it skipped; on any failure all
  ranks raise the same aggregated per-rank report. Phases, in order:
  1. `validate`: the construction handshake agrees on (world, rank,
     argument fingerprint) -- every constructor argument, including
     `fuse_fc1_epilogue` and the reduce out-dtype -- before any window
     allocation, since mismatched window sizes could otherwise fail or
     hang inside the cuMem mapping collectives. World and rank are
     derived from the communicator, not trusted from the caller; the
     window layout is a pure function of the fingerprinted arguments. A
     separate guarded phase then probes the device (CUDA present, SM90,
     MIG rejection).
  2. `a2a-jit`: the A2A protocol module JIT (`build_and_load`), a
     rank-asymmetric failure source, plus hostname/device topology info.
  3. `peer-topology`: same-node check (hostname agreement), per-peer
     `torch.cuda.can_device_access_peer`, and NVLink-native system-scope
     atomics via the module's `sm90_push_p2p_native_atomics` helper
     (`cudaDeviceGetP2PAttribute`; torch does not expose it).
  4. `window+scratch`: symmetric window mapping (only reached when every
     rank reported all-green, so the handle-exchange collectives can never
     be entered against an already-failed peer), views, persistent scratch,
     one-time whole-window zero + synchronize; the phase's closing
     allgather doubles as the barrier that orders the zero against every
     peer's first dispatch.
  5. `gemm-resources` (in `_Sm90PushMoERunner.__init__`): private stateful
     DeepGEMM runner, capacity buffers, workspace freeze, and the
     precompile of every GEMM key the mode uses (fused FC1 + FC2, or plain
     FC1 + FC2), reported by one more unconditional allgather.

  After construction, nothing on the forward path triggers nvcc, workspace
  reconfiguration, tensor allocation, or host-side shape reads; this is the
  prerequisite for CUDA-graph capture.
- Protocol memory model (PTX). The formal correctness argument lives here;
  the header carries only the per-helper invariant:
  - All protocol cells are 8-byte `{tag<<32 | value}` words: one
    `st.release.sys` publishes value and tag atomically with respect to
    readers, so no ordering between two separate stores is ever relied on.
  - Every publish is a release that follows the payload stores in program
    order; every consumer starts with an acquire of the same cell
    (`ld.acquire.sys`) before touching payload. Release/acquire pairs on
    the same address establish the happens-before edge for the payload
    bytes across GPUs (NVLink-native system scope).
  - The explicit `"memory"` clobbers are part of the argument: the PTX
    orders the memory system, the clobber stops the C++ compiler from
    reordering ordinary payload loads/stores across the asm.
  - Completion counting: each `done[key]` update is a system-scope atomic
    RMW on the destination's window; the last writer (RMW returns
    `expected - 1`) performs `__threadfence_system()` then the release
    publish, so a consumer that acquires the publish observes every
    contributor's payload, not just the last writer's.
  - Tags are free-running uint32 round numbers from a device counter
    (bumped at round start; the one-time window zero doubles as the round-0
    ack). All tag checks are equality-based, so uint32 wraparound is safe;
    tag reuse after 2^32 rounds per pipe is the documented limit.
  - The wait loops poll with `ld.acquire.sys` + `__nanosleep`, and every
    wait carries a ~300s clock64 timeout trap so a deadlock surfaces as a
    device trap instead of a silent hang.
- Round counter: increments as uint32 (wraparound well-defined; `int32 += 1`
  would be UB at 2^31); every tag check is equality-based, so wrap is safe.
  Tag reuse after 2^32 rounds per pipe remains undisambiguated (documented
  limit, section 8).
- Malformed device-side inputs trap in-kernel with printf context:
  nonzero-start/decreasing/over-capacity offsets in the fused kernel's
  prologue and, for checked plain grouped GEMM calls, in a small same-stream
  `moe_offsets_preflight_kernel` launched by the binding right before the
  GEMM (graph-safe, no host sync). The internal pipeline skips that launch
  only after its wait-prefix phase has produced the offsets.
  Host-side FFI checks are factored into a strict validator shared by the
  fused and plain `moe_gemm` bindings and cover everything checkable
  without a stream sync (contiguity/device/rank/dtype, positive N/K,
  offsets typing, frozen group/N/K and workspace contracts, TMA
  A/D-declaration rows, padded-stride and scale/B capacities, required
  scales).
- `flashinfer.fused_moe.api` deliberately carries only the surface this
  backend consumes (component configs + `MoEConfig` + `MoEWeightPack`); no
  backend-candidate lists, execution knobs, or capability predicates.
  Those land with the backends that actually consume them.

## 6. Verification inventory

- `tests/moe/test_sm90_push_megamoe.py`: the low-level protocol/kernel
  suite. EP1 gates (bitwise where bit-provable: payload, fused-act, dedup,
  fusion on/off across all four dedup x grouped states, graph replay;
  oracle-relative for grouped), single-feature gates including adversarial
  scale-dilution, K=2/6/8, DSV3 shape, expected_m<32 swapAB corner, FFI
  negative suites for both the fused and the plain `moe_gemm` bindings,
  device-trap subprocess tests (pool overflow, out-of-range expert ids, and
  decreasing offsets against both the fused prologue and the plain
  preflight kernel), fingerprint-mismatch gates, dist gates (uneven/empty
  ranks, skew, graph replay, 200-round soaks with per-round oracle checks:
  baseline, grouped, dedup+grouped, dedup+grouped+fused). The a2/sfa2 stage
  gate sentinel-poisons every output buffer (0x7F bytes, live + padding)
  and asserts padding stays poisoned, so under-writes and padding
  over-writes cannot hide as zero==zero.
- `tests/moe/test_moe_ep_sm90_push.py`: the public-entry suite.
  MoEEpLayer/MoEWeightPack construction, BF16 output contract (bitwise
  RN-of-fp32), staging rejection tests, public graph replay, output
  non-aliasing (fresh-copy default), the nn.Module contract (state_dict
  buffers; device/dtype moves rejected), dist forward.
- `benchmarks/bench_sm90_push_megamoe.py`: e2e + transport + NCCL baselines
  (hard-failing when requested but unavailable). The e2e twin runs real
  compute; the transport twin is a wire-only lower bound (it moves the
  bytes but does none of the compact/combine/reduce work).
  Provenance-stamped JSONL (case_id / run_id / git+source hash / GPU info;
  NaN -> null).
- `benchmarks/sm90_push_megamoe_quick/README.md`: file map, prerequisites,
  EP4/EP8 command lines, output layout, serving caveats, and the path from a
  short smoke run to the full acceptance gate.

## 7. Validation

The validation suite is EP1/2/4/8 correctness (both pytest suites, with
skips treated as failures), 200-round soaks per feature combination
(`SM90_PUSH_SOAK_ROUNDS`), the feature-combination bench matrix with
same-run NCCL rows (a requested `--nccl-baseline` that cannot initialize
exits non-zero), and DSV3-shape steps at EP8, driven through `tests/moe/`
and `benchmarks/bench_sm90_push_megamoe.py`. A compute-sanitizer (memcheck
+ racecheck) run of a representative dedup+grouped+fused EP4 config
complements the suite: the protocol relies on device-scope release/acquire
chains and must be sanitizer-clean at the tool's supported scope.

## 8. Known risks / open items

- The `__fdividef` cross-TU contract has a complete static argument
  (instruction-level equivalence + structural exclusion); the a2/sfa2 and
  e2e bit gates on Hopper hardware are the enforced proof.
- Small-M regression risk for `fuse_fc1_epilogue` (no swapAB form): keep
  the switch off below the measured break-even token count.
- `.ftz` residual: the fused and unfused paths' BF16 subnormal flush
  semantics may differ across the two translation units, so the bit-exact
  guarantee is scoped to the normal-value regimes the gates exercise
  (randn-scaled inputs). If a subnormal-heavy input regime appears, add a
  subnormal case to the a2/sfa2 gate and align the compile flags first.
- `capacity_factor < 1` sizes the local GEMM buffers back up to the frozen
  TMA declaration (descriptor-safe reads); the symmetric-window pools keep
  the reduction.
- Round tags increment as uint32 (well-defined wrap; equality-based
  checks), but tag reuse after 2^32 forward passes on one pipe is not
  disambiguated.
- The DeepGEMM JIT cache key includes a hash of the in-tree DeepGEMM source
  headers; `kFc1FusedKernelVersion` remains an additional variant salt.
- Raw-stream plumbing (`bootstrap.stream`) is deferred; a non-default
  handle is rejected rather than ignored (run under
  `torch.cuda.stream(...)` instead).
