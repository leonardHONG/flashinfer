"""SM90 push-based MegaMoE benchmark + hard performance/correctness gate."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

CONFIGS = {
    # hidden/intermediate multiples of 128; token_capacity defaults to tokens
    "TINY": dict(tokens=256, hidden_size=1024, intermediate=1024, experts=8, top_k=2),
    "SMALL": dict(
        tokens=2048, hidden_size=4096, intermediate=2048, experts=32, top_k=6
    ),
    "DSV3": dict(
        tokens=2048, hidden_size=7168, intermediate=2048, experts=256, top_k=8
    ),
}


@dataclass
class Result:
    mode: str
    ep: int
    rank: int
    config: dict = field(default_factory=dict)
    payload_dtype: str = ""
    combine_dtype: str = ""
    routing: str = ""
    graph: bool = False
    dedup: bool = False
    grouped_combine: bool = False
    fuse_fc1: bool = False
    run_id: str = ""
    case_id: str = ""
    warmup: int = 0
    iters: int = 0
    git_commit: str = ""
    git_dirty: bool = False
    source_hash: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpu_name: str = ""
    cos: float = float("nan")
    err_ratio: float = float("nan")
    growth: float = float("nan")
    p50_ms: float = float("nan")
    p99_ms: float = float("nan")
    baseline_p50_ms: float = float("nan")
    speedup_p50: float = float("nan")


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def collect_provenance(dev) -> dict:
    """Per-record run provenance (commit, dirty flag, source hash, stack)."""
    root = Path(__file__).resolve().parents[1]

    def _git(*args) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    h = hashlib.sha256()
    for rel in (
        "benchmarks/bench_sm90_push_megamoe.py",
        "flashinfer/fused_moe/sm90_push_a2a.py",
        "csrc/fused_moe/sm90_push_a2a_ops.cu",
        "include/flashinfer/fused_moe/sm90_push_a2a.cuh",
        "csrc/fp8_blockscale_gemm_sm90_binding.cu",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/scheduler.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/fp8_gemm.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/jit_utils.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/compiler.cuh",
        "csrc/nv_internal/tensorrt_llm/deep_gemm/runtime.cuh",
        "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h",
        "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.cu",
        "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm_kernel.cuh",
    ):
        p = root / rel
        if p.exists():
            h.update(p.read_bytes())
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain", "-uno")),
        source_hash=h.hexdigest()[:12],
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        gpu_name=torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "",
    )


def json_record(rec: dict) -> str:
    """One JSONL line with NaN/Inf mapped to null (bare NaN is invalid JSON)."""

    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        return v

    return json.dumps(clean(rec), allow_nan=False)


def make_weights(E, I, H, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H, generator=g) * (H**-0.5)).to(
        device=device, dtype=torch.bfloat16
    )
    w2 = (torch.randn(E, H, I, generator=g) * (I**-0.5)).to(
        device=device, dtype=torch.bfloat16
    )
    return w13, w2


def make_routing(T, E, K, routing, ep, rank, e_local, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if routing == "hot":
        # FULL hot (the historical definition): every route -> expert 0
        ids = torch.zeros(T, K, dtype=torch.int32)
    elif routing == "hot1":
        # milder 1/K skew: first route hot, the rest random
        ids = torch.randn(T, E, generator=g).topk(K, dim=1).indices.to(torch.int32)
        ids[:, 0] = 0
    elif routing == "all_remote":
        logits = torch.randn(T, E, generator=g)
        if ep > 1:
            logits[:, rank * e_local : (rank + 1) * e_local] = float("-inf")
        ids = logits.topk(K, dim=1).indices.to(torch.int32)
    else:  # random
        ids = torch.randn(T, E, generator=g).topk(K, dim=1).indices.to(torch.int32)
    w = torch.rand(T, K, generator=g) + 0.1
    w = w / w.sum(dim=1, keepdim=True)
    return ids.to(device), w.to(device=device, dtype=torch.float32)


def time_rounds(fn, warmup, iters, world=1):
    """Barrier-aligned per-round times; returns (p50, p99) in ms."""
    dist = None
    if world > 1:
        import torch.distributed as dist  # noqa: PLC0415

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if dist is not None:
            dist.barrier()  # align hosts; previous iter already drained
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    t = torch.tensor(times, dtype=torch.float64)
    if dist is not None:
        dist.all_reduce(t, op=dist.ReduceOp.MAX)  # group round time per iter
    vals = t.sort().values
    p50 = float(vals[len(vals) // 2])
    p99 = float(vals[min(len(vals) - 1, int(len(vals) * 0.99))])
    return p50, p99


def transport_round(mm, x, ids, wts):
    """Protocol-only round: same NVLink traffic, GEMM/activation elided."""
    pipe = mm.pipe
    pipe.proto_begin_round()
    pipe.proto_dispatch(x, ids, wts)
    pipe.proto_wait_prefix()
    pipe.proto_compact(mm.a1, mm.sfa1, mm.meta, mm.row_expert)
    pipe.proto_combine(mm.y, mm.meta)
    pipe.proto_wait_combine()
    out = pipe.proto_reduce(x.shape[0])
    pipe.proto_ack()
    return out


def make_nccl_transport_twin(
    x, ids, world, e_local, token_capacity, K, H, dev, fp8_payload, fp8_combine
):
    """WIRE-ONLY LOWER BOUND: per-route NCCL a2a of the transport bytes; does
    none of the push round's compact/pre-reduce/reduce/ack work, so the
    push-vs-twin ratio is conservative. Byte-matched only with dedup and
    grouped combine off."""
    import torch.distributed as dist

    T = x.shape[0]
    R = T * K
    nkb = H // 128
    pay = H if fp8_payload else 2 * H
    rec_bytes = pay + (nkb * 4 if fp8_payload else 0) + 16  # payload+scales+meta
    back_bytes = (H + nkb * 4) if fp8_combine else 2 * H  # combine partial bytes

    rec = torch.zeros(max(T, 1), rec_bytes, dtype=torch.uint8, device=dev)
    if T > 0:
        if fp8_payload:
            xf = x.float().reshape(T, nkb, 128)
            amax = xf.abs().amax(-1)
            sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
            q = (xf / sc.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
            rec[:, :H] = q.reshape(T, H).view(torch.uint8)
            rec[:, H : H + nkb * 4] = sc.view(torch.uint8).reshape(T, nkb * 4)
        else:
            rec[:, : 2 * H] = x.view(torch.uint8).reshape(T, 2 * H)

    max_recv = world * token_capacity * K
    recv = torch.empty(max_recv * rec_bytes, dtype=torch.uint8, device=dev)
    back_send = torch.zeros(max_recv * back_bytes, dtype=torch.uint8, device=dev)
    back_recv = torch.empty(max(R, 1) * back_bytes, dtype=torch.uint8, device=dev)
    counts_in = torch.empty(world, dtype=torch.int64, device=dev)

    def twin():
        flat = ids.reshape(-1).long()
        dst = torch.where(flat >= 0, flat // e_local, torch.full_like(flat, world))
        order = torch.argsort(dst, stable=True)
        send_counts = torch.bincount(dst, minlength=world + 1)[:world]
        dist.all_to_all_single(counts_in, send_counts)
        sc_l = send_counts.cpu().tolist()  # host sync #1: my split sizes
        rc_l = counts_in.cpu().tolist()  # host sync #2: peer split sizes
        nsend, nrecv = sum(sc_l), sum(rc_l)
        send = rec[order[:nsend] // K].reshape(-1)  # pack rows by destination
        dist.all_to_all_single(
            recv[: nrecv * rec_bytes],
            send,
            output_split_sizes=[c * rec_bytes for c in rc_l],
            input_split_sizes=[c * rec_bytes for c in sc_l],
        )
        dist.all_to_all_single(
            back_recv[: nsend * back_bytes],
            back_send[: nrecv * back_bytes],
            output_split_sizes=[c * back_bytes for c in sc_l],
            input_split_sizes=[c * back_bytes for c in rc_l],
        )

    return twin


def make_nccl_e2e_baseline(
    x,
    ids,
    wts,
    mm,
    pipe,
    world,
    rank,
    e_local,
    token_capacity,
    K,
    H,
    I,
    dev,
    plain_weights,
):
    """NCCL e2e twin: a2a dispatch -> identical quant/activation/GEMM compute
    -> a2a combine -> source-side weighted reduce (bf16 per-route wire)."""
    import torch.distributed as dist

    from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90

    T = x.shape[0]
    module = pipe.module
    flat = ids.reshape(-1).long()
    dst = torch.where(flat >= 0, flat // e_local, torch.full_like(flat, world))
    perm = torch.argsort(dst, stable=True)
    send_counts = torch.bincount(dst, minlength=world + 1)[:world]
    counts_in = torch.empty(world, dtype=torch.int64, device=dev)
    dist.all_to_all_single(counts_in, send_counts.to(dev))
    in_sp = send_counts.tolist()
    out_sp = counts_in.cpu().tolist()
    nsend, Mr = sum(in_sp), sum(out_sp)
    tok_of = torch.arange(T, device=dev).repeat_interleave(K)
    rows = x[tok_of][perm[:nsend]].contiguous()
    sent_exp = flat.to(dev)[perm[:nsend]].contiguous()
    recv_rows = torch.empty(max(Mr, 1), H, dtype=torch.bfloat16, device=dev)
    recv_exp = torch.empty(max(Mr, 1), dtype=torch.int64, device=dev)
    dist.all_to_all_single(recv_exp[:Mr], sent_exp, out_sp, in_sp)
    lexp = (recv_exp[:Mr] - rank * e_local).clamp_(0, max(e_local - 1, 0))
    grp = torch.argsort(lexp, stable=True)
    inv_grp = torch.argsort(grp)
    m_ws = world * token_capacity * K
    m_buf = max((max(Mr, 1) + 127) // 128 * 128, (m_ws + 3) // 4 * 4)
    row_e = torch.zeros(m_buf, dtype=torch.int32, device=dev)
    row_e[:Mr] = lexp[grp].to(torch.int32)
    cnts = torch.bincount(lexp, minlength=e_local)
    offs = torch.zeros(e_local + 1, dtype=torch.int64, device=dev)
    offs[1:] = cnts.cumsum(0)
    pad = (
        (offs[:e_local] + torch.arange(e_local, device=dev, dtype=torch.int64) * 31)
        // 32
        * 32
    ).to(torch.int32)
    P = max((m_ws + e_local * 31) // 32 * 32, 1)
    m_dev = torch.tensor([Mr], dtype=torch.int32, device=dev)
    p_dev = torch.tensor([P], dtype=torch.int32, device=dev)
    a1 = torch.empty(m_buf, H, dtype=torch.uint8, device=dev)
    sfa1 = torch.empty((H // 128) * P + 128, dtype=torch.float32, device=dev)
    h = torch.empty(m_buf, 2 * I, dtype=torch.bfloat16, device=dev)
    a2 = torch.empty(m_buf, I, dtype=torch.uint8, device=dev)
    sfa2 = torch.empty((I // 128) * P + 128, dtype=torch.float32, device=dev)
    y = torch.empty(m_buf, H, dtype=torch.bfloat16, device=dev)
    back = torch.empty(max(nsend, 1), H, dtype=torch.bfloat16, device=dev)
    inv_perm = torch.argsort(perm[:nsend])
    w_of = wts.reshape(-1, 1).float()
    tw13_fp8, tw13_sf, tw2_fp8, tw2_sf = plain_weights
    runner = create_fp8_blockscale_gemm_runner_sm90()
    sz = runner.get_moe_workspace_size(
        token_capacity * K, max(2 * I, H), max(H, I), world, e_local, True, True
    )
    ws = torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8)
    runner.configure_workspace(ws)

    def rnd(ret=False):
        dist.all_to_all_single(recv_rows[:Mr], rows, out_sp, in_sp)
        if Mr > 0:
            xr = recv_rows[:Mr][grp].contiguous()
            module.sm90_quant_grouped(
                a1, sfa1, xr, offs, pad, m_dev, p_dev, row_e, m_buf
            )
            runner.moe_gemm(
                h,
                a1.view(torch.float8_e4m3fn),
                tw13_fp8,
                offs,
                2 * I,
                H,
                sfa1,
                tw13_sf,
                False,
            )
            module.sm90_silu_mul_quant_grouped(
                a2, sfa2, h, offs, pad, m_dev, p_dev, row_e, m_buf
            )
            runner.moe_gemm(
                y,
                a2.view(torch.float8_e4m3fn),
                tw2_fp8,
                offs,
                H,
                I,
                sfa2,
                tw2_sf,
                False,
            )
            yb = y[:Mr][inv_grp].contiguous()
        else:
            yb = y[:0]
        dist.all_to_all_single(back[:nsend], yb, in_sp, out_sp)
        out = (back[:nsend][inv_perm] * w_of).view(T, K, H).sum(1)
        if ret:
            return out
        return None

    rnd._keep = (ws, runner, row_e, offs, pad, m_dev, p_dev, plain_weights)
    return rnd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=list(CONFIGS), default="SMALL")
    ap.add_argument("--tokens", type=int, help="override config token count")
    ap.add_argument(
        "--token-capacity", type=int, help="token capacity (default: tokens)"
    )
    ap.add_argument("--payload-dtype", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--combine-dtype", choices=["fp8", "bf16"], default="fp8")
    ap.add_argument("--no-fuse-act", action="store_true")
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="dedup dispatch: store payload once per (token, dst rank)",
    )
    ap.add_argument(
        "--grouped-combine",
        action="store_true",
        help="rank-group pre-reduced fp8 combine (Sm90PushConfig.grouped_combine; "
        "requires --combine fp8). A/B against the default per-route path; "
        "composes with --dedup.",
    )
    ap.add_argument(
        "--fuse-fc1",
        action="store_true",
        help="fuse the SwiGLU+quant epilogue into the FC1 GEMM "
        "(fuse_fc1_epilogue=True + gate/up-interleaved weights; bit-exact "
        "with the unfused path by contract; composes with --dedup and "
        "--grouped-combine). NOTE: no small-M swapAB tactic -- at small "
        "--tokens (<~512/rank) the two-pass K sweep can LOSE to the unfused "
        "path; A/B both settings before adopting.",
    )
    ap.add_argument(
        "--routing", choices=["random", "hot", "hot1", "all_remote"], default="random"
    )
    ap.add_argument("--graph", action="store_true", help="time CUDA-graph replays")
    ap.add_argument("--preflight-ab", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument(
        "--nccl-baseline",
        action="store_true",
        help="also time the NCCL all_to_all transport twin (EP > 1)",
    )
    ap.add_argument(
        "--nvtx",
        action="store_true",
        help="per-stage NVTX ranges in forward (profiling; splits FC1/FC2)",
    )
    ap.add_argument(
        "--torch-profile",
        action="store_true",
        help="profile the timed e2e rounds with torch.profiler (kineto/CUPTI) "
        "and print rank0 per-stage + per-kernel GPU-time tables; all ranks "
        "exit right after (same discipline as --nsys-capture)",
    )
    ap.add_argument(
        "--nsys-capture",
        action="store_true",
        help="bracket the TIMED e2e rounds with cudaProfilerStart/Stop for "
        "nsys --capture-range=cudaProfilerApi (warmup stays outside)",
    )
    ap.add_argument(
        "--case-id",
        default="",
        help="tag stamped into every record of this invocation (default: a "
        "fresh uuid). Legs of one A/B (or N-tier) comparison should share "
        "one case id and differ only in the flags under test; reporters "
        "group strictly by it, so a full run and a token sweep at the same "
        "shape can never cross-pair.",
    )
    ap.add_argument("--json", type=Path, help="append result records (one JSON/line)")
    ap.add_argument("--assert-cos-min", type=float)
    ap.add_argument("--assert-growth-max", type=float)
    ap.add_argument("--assert-speedup-min", type=float)
    ap.add_argument("--assert-p99-jitter-max", type=float)
    args = ap.parse_args()
    if args.grouped_combine and args.combine_dtype != "fp8":
        raise SystemExit("--grouped-combine requires --combine fp8")
    if args.fuse_fc1 and args.no_fuse_act:
        raise SystemExit(
            "--fuse-fc1 requires the fused activation contract (drop "
            "--no-fuse-act): the FC1 epilogue subsumes the activation kernel"
        )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    comm = None
    nccl_ok = False
    if world > 1:
        import torch.distributed as dist

        try:
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
            nccl_ok = True
        except (ValueError, RuntimeError):
            dist.init_process_group(backend="gloo")
        torch.cuda.set_device(rank)
        from flashinfer.comm.mnnvl import TorchDistBackend

        comm = TorchDistBackend()

    if args.nccl_baseline and world > 1 and not nccl_ok:
        raise SystemExit(
            "--nccl-baseline requested but init_process_group('cpu:gloo,"
            "cuda:nccl') fell back to gloo (no cuda:nccl available) -- fix "
            "the NCCL setup or drop --nccl-baseline"
        )

    ids_box = [
        time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}" if rank == 0 else None,
        (args.case_id or uuid.uuid4().hex[:12]) if rank == 0 else None,
    ]
    if world > 1:
        import torch.distributed as dist

        dist.broadcast_object_list(ids_box, src=0)
    run_id, case_id = ids_box

    from flashinfer.fused_moe.sm90_push_a2a import (
        _Sm90PushMoERunner,
        _Sm90PushPipe,
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushPayload,
        make_sm90_push_weights,
    )
    from tests.moe.reference_moe import reference_moe_fp8_weights_streaming

    cfg = dict(CONFIGS[args.config])
    if args.tokens:
        cfg["tokens"] = args.tokens
    T, H, I, E, K = (
        cfg["tokens"],
        cfg["hidden_size"],
        cfg["intermediate"],
        cfg["experts"],
        cfg["top_k"],
    )
    token_capacity = args.token_capacity or T
    if E % world != 0:
        raise SystemExit(f"experts {E} not divisible by world size {world}")
    e_local = E // world
    dev = torch.device("cuda", rank if world > 1 else 0)

    def build(combine_mode):
        pcfg = Sm90PushConfig(
            payload_dtype=Sm90PushPayload(args.payload_dtype),
            combine_dtype=Sm90PushCombine(combine_mode),
            fuse_act=not args.no_fuse_act,
            dedup_dispatch=args.dedup,
            grouped_combine=args.grouped_combine and combine_mode == "fp8",
            fuse_fc1_epilogue=args.fuse_fc1,
        )
        pipe = _Sm90PushPipe(
            ep_size=world,
            rank=rank,
            num_local_experts=e_local,
            hidden_size=H,
            top_k=K,
            token_capacity=token_capacity,
            device_index=dev.index,
            config=pcfg,
            comm_backend=comm,
        )
        lo, hi = rank * e_local, (rank + 1) * e_local
        mm = _Sm90PushMoERunner(
            pipe,
            make_sm90_push_weights(
                w13[lo:hi], w2[lo:hi], interleave_gate_up=args.fuse_fc1
            ),
        )
        return pipe, mm

    w13, w2 = make_weights(E, I, H, seed=7, device=dev)
    x = torch.randn(
        T, H, generator=torch.Generator(device="cpu").manual_seed(11 + rank)
    ).to(device=dev, dtype=torch.bfloat16)
    ids, wts = make_routing(
        T, E, K, args.routing, world, rank, e_local, seed=13 + rank, device=dev
    )

    meta = dict(
        dedup=args.dedup,
        grouped_combine=args.grouped_combine,
        fuse_fc1=args.fuse_fc1,
        run_id=run_id,
        case_id=case_id,
        warmup=args.warmup,
        iters=args.iters,
        **collect_provenance(dev),
    )

    pipe, mm = build(args.combine_dtype)
    mm.record_stages = args.nvtx
    res = Result(
        mode="e2e",
        ep=world,
        rank=rank,
        config=cfg,
        payload_dtype=args.payload_dtype,
        combine_dtype=args.combine_dtype,
        routing=args.routing,
        graph=args.graph,
        **meta,
    )

    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    if not torch.isfinite(out).all():
        raise SystemExit(f"[rank {rank}] FATAL: NaN/Inf in output")
    ref = reference_moe_fp8_weights_streaming(x, w13, w2, ids, wts)
    nrm = ref.float().pow(2).mean().sqrt().clamp_min(1e-6)
    res.err_ratio = float((out.float() - ref).pow(2).mean().sqrt() / nrm)
    res.cos = float(
        torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    )
    if args.combine_dtype == "fp8":
        _, mm_anchor = build("bf16")
        out_a = mm_anchor(x, ids, wts).clone()
        torch.cuda.synchronize()
        err_a = float((out_a.float() - ref).pow(2).mean().sqrt() / nrm)
        res.growth = res.err_ratio / max(err_a, 1e-12)
    log(
        rank,
        f"correctness: cos={res.cos:.5f} err={res.err_ratio:.4f} "
        f"growth={res.growth:.3f}",
    )

    if args.graph:
        static_x = x.clone()
        static_ids = ids.clone()
        static_wts = wts.clone()
        for _ in range(2):
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                mm(static_x, static_ids, static_wts)
            torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = mm(static_x, static_ids, static_wts)

        x2 = torch.randn(
            T, H, generator=torch.Generator(device="cpu").manual_seed(1011 + rank)
        ).to(device=dev, dtype=torch.bfloat16)
        ids2, wts2 = make_routing(
            T,
            E,
            K,
            args.routing,
            world,
            rank,
            e_local,
            seed=1013 + rank,
            device=dev,
        )
        static_x.copy_(x2)
        static_ids.copy_(ids2)
        static_wts.copy_(wts2)
        g.replay()
        torch.cuda.synchronize()
        graph_out = static_out.clone()
        ref2 = reference_moe_fp8_weights_streaming(x2, w13, w2, ids2, wts2)
        nrm2 = ref2.float().pow(2).mean().sqrt().clamp_min(1e-6)
        graph_err = float((graph_out.float() - ref2).pow(2).mean().sqrt() / nrm2)
        graph_cos = float(
            torch.nn.functional.cosine_similarity(
                graph_out.flatten(), ref2.flatten(), dim=0
            )
        )
        res.err_ratio = max(res.err_ratio, graph_err)
        res.cos = min(res.cos, graph_cos)
        if args.combine_dtype == "fp8":
            out_a2 = mm_anchor(x2, ids2, wts2).clone()
            torch.cuda.synchronize()
            err_a2 = float((out_a2.float() - ref2).pow(2).mean().sqrt() / nrm2)
            graph_growth = graph_err / max(err_a2, 1e-12)
            res.growth = max(res.growth, graph_growth)
        if not torch.isfinite(graph_out).all():
            raise SystemExit(f"[rank {rank}] FATAL: NaN/Inf in graph replay output")
        if args.assert_cos_min is not None and graph_cos < args.assert_cos_min:
            raise SystemExit(
                f"[rank {rank}] graph cos {graph_cos:.5f} < {args.assert_cos_min}"
            )
        if (
            args.combine_dtype == "fp8"
            and args.assert_growth_max is not None
            and graph_growth > args.assert_growth_max
        ):
            raise SystemExit(
                f"[rank {rank}] graph growth {graph_growth:.3f} "
                f"> {args.assert_growth_max}"
            )
        log(
            rank,
            f"graph correctness: cos={graph_cos:.5f} err={graph_err:.4f} "
            f"growth={graph_growth if args.combine_dtype == 'fp8' else float('nan'):.3f}",
        )
        fwd = g.replay
    else:
        fwd = lambda: mm(x, ids, wts)  # noqa: E731
    if args.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        for _ in range(args.warmup):
            fwd()
        torch.cuda.synchronize()
        dist_mod = None
        if world > 1:
            import torch.distributed as dist_mod
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iters):
                if dist_mod is not None:
                    dist_mod.barrier()  # align rounds: wait_prefix must measure
                    # flight/skew of THIS round, not free-running rank drift
                fwd()
            torch.cuda.synchronize()
        if rank == 0:
            print("==== STAGE/KERNEL GPU-TIME TABLE (rank0) ====", flush=True)
            print(
                prof.key_averages().table(
                    sort_by="cuda_time_total", row_limit=60, max_name_column_width=60
                ),
                flush=True,
            )
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
        return

    if args.nsys_capture:
        for _ in range(args.warmup):  # warmup OUTSIDE the capture window
            fwd()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()  # no-op unless nsys is attached
        res.p50_ms, res.p99_ms = time_rounds(fwd, 0, args.iters, world)
        torch.cuda.profiler.stop()
    else:
        res.p50_ms, res.p99_ms = time_rounds(fwd, args.warmup, args.iters, world)
    log(rank, f"e2e: p50={res.p50_ms:.3f} ms  p99={res.p99_ms:.3f} ms")

    if args.preflight_ab:

        def run_fc2(trusted_offsets):
            mm.runner.moe_gemm(
                mm.y,
                mm.a2.view(torch.float8_e4m3fn),
                mm.w2_fp8,
                pipe._offsets,
                pipe.H,
                mm.I,
                mm.sfa2,
                mm.w2_sf,
                trusted_offsets,
            )

        run_fc2(False)
        torch.cuda.synchronize()
        checked_out = mm.y.clone()
        run_fc2(True)
        torch.cuda.synchronize()
        if not torch.equal(checked_out, mm.y):
            raise SystemExit(
                f"[rank {rank}] FATAL: checked/trusted GEMM outputs differ"
            )
        checked_ms, _ = time_rounds(lambda: run_fc2(False), 5, 30, world)
        trusted_ms, _ = time_rounds(lambda: run_fc2(True), 5, 30, world)
        log(
            rank,
            f"preflight A/B: checked={checked_ms:.4f} ms "
            f"trusted={trusted_ms:.4f} ms delta={checked_ms - trusted_ms:.4f} ms",
        )

    if args.nsys_capture:
        if args.json and rank == 0:
            with open(args.json, "a") as f:
                f.write(json_record(asdict(res)) + chr(10))
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
        return

    tres = Result(
        mode="transport",
        graph=False,  # never measured under capture
        ep=world,
        rank=rank,
        config=cfg,
        payload_dtype=args.payload_dtype,
        combine_dtype=args.combine_dtype,
        routing=args.routing,
        **meta,
    )
    tres.p50_ms, tres.p99_ms = time_rounds(
        lambda: transport_round(mm, x, ids, wts), args.warmup, args.iters, world
    )
    log(rank, f"transport: p50={tres.p50_ms:.3f} ms  p99={tres.p99_ms:.3f} ms")

    tw = None
    records_extra = []
    if args.nccl_baseline and world > 1:
        if not nccl_ok:
            raise SystemExit(
                "--nccl-baseline requested but the cuda:nccl process group "
                "is unavailable -- refusing to report without the baseline"
            )
        else:
            twin = make_nccl_transport_twin(
                x,
                ids,
                world,
                e_local,
                token_capacity,
                K,
                H,
                dev,
                fp8_payload=args.payload_dtype == "fp8",
                fp8_combine=args.combine_dtype == "fp8",
            )
            tw = Result(
                mode="transport_nccl",
                graph=False,  # never measured under capture
                ep=world,
                rank=rank,
                config=cfg,
                payload_dtype=args.payload_dtype,
                combine_dtype=args.combine_dtype,
                routing=args.routing,
                **meta,
            )
            tw.p50_ms, tw.p99_ms = time_rounds(twin, args.warmup, args.iters, world)
            twin_kind = (
                "bytes-matched per-route twin"
                if not (args.dedup or args.grouped_combine)
                else "standard per-route NCCL; push moves fewer bytes by design"
            )
            log(
                rank,
                f"transport_nccl [wire-only lower bound]: p50={tw.p50_ms:.3f} ms  "
                f"p99={tw.p99_ms:.3f} ms  -> push transport speedup "
                f"{tw.p50_ms / tres.p50_ms:.3f}x (p50 vs {twin_kind}; twin moves "
                f"the bytes but does NO compact/combine/reduce work and excludes "
                f"source quant)",
            )

    if args.nccl_baseline and world > 1 and nccl_ok:
        lo_, hi_ = rank * e_local, (rank + 1) * e_local
        if args.fuse_fc1:
            from flashinfer.fused_moe import transform_weights_for_sm90_push

            plain_w = transform_weights_for_sm90_push(w13[lo_:hi_], w2[lo_:hi_])
        else:
            plain_w = (mm.w13_fp8, mm.w13_sf, mm.w2_fp8, mm.w2_sf)
        e2n = make_nccl_e2e_baseline(
            x,
            ids,
            wts,
            mm,
            pipe,
            world,
            rank,
            e_local,
            token_capacity,
            K,
            H,
            I,
            dev,
            plain_w,
        )
        out_n = e2n(ret=True)
        torch.cuda.synchronize()
        nres = Result(
            mode="e2e_nccl",
            graph=False,  # never measured under capture
            ep=world,
            rank=rank,
            config=cfg,
            payload_dtype="bf16-wire",
            combine_dtype="bf16-wire",
            routing=args.routing,
            **meta,
        )
        nrm_n = ref.float().pow(2).mean().sqrt().clamp_min(1e-6)
        nres.err_ratio = float((out_n.float() - ref).pow(2).mean().sqrt() / nrm_n)
        nres.cos = float(
            torch.nn.functional.cosine_similarity(out_n.flatten(), ref.flatten(), dim=0)
        )
        nres.p50_ms, nres.p99_ms = time_rounds(e2n, args.warmup, args.iters, world)
        log(
            rank,
            f"e2e_nccl: p50={nres.p50_ms:.3f} ms  p99={nres.p99_ms:.3f} ms  "
            f"cos={nres.cos:.5f} -> push e2e speedup {nres.p50_ms / res.p50_ms:.3f}x "
            f"(p50 vs standard per-route bf16-wire NCCL; baseline metadata "
            f"precomputed = flattered)",
        )
        records_extra.append(asdict(nres))

    if world == 1 and not args.skip_baseline:
        from types import SimpleNamespace

        from tests.moe.sm90_moe_baseline_path import sm90_moe_baseline_local

        inp = SimpleNamespace(
            hidden_states=x, w13=w13, w2=w2, topk_ids=ids, topk_weights=wts
        )
        base_out = sm90_moe_baseline_local(inp)
        cos_b = torch.nn.functional.cosine_similarity(
            base_out.flatten(), ref.flatten(), dim=0
        )
        res.baseline_p50_ms, _ = time_rounds(
            lambda: sm90_moe_baseline_local(inp),
            max(args.warmup // 2, 2),
            max(args.iters // 2, 10),
        )
        res.speedup_p50 = res.baseline_p50_ms / res.p50_ms
        log(
            rank,
            f"baseline: p50={res.baseline_p50_ms:.3f} ms "
            f"(cos={float(cos_b):.5f})  speedup={res.speedup_p50:.3f}x",
        )

    records = [asdict(res), asdict(tres)] + records_extra
    if tw is not None:
        records.append(asdict(tw))
    if world > 1:
        import torch.distributed as dist

        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        records = [r for sub in gathered for r in sub]

    failures = []
    if rank == 0:
        e2e = [r for r in records if r["mode"] == "e2e"]
        worst_cos = min(r["cos"] for r in e2e)
        worst_growth = max(
            (r["growth"] for r in e2e if r["growth"] == r["growth"]), default=None
        )
        worst_jitter = max(r["p99_ms"] / r["p50_ms"] for r in e2e)
        if args.assert_cos_min is not None and worst_cos < args.assert_cos_min:
            failures.append(f"cos {worst_cos:.5f} < {args.assert_cos_min}")
        if args.assert_cos_min is not None:
            for r in records:
                if r["mode"] != "e2e_nccl":
                    continue
                c = r["cos"]
                if c != c or c < args.assert_cos_min:  # NaN cos also fails
                    failures.append(
                        f"e2e_nccl cos {c:.5f} < {args.assert_cos_min} "
                        f"(rank {r['rank']}; NCCL baseline compute broken?)"
                    )
                    break
        if (
            args.assert_growth_max is not None
            and worst_growth is not None
            and worst_growth > args.assert_growth_max
        ):
            failures.append(f"growth {worst_growth:.3f} > {args.assert_growth_max}")
        if args.assert_speedup_min is not None:
            sp = [r["speedup_p50"] for r in e2e if r["speedup_p50"] == r["speedup_p50"]]
            if sp and min(sp) < args.assert_speedup_min:
                failures.append(f"speedup {min(sp):.3f} < {args.assert_speedup_min}")
        if (
            args.assert_p99_jitter_max is not None
            and worst_jitter > args.assert_p99_jitter_max
        ):
            failures.append(
                f"p99/p50 jitter {worst_jitter:.3f} > {args.assert_p99_jitter_max}"
            )
        if args.json:
            with open(args.json, "a") as f:
                for r in records:
                    f.write(json_record(r) + "\n")
        for f_ in failures:
            print(f"GATE FAIL: {f_}", flush=True)
        if not failures:
            print("ALL GATES PASS", flush=True)

    if world > 1:
        import torch.distributed as dist

        n_fail = [len(failures)]
        dist.broadcast_object_list(n_fail, src=0)
        dist.barrier()
        if n_fail[0]:
            sys.exit(1)
    elif failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
