"""

Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Correctness gates for the SM90 push-based MegaMoE (single-node NVLink EP).
"""

import os
import subprocess
import sys

import pytest
import torch

from .reference_moe import dequant_weight_128x128, reference_moe


def _sm90_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.utils import is_sm90a_supported

        return is_sm90a_supported(torch.device("cuda"))
    except Exception:
        return False


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))

requires_sm90 = pytest.mark.skipif(
    not _sm90_available() or _WORLD > 1,
    reason="requires an SM90 (Hopper) GPU outside torchrun",
)

requires_dist = pytest.mark.skipif(
    _WORLD < 2 or not _sm90_available(),
    reason="requires torchrun with WORLD_SIZE >= 2 on SM90 GPUs",
)

# small but non-trivial default shapes (H, I multiples of 128; E > 1)
H, I, E_TOTAL, TOPK, T_CAP = 512, 768, 4, 2, 64

_KEEP_ALIVE = []


def _make_weights(E: int, seed: int, device, h: int = H, i: int = I) -> tuple:
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (torch.randn(E, 2 * i, h, generator=g) * (h**-0.5)).to(
        device=device, dtype=torch.bfloat16
    )
    w2 = (torch.randn(E, h, i, generator=g) * (i**-0.5)).to(
        device=device, dtype=torch.bfloat16
    )
    return w13, w2


def _make_routing(
    T: int,
    E: int,
    K: int,
    seed: int,
    device,
    mode: str = "random",
    rank: int = 0,
    e_local: int = 0,
):
    """mode: random | hot (every route -> expert 0) | hot1 | all_remote | skew."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "hot":
        ids = torch.zeros(T, K, dtype=torch.int32)
    elif mode == "all_remote":
        logits = torch.randn(T, E, generator=g)
        if e_local > 0 and e_local < E:
            logits[:, rank * e_local : (rank + 1) * e_local] = float("-inf")
        ids = logits.topk(K, dim=1).indices.to(torch.int32)
    else:
        logits = torch.randn(T, E, generator=g)
        ids = logits.topk(K, dim=1).indices.to(torch.int32)
    w = torch.rand(T, K, generator=g) + 0.1
    w = w / w.sum(dim=1, keepdim=True)
    return ids.to(device), w.to(device=device, dtype=torch.float32)


def _make_x(T: int, seed: int, device, h: int = H) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(T, h, generator=g).to(device=device, dtype=torch.bfloat16)


def _build(
    payload_dtype="fp8",
    combine_dtype="fp8",
    fuse_act=True,
    capacity_factor=1.0,
    device_index=0,
    ep=1,
    rank=0,
    comm=None,
    e_total=E_TOTAL,
    token_capacity=T_CAP,
    dedup=False,
    grouped_combine=False,
    fuse_fc1_epilogue=False,
    top_k=TOPK,
    h=H,
    i=I,
):
    from flashinfer.fused_moe.sm90_push_a2a import (
        _Sm90PushMoERunner,
        _Sm90PushPipe,
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushPayload,
        make_sm90_push_weights,
        transform_weights_for_sm90_push,
    )

    assert e_total % ep == 0
    e_local = e_total // ep
    cfg = Sm90PushConfig(
        payload_dtype=Sm90PushPayload(payload_dtype),
        combine_dtype=Sm90PushCombine(combine_dtype),
        fuse_act=fuse_act,
        capacity_factor=capacity_factor,
        dedup_dispatch=dedup,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    pipe = _Sm90PushPipe(
        ep_size=ep,
        rank=rank,
        num_local_experts=e_local,
        hidden_size=h,
        top_k=top_k,
        token_capacity=token_capacity,
        device_index=device_index,
        config=cfg,
        comm_backend=comm,
    )
    dev = torch.device("cuda", device_index)
    w13, w2 = _make_weights(e_total, seed=7, device=dev, h=h, i=i)
    lo, hi = rank * e_local, (rank + 1) * e_local
    if fuse_fc1_epilogue:
        mm = _Sm90PushMoERunner(
            pipe,
            make_sm90_push_weights(w13[lo:hi], w2[lo:hi], interleave_gate_up=True),
        )
        fp8_w = transform_weights_for_sm90_push(w13[lo:hi], w2[lo:hi])
    else:
        w13_fp8, w13_sf, w2_fp8, w2_sf = transform_weights_for_sm90_push(
            w13[lo:hi], w2[lo:hi]
        )
        mm = _Sm90PushMoERunner(pipe, w13_fp8, w13_sf, w2_fp8, w2_sf)
        fp8_w = (w13_fp8, w13_sf, w2_fp8, w2_sf)
    _KEEP_ALIVE.append(pipe)
    return pipe, mm, (w13, w2), fp8_w


def _dequant_reference(x, ids, wts, fp8_weights, e_total: int):
    """Oracle on the DEQUANTIZED weights: isolates activation-quant noise."""
    w13_fp8, w13_sf, w2_fp8, w2_sf = fp8_weights
    E = w13_fp8.shape[0]
    assert e_total == E, "dist ranks must pass the gathered full weight set"
    w13d = torch.stack(
        [dequant_weight_128x128(w13_fp8[e], w13_sf[e]) for e in range(E)]
    )
    w2d = torch.stack([dequant_weight_128x128(w2_fp8[e], w2_sf[e]) for e in range(E)])
    return reference_moe(x, w13d, w2d, ids, wts)


def _err(out: torch.Tensor, ref: torch.Tensor) -> float:
    nrm = ref.float().pow(2).mean().sqrt().clamp_min(1e-6)
    return float((out.float() - ref.float()).pow(2).mean().sqrt() / nrm)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
    )


def test_constructor_validation():
    from flashinfer.fused_moe.sm90_push_a2a import _Sm90PushPipe, Sm90PushConfig

    def build(**kw):
        args = dict(
            ep_size=1,
            rank=0,
            num_local_experts=4,
            hidden_size=512,
            top_k=2,
            token_capacity=64,
            device_index=0,
        )
        args.update(kw)
        return _Sm90PushPipe(**args)

    with pytest.raises(ValueError, match="ep_size"):
        build(ep_size=0)
    with pytest.raises(ValueError, match="ep_size"):
        build(ep_size=64)
    with pytest.raises(RuntimeError, match="topology/fingerprint mismatch"):
        build(rank=1)
    with pytest.raises(RuntimeError, match="num_local_experts"):
        build(num_local_experts=0)
    with pytest.raises(RuntimeError, match="hidden"):
        build(hidden_size=500)
    with pytest.raises(RuntimeError, match="top_k"):
        build(top_k=3)
    with pytest.raises(RuntimeError, match="token_capacity"):
        build(token_capacity=0)
    with pytest.raises(RuntimeError, match="capacity_factor"):
        build(config=Sm90PushConfig(capacity_factor=0.0))
    with pytest.raises(RuntimeError, match="capacity_factor"):
        build(config=Sm90PushConfig(capacity_factor=1.5))
    from flashinfer.fused_moe import Sm90PushCombine

    with pytest.raises(RuntimeError, match="grouped_combine"):
        build(
            config=Sm90PushConfig(
                combine_dtype=Sm90PushCombine.BF16, grouped_combine=True
            )
        )
    with pytest.raises(RuntimeError, match="fuse_fc1_epilogue"):
        build(config=Sm90PushConfig(fuse_act=False, fuse_fc1_epilogue=True))


def test_weight_transform_validation():
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13 = torch.zeros(2, 2 * 256, 256, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 256, 256, dtype=torch.bfloat16)
    for fmt in ("int4", "mxfp8", "nvfp4"):
        with pytest.raises(ValueError, match="weight_format"):
            transform_weights_for_sm90_push(w13, w2, weight_format=fmt)
    with pytest.raises(ValueError, match="BF16"):
        transform_weights_for_sm90_push(w13.float(), w2)
    with pytest.raises(ValueError, match="inconsistent"):
        transform_weights_for_sm90_push(
            w13, torch.zeros(2, 128, 256, dtype=torch.bfloat16)
        )


def test_slotmeta_rank_k_pack_roundtrip():
    def pack(rank: int, k: int) -> int:
        return (k << 16) | rank

    for rank in (0, 1, 31):
        for k in (0, 1, 7):
            v = pack(rank, k)
            assert 0 <= v <= 2**31 - 1, "must not wrap into the sign bit"
            assert v & 0xFFFF == rank
            assert v >> 16 == k
    # round-trip through the int32 carrier the pool/meta_out actually use
    t = torch.tensor([pack(31, 7)], dtype=torch.int32)
    v = int(t.item())
    assert v & 0xFFFF == 31 and v >> 16 == 7


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4"])
def test_weight_transform_fold_growth(fmt):
    from flashinfer.fused_moe.sm90_push_a2a import (
        _mxfp8_roundtrip,
        _nvfp4_roundtrip,
        transform_weights_for_sm90_push,
    )

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    N, K, M = 512, 1024, 128
    w = (torch.randn(N, K, device=dev) * K**-0.5).to(torch.bfloat16)
    x = torch.randn(M, K, device=dev)

    roundtrip = _mxfp8_roundtrip if fmt == "mxfp8" else _nvfp4_roundtrip
    w_native = roundtrip(w)

    w_native_bf16 = w_native.to(torch.bfloat16)
    w13 = torch.cat([w_native_bf16, w_native_bf16], dim=0).reshape(1, 2 * N, K)
    w2 = w_native_bf16.T.reshape(1, K, N).contiguous()
    w13_fp8, w13_sf, _, _ = transform_weights_for_sm90_push(w13, w2)
    w_fold = dequant_weight_128x128(w13_fp8[0, :N], w13_sf[0, : N // 128])

    y_ref = x @ w.float().T
    y_native = x @ w_native.T
    y_fold = x @ w_fold.T
    err_native = float((y_native - y_ref).norm() / (y_ref.norm() + 1e-9))
    err_fold = float((y_fold - y_ref).norm() / (y_ref.norm() + 1e-9))
    growth = err_fold / max(err_native, 1e-12)
    flush = float(((w_native != 0) & (w_fold == 0)).float().mean())
    assert growth <= 1.5, f"{fmt}: fold error growth {growth:.3f} > 1.5"
    assert flush < 1e-3, f"{fmt}: flush rate {flush:.2e}"
    assert _cos(y_fold, y_native) > 0.999


@requires_sm90
@pytest.mark.parametrize(
    "payload,combine",
    [("bf16", "bf16"), ("fp8", "bf16"), ("bf16", "fp8"), ("fp8", "fp8")],
)
def test_ep1_forward_configs(payload, combine):
    pipe, mm, _, fp8_w = _build(payload_dtype=payload, combine_dtype=combine)
    dev = pipe.device
    x = _make_x(T_CAP, seed=1, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=2, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert out.shape == (T_CAP, H)
    assert torch.isfinite(out).all(), "NaN/Inf in output"
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    cos = _cos(out, ref)
    err = _err(out, ref)
    assert cos > 0.997, f"{payload}/{combine}: cos {cos:.5f}"
    assert err < 0.10, f"{payload}/{combine}: err_ratio {err:.4f}"


@requires_sm90
def test_ep1_fp8_combine_growth():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=3, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=4, device=dev)

    pipe_a, mm_a, _, fp8_w = _build(payload_dtype="fp8", combine_dtype="bf16")
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    pipe_b, mm_b, _, _ = _build(payload_dtype="fp8", combine_dtype="fp8")
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()

    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    growth = _err(out_b, ref) / max(_err(out_a, ref), 1e-12)
    assert growth <= 1.25, f"fp8 combine error growth {growth:.3f} > 1.25"


@requires_sm90
def test_ep1_payload_bitwise_equal():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=5, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=6, device=dev)
    _, mm_a, _, _ = _build(payload_dtype="bf16", combine_dtype="bf16")
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_b, _, _ = _build(payload_dtype="fp8", combine_dtype="bf16")
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b), "fp8 payload != bf16 payload (bitwise)"


@requires_sm90
def test_ep1_fused_act_bitwise_equal():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=7, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=8, device=dev)
    _, mm_a, _, _ = _build(fuse_act=True)
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_b, _, _ = _build(fuse_act=False)
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b), "fused activation != split pair (bitwise)"


def test_weight_transform_interleave():
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    torch.manual_seed(3)
    E, I_, H_ = 2, 384, 256  # I/128 = 3 blocks: exercises non-trivial order
    w13 = (torch.randn(E, 2 * I_, H_) * H_**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H_, I_) * I_**-0.5).to(torch.bfloat16)

    plain_fp8, plain_sf, w2_fp8_a, w2_sf_a = transform_weights_for_sm90_push(w13, w2)
    il_fp8, il_sf, w2_fp8_b, w2_sf_b = transform_weights_for_sm90_push(
        w13, w2, interleave_gate_up=True
    )

    nb = I_ // 128
    for e in range(E):
        for b in range(nb):
            # interleaved row-block 2b   == gate block b
            assert torch.equal(
                il_fp8[e, (2 * b) * 128 : (2 * b + 1) * 128].view(torch.uint8),
                plain_fp8[e, b * 128 : (b + 1) * 128].view(torch.uint8),
            )
            # interleaved row-block 2b+1 == up block b
            assert torch.equal(
                il_fp8[e, (2 * b + 1) * 128 : (2 * b + 2) * 128].view(torch.uint8),
                plain_fp8[e, (nb + b) * 128 : (nb + b + 1) * 128].view(torch.uint8),
            )
            assert torch.equal(il_sf[e, 2 * b], plain_sf[e, b])
            assert torch.equal(il_sf[e, 2 * b + 1], plain_sf[e, nb + b])
    # w2 is untouched by the flag
    assert torch.equal(w2_fp8_a.view(torch.uint8), w2_fp8_b.view(torch.uint8))
    assert torch.equal(w2_sf_a, w2_sf_b)


@pytest.mark.parametrize("fmt", ["mxfp8", "nvfp4"])
def test_weight_transform_interleave_after_roundtrip(fmt):
    from flashinfer.fused_moe.sm90_push_a2a import (
        _mxfp8_roundtrip,
        _nvfp4_roundtrip,
        transform_weights_for_sm90_push,
    )

    torch.manual_seed(5)
    E, I_, H_ = 1, 256, 256
    w13 = (torch.randn(E, 2 * I_, H_) * H_**-0.5).to(torch.bfloat16)
    w2 = (torch.randn(E, H_, I_) * I_**-0.5).to(torch.bfloat16)
    roundtrip = _mxfp8_roundtrip if fmt == "mxfp8" else _nvfp4_roundtrip
    w13 = roundtrip(w13[0]).unsqueeze(0).to(torch.bfloat16)
    w2 = roundtrip(w2[0]).unsqueeze(0).to(torch.bfloat16)
    plain_fp8, plain_sf, _, _ = transform_weights_for_sm90_push(w13, w2)
    il_fp8, il_sf, _, _ = transform_weights_for_sm90_push(
        w13, w2, interleave_gate_up=True
    )
    nb = I_ // 128
    perm = []
    for b in range(nb):
        perm.extend([b, nb + b])
    perm = torch.tensor(perm)
    # Index bytes because CPU FP8 indexing is unsupported.
    ref_fp8_u8 = (
        plain_fp8.view(torch.uint8)
        .reshape(E, 2 * nb, 128, H_)[:, perm]
        .reshape(E, 2 * I_, H_)
    )
    ref_sf = plain_sf.reshape(E, 2 * nb, H_ // 128)[:, perm].reshape(
        E, 2 * I_ // 128, H_ // 128
    )
    assert torch.equal(il_fp8.view(torch.uint8), ref_fp8_u8)
    assert torch.equal(il_sf, ref_sf)


def _assert_device_trap(r, marker):
    combined = r.stdout + r.stderr
    assert r.returncode != 0, (
        f"expected a device trap, process passed: {combined[-1500:]!r}"
    )
    assert "UNEXPECTED-SURVIVAL" not in r.stdout, combined[-1500:]
    assert marker in combined, (
        f"trap marker {marker!r} missing -- setup failure faking a pass? "
        f"{combined[-1500:]!r}"
    )
    for bad in ("ImportError", "ModuleNotFoundError"):
        assert bad not in combined, combined[-1500:]


class _StubComm:
    def allgather(self, data):
        return [data]


def test_fc1_fused_layout_mismatch_raises():
    from flashinfer.fused_moe.sm90_push_a2a import _Sm90PushMoERunner, Sm90PushConfig

    class _FakePipe:
        config = Sm90PushConfig(fuse_fc1_epilogue=True)
        E, H = 2, 256
        rank = 0
        _comm = _StubComm()

    w13 = torch.zeros(2, 2 * 256, 256, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="w13_interleaved"):
        _Sm90PushMoERunner(_FakePipe(), w13, w13, w13, w13, w13_interleaved=False)

    class _FakePipe2:
        config = Sm90PushConfig(fuse_fc1_epilogue=False)
        E, H = 2, 256
        rank = 0
        _comm = _StubComm()

    with pytest.raises(RuntimeError, match="w13_interleaved"):
        _Sm90PushMoERunner(_FakePipe2(), w13, w13, w13, w13, w13_interleaved=True)


def test_fc1_fused_weight_bundle_tag():
    from flashinfer.fused_moe.sm90_push_a2a import (
        _Sm90PushMoERunner,
        Sm90PushConfig,
        make_sm90_push_weights,
    )

    w13 = torch.zeros(2, 2 * 256, 256, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 256, 256, dtype=torch.bfloat16)
    wt_plain = make_sm90_push_weights(w13, w2)
    wt_il = make_sm90_push_weights(w13, w2, interleave_gate_up=True)
    assert wt_plain.w13_interleaved is False
    assert wt_il.w13_interleaved is True

    class _FusedPipe:
        config = Sm90PushConfig(fuse_fc1_epilogue=True)
        E, H = 2, 256
        rank = 0
        _comm = _StubComm()

    class _PlainPipe:
        config = Sm90PushConfig(fuse_fc1_epilogue=False)
        E, H = 2, 256
        rank = 0
        _comm = _StubComm()

    # tag/config mismatch caught in both directions, no hand-copied bool
    with pytest.raises(RuntimeError, match="w13_interleaved"):
        _Sm90PushMoERunner(_FusedPipe(), wt_plain)
    with pytest.raises(RuntimeError, match="w13_interleaved"):
        _Sm90PushMoERunner(_PlainPipe(), wt_il)
    # bundle + contradicting explicit kwarg is an error, not a silent pick
    with pytest.raises(RuntimeError, match="contradicts"):
        _Sm90PushMoERunner(_PlainPipe(), wt_plain, w13_interleaved=True)
    # bundle + extra raw tensors is an error
    with pytest.raises(RuntimeError, match="not both"):
        _Sm90PushMoERunner(_PlainPipe(), wt_plain, w13_sf=w13)


def test_megamoe_constructor_weight_forms():
    from flashinfer.fused_moe.sm90_push_a2a import (
        _Sm90PushMoERunner,
        Sm90PushConfig,
        Sm90PushWeights,
        make_sm90_push_weights,
    )

    class _FusedPipe:
        config = Sm90PushConfig(fuse_fc1_epilogue=True)
        E, H = 2, 256
        rank = 0
        _comm = _StubComm()

    class _PlainPipe:
        config = Sm90PushConfig(fuse_fc1_epilogue=False)
        E, H = 2, 256
        rank = 0
        _comm = _StubComm()

    w13 = torch.zeros(2, 2 * 256, 256, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 256, 256, dtype=torch.bfloat16)
    bundle = make_sm90_push_weights(w13, w2)
    q13 = torch.zeros(2, 512, 256, dtype=torch.float8_e4m3fn)
    s13 = torch.zeros(2, 4, 2, dtype=torch.float32)
    q2 = torch.zeros(2, 256, 256, dtype=torch.float8_e4m3fn)
    s2 = torch.zeros(2, 2, 2, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="w13_interleaved"):
        _Sm90PushMoERunner(_FusedPipe(), w13_fp8=q13, w13_sf=s13, w2_fp8=q2, w2_sf=s2)
    # both entry styles at once / neither are errors (guarded phase: the
    # weight arguments are per-rank state, so form errors aggregate too)
    with pytest.raises(RuntimeError, match="not both"):
        _Sm90PushMoERunner(_PlainPipe(), bundle, w13_fp8=q13)
    with pytest.raises(RuntimeError, match="required"):
        _Sm90PushMoERunner(_PlainPipe())
    # wrong scale dtype (would be reinterpreted as float* downstream)
    with pytest.raises(RuntimeError, match="float32"):
        _Sm90PushMoERunner(_PlainPipe(), q13, s13.to(torch.float64), q2, s2)
    q13_bad = torch.zeros(2, 384, 256, dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="multiple of 256"):
        _Sm90PushMoERunner(_PlainPipe(), q13_bad, s13, q2, s2)
    # hand-built bundles validate in __post_init__ (no pipe: plain ValueError)
    with pytest.raises(ValueError, match="float32"):
        Sm90PushWeights(q13, s13.to(torch.float64), q2, s2)
    with pytest.raises(ValueError, match="multiple of 256"):
        Sm90PushWeights(q13_bad, s13, q2, s2)


@requires_sm90
def test_ep1_constructor_back_compat_bitwise_equal():
    from flashinfer.fused_moe.sm90_push_a2a import (
        _Sm90PushMoERunner,
        _Sm90PushPipe,
        make_sm90_push_weights,
        transform_weights_for_sm90_push,
    )

    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=311, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=312, device=dev)
    w13, w2 = _make_weights(E_TOTAL, seed=7, device=dev)

    pipe_a = _Sm90PushPipe(
        ep_size=1,
        rank=0,
        num_local_experts=E_TOTAL,
        hidden_size=H,
        top_k=TOPK,
        token_capacity=T_CAP,
        device_index=0,
    )
    _KEEP_ALIVE.append(pipe_a)
    q13, s13, q2, s2 = transform_weights_for_sm90_push(w13, w2)
    mm_legacy = _Sm90PushMoERunner(pipe_a, w13_fp8=q13, w13_sf=s13, w2_fp8=q2, w2_sf=s2)
    out_legacy = mm_legacy(x, ids, wts).clone()
    torch.cuda.synchronize()

    pipe_b = _Sm90PushPipe(
        ep_size=1,
        rank=0,
        num_local_experts=E_TOTAL,
        hidden_size=H,
        top_k=TOPK,
        token_capacity=T_CAP,
        device_index=0,
    )
    _KEEP_ALIVE.append(pipe_b)
    mm_bundle = _Sm90PushMoERunner(pipe_b, make_sm90_push_weights(w13, w2))
    mm_bundle.configure_workspace()
    out_bundle = mm_bundle(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_legacy, out_bundle), "legacy kwarg form != bundle form"


@requires_sm90
def test_ep1_fc1_fused_tiny_expected_m_swapab():
    token_capacity = 8
    T = 8
    dev = torch.device("cuda", 0)
    x = _make_x(T, seed=321, device=dev)
    ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=322, device=dev)
    _, mm_fused, _, fp8_w = _build(
        fuse_fc1_epilogue=True, token_capacity=token_capacity
    )
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build(token_capacity=token_capacity)
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), (
        "expected_m<32: fc1-fused != unfused (swapAB-tactic corner, bitwise)"
    )
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_fused, ref) > 0.997


@requires_sm90
def test_fc1_fused_ffi_rejects_bad_args():
    from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90

    dev = torch.device("cuda", 0)
    e_total, m, h, i = 2, 64, 256, 256
    runner = create_fp8_blockscale_gemm_runner_sm90()
    sz = runner.get_moe_workspace_size(
        m, max(2 * i, h), max(h, i), 1, e_total, True, True
    )
    workspace = torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8)
    runner.configure_workspace(workspace)
    p = (m + e_total * 31) // 32 * 32

    a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=dev)
    b = torch.zeros(e_total, 2 * i, h, dtype=torch.float8_e4m3fn, device=dev)
    sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=dev)
    sfb = torch.zeros(e_total, 2 * i // 128, h // 128, dtype=torch.float32, device=dev)
    d = torch.zeros(m, i, dtype=torch.uint8, device=dev)
    sfd = torch.zeros((i // 128) * p + 128, dtype=torch.float32, device=dev)
    offs = torch.zeros(e_total + 1, dtype=torch.int64, device=dev)

    def call(
        d_=None,
        sfd_=None,
        a_=None,
        b_=None,
        offs_=None,
        n=2 * i,
        k=h,
        sfa_=None,
        sfb_=None,
    ):
        runner.moe_gemm_fc1_fused(
            d if d_ is None else d_,
            sfd if sfd_ is None else sfd_,
            a if a_ is None else a_,
            b if b_ is None else b_,
            offs if offs_ is None else offs_,
            n,
            k,
            sfa if sfa_ is None else sfa_,
            sfb if sfb_ is None else sfb_,
            False,
        )

    call()  # the valid baseline must pass (all-empty groups)
    torch.cuda.synchronize()
    with pytest.raises(Exception, match="int64"):
        call(offs_=offs.to(torch.int32))
    with pytest.raises(Exception, match="contiguous"):  # CHECK_INPUT(offsets)
        call(offs_=torch.zeros(2 * (e_total + 1), dtype=torch.int64, device=dev)[::2])
    with pytest.raises(Exception, match="positive"):
        call(n=0)
    with pytest.raises(Exception, match="multiple of 256"):
        call(n=2 * i + 128)
    with pytest.raises(Exception, match="float32"):
        call(sfa_=sfa.to(torch.float64))
    with pytest.raises(Exception, match="scales_a"):
        call(sfa_=sfa[: (h // 128) * p // 2])
    with pytest.raises(Exception, match="sfa2"):
        call(sfd_=sfd[: (i // 128) * p // 2])
    with pytest.raises(Exception, match="declaration"):
        call(a_=a[: m // 2], d_=d)  # A shorter than the frozen TMA declaration
    with pytest.raises(Exception, match="rows"):
        call(d_=d[: m // 2])  # output rows below A's capacity


@requires_sm90
def test_moe_gemm_ffi_rejects_bad_args():
    from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90

    dev = torch.device("cuda", 0)
    e_total, m, h, i = 2, 64, 256, 256
    runner = create_fp8_blockscale_gemm_runner_sm90()
    sz = runner.get_moe_workspace_size(
        m, max(2 * i, h), max(h, i), 1, e_total, True, True
    )
    workspace = torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8)
    runner.configure_workspace(workspace)
    p = (m + e_total * 31) // 32 * 32

    a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=dev)
    b = torch.zeros(e_total, 2 * i, h, dtype=torch.float8_e4m3fn, device=dev)
    sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=dev)
    sfb = torch.zeros(e_total, 2 * i // 128, h // 128, dtype=torch.float32, device=dev)
    d = torch.zeros(m, 2 * i, dtype=torch.bfloat16, device=dev)
    offs = torch.zeros(e_total + 1, dtype=torch.int64, device=dev)

    def call(d_=None, a_=None, b_=None, offs_=None, n=2 * i, k=h, sfa_="d", sfb_="d"):
        runner.moe_gemm(
            d if d_ is None else d_,
            a if a_ is None else a_,
            b if b_ is None else b_,
            offs if offs_ is None else offs_,
            n,
            k,
            sfa if sfa_ == "d" else sfa_,
            sfb if sfb_ == "d" else sfb_,
            False,
        )

    call()  # the valid baseline must pass (all-empty groups)
    torch.cuda.synchronize()
    with pytest.raises(Exception, match="int64"):
        call(offs_=offs.to(torch.int32))
    with pytest.raises(Exception, match="contiguous"):  # CHECK_INPUT(offsets)
        call(offs_=torch.zeros(2 * (e_total + 1), dtype=torch.int64, device=dev)[::2])
    with pytest.raises(Exception, match="positive"):
        call(n=0)
    with pytest.raises(Exception, match="bfloat16"):
        call(d_=torch.zeros(m, 2 * i, dtype=torch.float32, device=dev))
    with pytest.raises(Exception, match="a is \\(M, K\\)"):
        call(a_=torch.zeros(m, h // 2, dtype=torch.float8_e4m3fn, device=dev))
    with pytest.raises(Exception, match="b must be"):
        call(b_=b[:, :i, :].contiguous())  # (G, I, K) != (G, N, K)
    with pytest.raises(Exception, match="float32"):
        call(sfa_=sfa.to(torch.float64))
    with pytest.raises(Exception, match="scales_a"):
        call(sfa_=sfa[: (h // 128) * p // 2])  # capacity below (K/128) * P
    with pytest.raises(Exception, match="requires scales_a"):
        call(sfa_=None)  # pre-quantized fp8 A without its grouped scales
    with pytest.raises(Exception, match="declaration"):
        call(a_=a[: m // 2])  # A shorter than the frozen TMA declaration
    with pytest.raises(Exception, match="rows"):
        call(d_=d[: m // 2])  # D rows below the frozen declaration
    with pytest.raises(Exception, match="runtime N exceeds"):
        call(
            d_=torch.zeros(m, 3 * i, dtype=torch.bfloat16, device=dev),
            b_=torch.zeros(e_total, 3 * i, h, dtype=torch.float8_e4m3fn, device=dev),
            n=3 * i,
        )
    with pytest.raises(Exception, match="runtime K exceeds"):
        call(
            a_=torch.zeros(m, h + 128, dtype=torch.float8_e4m3fn, device=dev),
            b_=torch.zeros(
                e_total,
                2 * i,
                h + 128,
                dtype=torch.float8_e4m3fn,
                device=dev,
            ),
            k=h + 128,
        )
    with pytest.raises(Exception, match="runtime group count"):
        call(
            b_=torch.zeros(
                e_total + 1, 2 * i, h, dtype=torch.float8_e4m3fn, device=dev
            ),
            offs_=torch.zeros(e_total + 2, dtype=torch.int64, device=dev),
        )


@requires_sm90
def test_moe_gemm_workspace_contract():
    from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90

    dev = torch.device("cuda", 0)
    runner = create_fp8_blockscale_gemm_runner_sm90()
    with pytest.raises(Exception, match="workspace-size query"):
        runner.configure_workspace(torch.empty(1, dtype=torch.uint8, device=dev))
    for args in (
        (0, 256, 256, 1, 2),
        (64, 0, 256, 1, 2),
        (64, 256, 0, 1, 2),
        (64, 256, 256, 0, 2),
        (64, 256, 256, 1, 0),
    ):
        with pytest.raises(Exception, match="must all be positive"):
            runner.get_moe_workspace_size(*args, False, False)

    size = int(runner.get_moe_workspace_size(64, 256, 256, 1, 2, False, False))
    assert size > 0
    with pytest.raises(Exception, match="uint8"):
        runner.configure_workspace(torch.empty(size, dtype=torch.float32, device=dev))
    with pytest.raises(Exception, match="must be a CUDA tensor"):
        runner.configure_workspace(torch.empty(size, dtype=torch.uint8))
    with pytest.raises(Exception, match="needs >="):
        runner.configure_workspace(torch.empty(size - 1, dtype=torch.uint8, device=dev))
    runner.configure_workspace(torch.empty(size, dtype=torch.uint8, device=dev))


@requires_sm90
@pytest.mark.parametrize("offset_values", ([0, 48, 16], [1, 32, 64]))
def test_moe_gemm_invalid_offsets_trap(offset_values):
    code = rf"""
import torch
from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90
dev = torch.device("cuda", 0)
e_total, m, h, i = 2, 64, 256, 256
runner = create_fp8_blockscale_gemm_runner_sm90()
sz = runner.get_moe_workspace_size(m, max(2 * i, h), max(h, i), 1, e_total, True, True)
runner.configure_workspace(torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8))
p = (m + e_total * 31) // 32 * 32
a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=dev)
b = torch.zeros(e_total, 2 * i, h, dtype=torch.float8_e4m3fn, device=dev)
sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=dev)
sfb = torch.zeros(e_total, 2 * i // 128, h // 128, dtype=torch.float32, device=dev)
d = torch.zeros(m, 2 * i, dtype=torch.bfloat16, device=dev)
offs = torch.tensor({list(offset_values)!r}, dtype=torch.int64, device=dev)
runner.moe_gemm(d, a, b, offs, 2 * i, h, sfa, sfb, False)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "moe_gemm: bad offsets")


@requires_sm90
@pytest.mark.parametrize("offset_values", ([0, 48, 16], [1, 32, 64]))
def test_fc1_fused_invalid_offsets_trap(offset_values):
    code = rf"""
import torch
from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90
dev = torch.device("cuda", 0)
e_total, m, h, i = 2, 64, 256, 256
runner = create_fp8_blockscale_gemm_runner_sm90()
sz = runner.get_moe_workspace_size(m, max(2 * i, h), max(h, i), 1, e_total, True, True)
runner.configure_workspace(torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8))
p = (m + e_total * 31) // 32 * 32
a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=dev)
b = torch.zeros(e_total, 2 * i, h, dtype=torch.float8_e4m3fn, device=dev)
sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=dev)
sfb = torch.zeros(e_total, 2 * i // 128, h // 128, dtype=torch.float32, device=dev)
d = torch.zeros(m, i, dtype=torch.uint8, device=dev)
sfd = torch.zeros((i // 128) * p + 128, dtype=torch.float32, device=dev)
offs = torch.tensor({list(offset_values)!r}, dtype=torch.int64, device=dev)
runner.moe_gemm_fc1_fused(d, sfd, a, b, offs, 2 * i, h, sfa, sfb, False)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "bad offsets")


@requires_sm90
def test_moe_gemm_offsets_exceed_a_capacity_traps():
    """BF16-A path: offsets within D rows but beyond A's rows must trap."""
    code = r"""
import torch
from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90
dev = torch.device("cuda", 0)
e_total, m, h, i = 2, 64, 256, 256
runner = create_fp8_blockscale_gemm_runner_sm90()
sz = runner.get_moe_workspace_size(m, max(2 * i, h), max(h, i), 1, e_total, False, False)
runner.configure_workspace(torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8))
a = torch.zeros(m // 2, h, dtype=torch.bfloat16, device=dev)  # HALF the rows
b = torch.zeros(e_total, 2 * i, h, dtype=torch.bfloat16, device=dev)
d = torch.zeros(m, 2 * i, dtype=torch.bfloat16, device=dev)
offs = torch.tensor([0, 32, 64], dtype=torch.int64, device=dev)  # 64 > a rows 32
runner.moe_gemm(d, a, b, offs, 2 * i, h, None, None, False)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "bad offsets")


@requires_sm90
def test_fc1_fused_offsets_exceed_capacity_traps():
    """Fused path: offsets total beyond the frozen A capacity must trap."""
    code = r"""
import torch
from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90
dev = torch.device("cuda", 0)
e_total, m, h, i = 2, 64, 256, 256
runner = create_fp8_blockscale_gemm_runner_sm90()
sz = runner.get_moe_workspace_size(m, max(2 * i, h), max(h, i), 1, e_total, True, True)
runner.configure_workspace(torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8))
p = (m + e_total * 31) // 32 * 32
a = torch.zeros(m, h, dtype=torch.float8_e4m3fn, device=dev)
b = torch.zeros(e_total, 2 * i, h, dtype=torch.float8_e4m3fn, device=dev)
sfa = torch.zeros((h // 128) * p + 128, dtype=torch.float32, device=dev)
sfb = torch.zeros(e_total, 2 * i // 128, h // 128, dtype=torch.float32, device=dev)
d = torch.zeros(2 * m, i, dtype=torch.uint8, device=dev)  # D roomy; A is the bound
sfd = torch.zeros((i // 128) * p + 128, dtype=torch.float32, device=dev)
offs = torch.tensor([0, m, 2 * m], dtype=torch.int64, device=dev)  # 128 > capacity 64
runner.moe_gemm_fc1_fused(d, sfd, a, b, offs, 2 * i, h, sfa, sfb, False)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "bad offsets")


@requires_sm90
def test_ep1_fc1_fused_a2_sfa2_bitwise_equal():
    from flashinfer.fused_moe import transform_weights_for_sm90_push
    from flashinfer.gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90
    from flashinfer.jit.fused_moe import gen_sm90_push_a2a_module

    dev = torch.device("cuda", 0)
    e_total = E_TOTAL
    group_ms = [37, 23, 41, 27]  # ragged groups; sums to 128 (tile-aligned)
    m_total = sum(group_ms)
    m_ws = m_total
    p_stride = (m_ws + e_total * 31) // 32 * 32

    offsets = torch.tensor(
        [0] + list(torch.tensor(group_ms).cumsum(0)), dtype=torch.int64, device=dev
    )
    pad_base = torch.tensor(
        [(int(offsets[e]) + e * 31) // 32 * 32 for e in range(e_total)],
        dtype=torch.int32,
        device=dev,
    )
    row_expert = torch.cat(
        [
            torch.full((m,), e, dtype=torch.int32, device=dev)
            for e, m in enumerate(group_ms)
        ]
    )
    m_dev = torch.tensor([m_total], dtype=torch.int32, device=dev)
    p_dev = torch.tensor([p_stride], dtype=torch.int32, device=dev)

    module = gen_sm90_push_a2a_module().build_and_load()
    g = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(m_total, H, generator=g).to(device=dev, dtype=torch.bfloat16)
    a1 = torch.zeros(m_total, H, dtype=torch.uint8, device=dev)
    sfa1 = torch.zeros((H // 128) * p_stride + 128, dtype=torch.float32, device=dev)
    module.sm90_quant_grouped(
        a1, sfa1, x, offsets, pad_base, m_dev, p_dev, row_expert, m_total
    )

    w13, w2 = _make_weights(e_total, seed=7, device=dev)
    w13_p, w13_sf_p, _, _ = transform_weights_for_sm90_push(w13, w2)
    w13_i, w13_sf_i, _, _ = transform_weights_for_sm90_push(
        w13, w2, interleave_gate_up=True
    )

    runner = create_fp8_blockscale_gemm_runner_sm90()
    sz = runner.get_moe_workspace_size(
        m_ws, max(2 * I, H), max(H, I), 1, e_total, True, True
    )
    workspace = torch.empty(max(int(sz), 1), device=dev, dtype=torch.uint8)
    runner.configure_workspace(workspace)

    def _poison(t):
        """0x7F sentinel in every byte: never a legal kernel output on this data."""
        t.view(torch.uint8).fill_(0x7F)
        return t

    # reference: unfused FC1 -> FA kernel (and the split pair as 2nd anchor)
    h = torch.zeros(m_total, 2 * I, dtype=torch.bfloat16, device=dev)
    runner.moe_gemm(
        h,
        a1.view(torch.float8_e4m3fn),
        w13_p,
        offsets,
        2 * I,
        H,
        sfa1,
        w13_sf_p,
        False,
    )
    a2_fa = _poison(torch.empty(m_total, I, dtype=torch.uint8, device=dev))
    sfa2_fa = _poison(
        torch.empty((I // 128) * p_stride + 128, dtype=torch.float32, device=dev)
    )
    module.sm90_silu_mul_quant_grouped(
        a2_fa, sfa2_fa, h, offsets, pad_base, m_dev, p_dev, row_expert, m_total
    )
    g_buf = torch.zeros(m_total, I, dtype=torch.bfloat16, device=dev)
    module.sm90_silu_mul_gated(g_buf, h, m_dev, m_total)
    a2_split = _poison(torch.empty_like(a2_fa))
    sfa2_split = _poison(torch.empty_like(sfa2_fa))
    module.sm90_quant_grouped(
        a2_split,
        sfa2_split,
        g_buf,
        offsets,
        pad_base,
        m_dev,
        p_dev,
        row_expert,
        m_total,
    )

    # fused: one call produces both tensors
    a2_fused = _poison(torch.empty_like(a2_fa))
    sfa2_fused = _poison(torch.empty_like(sfa2_fa))
    runner.moe_gemm_fc1_fused(
        a2_fused,
        sfa2_fused,
        a1.view(torch.float8_e4m3fn),
        w13_i,
        offsets,
        2 * I,
        H,
        sfa1,
        w13_sf_i,
        False,
    )
    torch.cuda.synchronize()

    assert torch.equal(a2_fa, a2_split), "FA != split pair on a2 (test harness bug?)"
    assert torch.equal(sfa2_fa, sfa2_split), "FA != split pair on sfa2"
    assert torch.equal(a2_fused, a2_fa), "fc1-fused a2 bytes != FA a2 (bitwise)"
    assert torch.equal(sfa2_fused.view(torch.int32), sfa2_fa.view(torch.int32)), (
        "fc1-fused sfa2 != FA sfa2 (bitwise)"
    )
    live_cols = torch.zeros(p_stride, dtype=torch.bool, device=dev)
    for e, m in enumerate(group_ms):
        pb = int(pad_base[e].item())
        live_cols[pb : pb + m] = True
    for name, sf in (("fa", sfa2_fa), ("split", sfa2_split), ("fused", sfa2_fused)):
        grid = sf[: (I // 128) * p_stride].view(I // 128, p_stride)
        pad_bytes = grid[:, ~live_cols].contiguous().view(torch.uint8)
        assert bool((pad_bytes == 0x7F).all()), (
            f"sfa2[{name}] wrote inter-group padding columns"
        )
        tail_bytes = sf[(I // 128) * p_stride :].contiguous().view(torch.uint8)
        assert bool((tail_bytes == 0x7F).all()), f"sfa2[{name}] wrote the +128 tail"


@requires_sm90
def test_ep1_fc1_fused_bitwise_equal():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=41, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=42, device=dev)
    _, mm_fused, _, _ = _build(fuse_fc1_epilogue=True)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_fa, _, _ = _build(fuse_act=True)
    mm_fa.configure_workspace()
    out_fa = mm_fa(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_split, _, _ = _build(fuse_act=False)
    mm_split.configure_workspace()
    out_split = mm_split(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_fa), "fc1-fused != FA path (bitwise)"
    assert torch.equal(out_fused, out_split), "fc1-fused != split pair (bitwise)"


@requires_sm90
@pytest.mark.parametrize("T", [1, 37, T_CAP // 3])
def test_ep1_fc1_fused_m_less_than_cap(T):
    dev = torch.device("cuda", 0)
    x = _make_x(T, seed=51 + T, device=dev)
    ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=52 + T, device=dev)
    _, mm_fused, _, fp8_w = _build(fuse_fc1_epilogue=True)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build()
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), f"T={T}: fc1-fused != unfused (bitwise)"
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_fused, ref) > 0.997


@requires_sm90
def test_ep1_fc1_fused_small_m_block64():
    token_capacity = 32
    dev = torch.device("cuda", 0)
    x = _make_x(token_capacity, seed=57, device=dev)
    ids, wts = _make_routing(token_capacity, E_TOTAL, TOPK, seed=58, device=dev)
    _, mm_fused, _, fp8_w = _build(
        fuse_fc1_epilogue=True, token_capacity=token_capacity
    )
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build(token_capacity=token_capacity)
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), "block_m=64: fc1-fused != unfused"
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_fused, ref) > 0.997


@requires_sm90
def test_ep1_fc1_fused_hot_routing():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=61, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=62, device=dev, mode="hot")
    _, mm_fused, _, _ = _build(fuse_fc1_epilogue=True)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build()
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), "hot: fc1-fused != unfused (bitwise)"


@requires_sm90
def test_ep1_fc1_fused_masked_routes():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=63, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=64, device=dev)
    ids = ids.clone()
    ids[::3, 0] = -1
    _, mm_fused, _, _ = _build(fuse_fc1_epilogue=True)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build()
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), "masked: fc1-fused != unfused (bitwise)"


@requires_sm90
def test_ep1_fc1_fused_empty_tokens():
    pipe, mm_fused, _, _ = _build(fuse_fc1_epilogue=True)
    dev = pipe.device
    x0 = torch.empty(0, H, dtype=torch.bfloat16, device=dev)
    ids0 = torch.empty(0, TOPK, dtype=torch.int32, device=dev)
    wts0 = torch.empty(0, TOPK, dtype=torch.float32, device=dev)
    out0 = mm_fused(x0, ids0, wts0)
    torch.cuda.synchronize()
    assert out0.shape == (0, H)

    x = _make_x(T_CAP, seed=71, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=72, device=dev)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build()
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), (
        "round after a T=0 fused round != unfused (bitwise)"
    )


@requires_sm90
@pytest.mark.parametrize("top_k", [6, 8])
def test_ep1_fc1_fused_topk(top_k):
    e_total = 8
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=81 + top_k, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, top_k, seed=82 + top_k, device=dev)
    _, mm_fused, _, fp8_w = _build(fuse_fc1_epilogue=True, e_total=e_total, top_k=top_k)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build(e_total=e_total, top_k=top_k)
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), (
        f"top_k={top_k}: fc1-fused != unfused (bitwise)"
    )
    ref = _dequant_reference(x, ids, wts, fp8_w, e_total)
    assert _cos(out_fused, ref) > 0.997


@requires_sm90
def test_ep1_fc1_fused_large_shape():
    h, i = 1536, 2048
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=91, device=dev, h=h)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=92, device=dev)
    _, mm_fused, _, fp8_w = _build(fuse_fc1_epilogue=True, h=h, i=i)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build(h=h, i=i)
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), "large shape: fc1-fused != unfused"
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_fused, ref) > 0.997


@requires_sm90
@pytest.mark.skipif(
    not os.environ.get("SM90_PUSH_HEAVY"),
    reason="DSV3-shape gate is heavy; set SM90_PUSH_HEAVY=1 to run",
)
def test_ep1_fc1_fused_dsv3_shape():
    h, i = 7168, 2048
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=93, device=dev, h=h)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=94, device=dev)
    _, mm_fused, _, _ = _build(fuse_fc1_epilogue=True, h=h, i=i)
    out_fused = mm_fused(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_ref, _, _ = _build(h=h, i=i)
    mm_ref.configure_workspace()
    out_ref = mm_ref(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_fused, out_ref), "DSV3 shape: fc1-fused != unfused"


@requires_sm90
def test_ep1_fc1_fused_graph_replay():
    _, mm_fused, _, _ = _build(fuse_fc1_epilogue=True)
    dev = torch.device("cuda", 0)

    xs = [_make_x(T_CAP, seed=95 + i_, device=dev) for i_ in range(2)]
    routing = [
        _make_routing(T_CAP, E_TOTAL, TOPK, seed=97 + i_, device=dev) for i_ in range(2)
    ]
    # fused eager references
    eager = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        eager.append(mm_fused(x, ids, wts).clone())
        torch.cuda.synchronize()
    # unfused eager anchors
    _, mm_ref, _, _ = _build()
    mm_ref.configure_workspace()
    anchors = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        anchors.append(mm_ref(x, ids, wts).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(xs[0])
    static_ids = torch.empty_like(routing[0][0])
    static_w = torch.empty_like(routing[0][1])
    static_x.copy_(xs[0])
    static_ids.copy_(routing[0][0])
    static_w.copy_(routing[0][1])

    for _ in range(2):  # warmup on a side stream, as the capture will run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mm_fused(static_x, static_ids, static_w)
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = mm_fused(static_x, static_ids, static_w)

    outs = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_w.copy_(wts)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(static_out.clone())

    assert not torch.equal(outs[0], outs[1]), "replay ignored the input change"
    for i_ in range(2):
        assert torch.equal(outs[i_], eager[i_]), (
            f"fused graph replay {i_} != fused eager (bitwise)"
        )
        assert torch.equal(outs[i_], anchors[i_]), (
            f"fused graph replay {i_} != unfused eager (bitwise)"
        )


@requires_sm90
@pytest.mark.parametrize(
    "dedup,grouped",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_ep1_fusion_bitwise_equal_across_dedup_grouped_states(dedup, grouped):
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=241, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=242, device=dev)
    _, mm_off, _, fp8_w = _build(dedup=dedup, grouped_combine=grouped)
    out_off = mm_off(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_on, _, _ = _build(
        dedup=dedup, grouped_combine=grouped, fuse_fc1_epilogue=True
    )
    mm_on.configure_workspace()
    out_on = mm_on(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_on, out_off), (
        f"dedup={dedup} grouped={grouped}: fusion ON != OFF (bitwise) -- the "
        "FC1 fusion is no longer transparent at this flag combination"
    )
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_on, ref) > 0.997


@requires_sm90
@pytest.mark.parametrize(
    "mode,top_k",
    [
        ("random", 2),
        ("hot", 2),
        ("random", 6),
        ("hot", 6),
        ("random", 8),
        ("hot", 8),
    ],
)
def test_ep1_grouped_combine_accuracy(mode, top_k):
    # routing needs top_k <= num experts; 8 keeps route variety at K=6
    e_total = E_TOTAL if top_k <= E_TOTAL else 8
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=41, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, top_k, seed=42, device=dev, mode=mode)
    _, mm_r, _, fp8_w = _build(e_total=e_total, top_k=top_k)
    out_r = mm_r(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_g, _, _ = _build(e_total=e_total, top_k=top_k, grouped_combine=True)
    mm_g.configure_workspace()
    out_g = mm_g(x, ids, wts).clone()
    torch.cuda.synchronize()
    ref = _dequant_reference(x, ids, wts, fp8_w, e_total)
    err_r, err_g = _err(out_r, ref), _err(out_g, ref)
    cos_g = _cos(out_g, ref)
    assert cos_g > 0.997, f"{mode}/K{top_k}: grouped cos {cos_g:.5f}"
    assert err_g <= err_r * 1.05 + 1e-7, (
        f"{mode}/K{top_k}: grouped err {err_g:.5f} > 1.05 * per-route err {err_r:.5f}"
    )


@requires_sm90
def test_ep1_grouped_scale_dilution_extreme_weights():
    dev = torch.device("cuda", 0)
    T = T_CAP
    g = torch.Generator(device="cpu").manual_seed(97)
    base = torch.arange(T, dtype=torch.int32) % E_TOTAL
    ids = torch.stack([base, (base + 1) % E_TOTAL], dim=1).to(dev)
    w_big = 0.9 + 0.2 * torch.rand(T, 1, generator=g)
    w_small = 5e-4 + 1.5e-3 * torch.rand(T, 1, generator=g)
    wts = torch.cat([w_big, w_small], dim=1).to(device=dev, dtype=torch.float32)
    x = _make_x(T, seed=98, device=dev)

    _, mm_r, _, fp8_w = _build()
    out_r = mm_r(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_g, _, _ = _build(grouped_combine=True)
    mm_g.configure_workspace()
    out_g = mm_g(x, ids, wts).clone()
    torch.cuda.synchronize()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    err_r, err_g = _err(out_r, ref), _err(out_g, ref)
    cos_g = _cos(out_g, ref)
    assert cos_g > 0.997, f"dilution: grouped cos {cos_g:.5f}"
    assert err_g <= err_r * 1.05 + 1e-7, (
        f"scale dilution: grouped err {err_g:.5f} > 1.05 * per-route "
        f"{err_r:.5f} (see docstring for the observation recipe)"
    )


@requires_sm90
def test_ep1_grouped_masked_routes():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=43, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=44, device=dev)
    ids = ids.clone()
    ids[::3, 0] = -1
    _, mm_r, _, fp8_w = _build()
    out_r = mm_r(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_g, _, _ = _build(grouped_combine=True)
    mm_g.configure_workspace()
    out_g = mm_g(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.isfinite(out_g).all()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_g, ref) > 0.997
    assert _err(out_g, ref) <= _err(out_r, ref) * 1.05 + 1e-7


@requires_sm90
@pytest.mark.parametrize("T", [1, 37])
def test_ep1_grouped_m_less_than_cap(T):
    pipe, mm, _, fp8_w = _build(grouped_combine=True)
    dev = pipe.device
    x = _make_x(T, seed=45 + T, device=dev)
    ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=46 + T, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert out.shape == (T, H)
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997


@requires_sm90
def test_ep1_grouped_empty_then_recover():
    pipe, mm, _, fp8_w = _build(grouped_combine=True)
    dev = pipe.device
    x0 = torch.empty(0, H, dtype=torch.bfloat16, device=dev)
    ids0 = torch.empty(0, TOPK, dtype=torch.int32, device=dev)
    wts0 = torch.empty(0, TOPK, dtype=torch.float32, device=dev)
    out0 = mm(x0, ids0, wts0)
    torch.cuda.synchronize()
    assert out0.shape == (0, H)
    x = _make_x(T_CAP, seed=47, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=48, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997, "grouped round after a T=0 round is broken"


@requires_sm90
def test_ep1_grouped_graph_replay():
    pipe, mm, _, _ = _build(grouped_combine=True)
    dev = pipe.device

    xs = [_make_x(T_CAP, seed=51 + i, device=dev) for i in range(2)]
    routing = [
        _make_routing(T_CAP, E_TOTAL, TOPK, seed=61 + i, device=dev) for i in range(2)
    ]
    eager = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        eager.append(mm(x, ids, wts).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(xs[0])
    static_ids = torch.empty_like(routing[0][0])
    static_w = torch.empty_like(routing[0][1])
    static_x.copy_(xs[0])
    static_ids.copy_(routing[0][0])
    static_w.copy_(routing[0][1])

    for _ in range(2):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mm(static_x, static_ids, static_w)
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = mm(static_x, static_ids, static_w)

    outs = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_w.copy_(wts)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(static_out.clone())

    assert not torch.equal(outs[0], outs[1]), "replay ignored the input change"
    for i in range(2):
        assert torch.equal(outs[i], eager[i]), (
            f"grouped graph replay {i} != eager (bitwise)"
        )


@requires_sm90
@pytest.mark.parametrize("T", [1, 37, T_CAP // 3])
def test_ep1_m_less_than_cap(T):
    pipe, mm, _, fp8_w = _build()
    dev = pipe.device
    x = _make_x(T, seed=9 + T, device=dev)
    ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=10 + T, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert out.shape == (T, H)
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997


@requires_sm90
def test_ep1_empty_tokens():
    pipe, mm, _, fp8_w = _build()
    dev = pipe.device
    x0 = torch.empty(0, H, dtype=torch.bfloat16, device=dev)
    ids0 = torch.empty(0, TOPK, dtype=torch.int32, device=dev)
    wts0 = torch.empty(0, TOPK, dtype=torch.float32, device=dev)
    out0 = mm(x0, ids0, wts0)
    torch.cuda.synchronize()
    assert out0.shape == (0, H)

    x = _make_x(T_CAP, seed=11, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=12, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997, "round after a T=0 round is broken"


@requires_sm90
def test_ep1_masked_expert_ids():
    pipe, mm, _, fp8_w = _build()
    dev = pipe.device
    x = _make_x(T_CAP, seed=13, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=14, device=dev)
    ids = ids.clone()
    ids[::3, 0] = -1  # mask every third token's first route
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)  # -1 never matches
    assert _cos(out, ref) > 0.997


@requires_sm90
def test_ep1_hot_routing():
    pipe, mm, _, fp8_w = _build()
    dev = pipe.device
    x = _make_x(T_CAP, seed=15, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=16, device=dev, mode="hot")
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997


@requires_sm90
def test_ep1_repeated_rounds():
    pipe, mm, _, fp8_w = _build()
    dev = pipe.device
    for r in range(5):
        T = [T_CAP, 5, T_CAP // 2, T_CAP, 17][r]
        x = _make_x(T, seed=100 + r, device=dev)
        ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=200 + r, device=dev)
        out = mm(x, ids, wts).clone()
        torch.cuda.synchronize()
        ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
        assert _cos(out, ref) > 0.997, f"round {r} diverged"


@requires_sm90
def test_ep1_soak():
    rounds = int(os.environ.get("SM90_PUSH_SOAK_ROUNDS", "60"))
    assert rounds >= 1, "SM90_PUSH_SOAK_ROUNDS must be >= 1 (0 passes vacuously)"
    pipe, mm, _, fp8_w = _build()
    dev = pipe.device
    t_choices = [0, 1, 7, T_CAP // 2, T_CAP]
    pending = []  # every round's stream-ordered output snapshot until the drain
    for r in range(rounds):
        T = t_choices[(r * 7 + 3) % len(t_choices)]
        mode = "hot" if r % 3 == 2 else "random"
        x = _make_x(T, seed=1000 + r, device=dev)
        ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=2000 + r, device=dev, mode=mode)
        out = mm(x, ids, wts)
        pending.append((r, mode, x, ids, wts, out.clone()))
        if r % 10 == 9 or r == rounds - 1:
            torch.cuda.synchronize()  # drain the queue, then check every round
            for rr, mmode, xx, iids, wwts, oo in pending:
                if xx.shape[0] == 0:
                    continue
                assert torch.isfinite(oo).all(), f"soak round {rr}: NaN/Inf"
                ref = _dequant_reference(xx, iids, wwts, fp8_w, E_TOTAL)
                cos = _cos(oo, ref)
                assert cos > 0.997, (
                    f"soak round {rr} ({mmode}, T={xx.shape[0]}): cos {cos:.5f}"
                )
            pending.clear()


@requires_sm90
def test_ep1_graph_replay():
    pipe, mm, _, _ = _build()
    dev = pipe.device

    xs = [_make_x(T_CAP, seed=21 + i, device=dev) for i in range(2)]
    routing = [
        _make_routing(T_CAP, E_TOTAL, TOPK, seed=31 + i, device=dev) for i in range(2)
    ]
    # eager references first (each is one protocol round)
    eager = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        eager.append(mm(x, ids, wts).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(xs[0])
    static_ids = torch.empty_like(routing[0][0])
    static_w = torch.empty_like(routing[0][1])
    static_x.copy_(xs[0])
    static_ids.copy_(routing[0][0])
    static_w.copy_(routing[0][1])

    for _ in range(2):  # warmup on a side stream, as the capture will run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mm(static_x, static_ids, static_w)
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = mm(static_x, static_ids, static_w)

    outs = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_w.copy_(wts)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(static_out.clone())

    assert not torch.equal(outs[0], outs[1]), "replay ignored the input change"
    for i in range(2):
        assert torch.equal(outs[i], eager[i]), f"graph replay {i} != eager (bitwise)"


@requires_sm90
def test_ep1_pool_overflow_traps():
    code = r"""
import torch
from flashinfer.fused_moe.sm90_push_a2a import (
    _Sm90PushMoERunner, _Sm90PushPipe, Sm90PushConfig, transform_weights_for_sm90_push,
)
H, I, E, K, T = 512, 768, 4, 2, 64
cfg = Sm90PushConfig(capacity_factor=0.25)
pipe = _Sm90PushPipe(ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
                token_capacity=T, device_index=0, config=cfg)
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * I ** -0.5
mm = _Sm90PushMoERunner(pipe, *transform_weights_for_sm90_push(w13, w2))
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
w = torch.rand(T, K, dtype=torch.float32, device="cuda")
out = mm(x, ids, w)   # T*K routes > 0.25*T*K pool rows -> device trap
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "sm90_push: pool overflow")


@requires_sm90
def test_ep1_out_of_range_expert_id_traps():
    code = r"""
import torch
from flashinfer.fused_moe.sm90_push_a2a import (
    _Sm90PushMoERunner, _Sm90PushPipe, transform_weights_for_sm90_push,
)
H, I, E, K, T = 512, 768, 4, 2, 64
pipe = _Sm90PushPipe(ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
                token_capacity=T, device_index=0)
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * I ** -0.5
mm = _Sm90PushMoERunner(pipe, *transform_weights_for_sm90_push(w13, w2))
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
ids[3, 1] = E + 7  # out of range
w = torch.rand(T, K, dtype=torch.float32, device="cuda")
out = mm(x, ids, w)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "sm90_push: invalid expert id")


@requires_sm90
@pytest.mark.parametrize("payload", ["fp8", "bf16"])
def test_ep1_dedup_bitwise_equal(payload):
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=41, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=42, device=dev)
    ids = ids.clone()
    ids[::3, 0] = -1  # masked first route: the carrier shifts to route 1
    _, mm_a, _, _ = _build(payload_dtype=payload, combine_dtype="bf16")
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_b, _, _ = _build(payload_dtype=payload, combine_dtype="bf16", dedup=True)
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b), f"dedup != per-route ({payload}, bitwise)"


@requires_sm90
def test_ep1_dedup_hot():
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=43, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=44, device=dev, mode="hot")
    _, mm_a, _, fp8_w = _build()
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_b, _, _ = _build(dedup=True)
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b), "dedup != per-route on hot (bitwise)"
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out_b, ref) > 0.997


@requires_sm90
@pytest.mark.parametrize("T", [1, 37])
def test_ep1_dedup_m_less_than_cap(T):
    pipe, mm, _, fp8_w = _build(dedup=True)
    dev = pipe.device
    x = _make_x(T, seed=45 + T, device=dev)
    ids, wts = _make_routing(T, E_TOTAL, TOPK, seed=46 + T, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert out.shape == (T, H)
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997


@requires_sm90
@pytest.mark.parametrize("top_k", [6, 8])
def test_ep1_dedup_bitwise_equal_topk(top_k):
    e_total = 8
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=51, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, top_k, seed=52, device=dev)
    _, mm_a, _, _ = _build(e_total=e_total, top_k=top_k, combine_dtype="bf16")
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_b, _, _ = _build(
        e_total=e_total, top_k=top_k, combine_dtype="bf16", dedup=True
    )
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b), f"dedup != per-route (top_k={top_k}, bitwise)"


@requires_sm90
def test_ep1_dedup_same_rank_multi_expert_bitwise_equal():
    dev = torch.device("cuda", 0)
    T = T_CAP
    x = _make_x(T, seed=53, device=dev)
    # token i -> experts {i, i+1, ...} mod E: K distinct experts, all rank 0
    base = torch.arange(T, dtype=torch.int64).unsqueeze(1)
    ids = ((base + torch.arange(TOPK).unsqueeze(0)) % E_TOTAL).to(torch.int32)
    ids = ids.to(dev)
    g = torch.Generator(device="cpu").manual_seed(54)
    wts = torch.rand(T, TOPK, generator=g) + 0.1
    wts = (wts / wts.sum(dim=1, keepdim=True)).to(device=dev, dtype=torch.float32)
    _, mm_a, _, _ = _build(combine_dtype="bf16")
    out_a = mm_a(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_b, _, _ = _build(combine_dtype="bf16", dedup=True)
    mm_b.configure_workspace()
    out_b = mm_b(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b), "same-rank multi-expert sharing (bitwise)"


@requires_sm90
def test_ep1_dedup_graph_replay():
    pipe, mm, _, _ = _build(dedup=True)
    dev = pipe.device

    xs = [_make_x(T_CAP, seed=61 + i, device=dev) for i in range(2)]
    routing = [
        _make_routing(T_CAP, E_TOTAL, TOPK, seed=71 + i, device=dev) for i in range(2)
    ]
    eager = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        eager.append(mm(x, ids, wts).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(xs[0])
    static_ids = torch.empty_like(routing[0][0])
    static_w = torch.empty_like(routing[0][1])
    static_x.copy_(xs[0])
    static_ids.copy_(routing[0][0])
    static_w.copy_(routing[0][1])

    for _ in range(2):  # warmup on a side stream, as the capture will run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mm(static_x, static_ids, static_w)
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = mm(static_x, static_ids, static_w)

    outs = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_w.copy_(wts)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(static_out.clone())

    assert not torch.equal(outs[0], outs[1]), "replay ignored the input change"
    for i in range(2):
        assert torch.equal(outs[i], eager[i]), f"dedup graph replay {i} != eager"


@requires_sm90
def test_ep1_dedup_payload_pool_overflow_traps():
    code = r"""
import torch
from flashinfer.fused_moe.sm90_push_a2a import (
    _Sm90PushMoERunner, _Sm90PushPipe, Sm90PushConfig, transform_weights_for_sm90_push,
)
H, I, E, K = 512, 768, 4, 4
cfg = Sm90PushConfig(capacity_factor=0.25, dedup_dispatch=True)
pipe = _Sm90PushPipe(ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
                token_capacity=64, device_index=0, config=cfg)
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * I ** -0.5
mm = _Sm90PushMoERunner(pipe, *transform_weights_for_sm90_push(w13, w2))
T = 32
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.full((T, K), -1, dtype=torch.int32, device="cuda")
ids[:, 0] = torch.arange(T, dtype=torch.int32, device="cuda") % E
w = torch.rand(T, K, dtype=torch.float32, device="cuda")
out = mm(x, ids, w)   # 32 distinct (token, dst) > 16 payload rows -> trap
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "sm90_push: dedup pool overflow")


@requires_sm90
def test_ep1_dedup_meta_pool_overflow_traps():
    code = r"""
import torch
from flashinfer.fused_moe.sm90_push_a2a import (
    _Sm90PushMoERunner, _Sm90PushPipe, Sm90PushConfig, transform_weights_for_sm90_push,
)
H, I, E, K = 512, 768, 4, 4
cfg = Sm90PushConfig(capacity_factor=0.25, dedup_dispatch=True)
pipe = _Sm90PushPipe(ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
                token_capacity=64, device_index=0, config=cfg)
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * I ** -0.5
mm = _Sm90PushMoERunner(pipe, *transform_weights_for_sm90_push(w13, w2))
T = 32
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.zeros(T, K, dtype=torch.int32, device="cuda")  # expert 0, K times
w = torch.rand(T, K, dtype=torch.float32, device="cuda")
out = mm(x, ids, w)   # meta 32*K=128 > 64 records (and payload 32 > 16) -> trap
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    _assert_device_trap(r, "sm90_push: dedup pool overflow")


@requires_sm90
def test_ep1_dedup_empty_tokens():
    pipe, mm, _, fp8_w = _build(dedup=True)
    dev = pipe.device
    x0 = torch.empty(0, H, dtype=torch.bfloat16, device=dev)
    ids0 = torch.empty(0, TOPK, dtype=torch.int32, device=dev)
    wts0 = torch.empty(0, TOPK, dtype=torch.float32, device=dev)
    out0 = mm(x0, ids0, wts0)
    torch.cuda.synchronize()
    assert out0.shape == (0, H)

    x = _make_x(T_CAP, seed=47, device=dev)
    ids, wts = _make_routing(T_CAP, E_TOTAL, TOPK, seed=48, device=dev)
    out = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    ref = _dequant_reference(x, ids, wts, fp8_w, E_TOTAL)
    assert _cos(out, ref) > 0.997, "dedup round after a T=0 round is broken"


@requires_sm90
@pytest.mark.parametrize(
    "mode,top_k",
    [
        ("random", 2),
        ("hot", 2),
        ("random", 6),
        ("hot", 6),
        ("random", 8),
        ("hot", 8),
    ],
)
def test_ep1_dedup_grouped_accuracy(mode, top_k):
    e_total = E_TOTAL if top_k <= E_TOTAL else 8
    dev = torch.device("cuda", 0)
    x = _make_x(T_CAP, seed=141, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, top_k, seed=142, device=dev, mode=mode)
    _, mm_r, _, fp8_w = _build(e_total=e_total, top_k=top_k)  # per-route error bar
    out_r = mm_r(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_g, _, _ = _build(e_total=e_total, top_k=top_k, grouped_combine=True)
    mm_g.configure_workspace()
    out_g = mm_g(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_dg, _, _ = _build(
        e_total=e_total, top_k=top_k, dedup=True, grouped_combine=True
    )
    mm_dg.configure_workspace()
    out_dg = mm_dg(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_dg, out_g), (
        f"{mode}/K{top_k}: dedup+grouped != grouped-only (bitwise) -- dedup "
        "is no longer transparent to the combine stage"
    )
    ref = _dequant_reference(x, ids, wts, fp8_w, e_total)
    err_r, err_dg = _err(out_r, ref), _err(out_dg, ref)
    cos_dg = _cos(out_dg, ref)
    assert cos_dg > 0.997, f"{mode}/K{top_k}: dedup+grouped cos {cos_dg:.5f}"
    assert err_dg <= err_r * 1.05 + 1e-7, (
        f"{mode}/K{top_k}: dedup+grouped err {err_dg:.5f} > "
        f"1.05 * per-route {err_r:.5f}"
    )


def _dist_setup():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")  # handle transfer + barriers only
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    from flashinfer.comm.mnnvl import TorchDistBackend

    return rank, world, TorchDistBackend()


def _dist_forward_and_check(
    payload,
    combine,
    t_of_rank,
    seed=50,
    skew_rank=None,
    mode="random",
    t_of_rank_next=None,
    dedup=False,
    grouped_combine=False,
    fuse_fc1_epilogue=False,
):
    """One collective round, oracle-checked per rank; optionally a second
    round on the same pipe (recovery gate after edge-case rounds)."""
    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world  # keep E/rank constant as world grows
    pipe, mm, _, _ = _build(
        payload_dtype=payload,
        combine_dtype=combine,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
        dedup=dedup,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    dev = pipe.device
    # full weight set for the oracle (identical on every rank by seed)
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13, w2 = _make_weights(e_total, seed=7, device=dev)
    fp8_full = transform_weights_for_sm90_push(w13, w2)

    if skew_rank is not None and rank == skew_rank:
        torch.cuda._sleep(int(2e8))  # ~100ms of launch skew before round 0

    rounds = [t_of_rank] if t_of_rank_next is None else [t_of_rank, t_of_rank_next]
    for ridx, t_fn in enumerate(rounds):
        T = t_fn(rank)
        x = _make_x(T, seed=seed + 1000 * ridx + rank, device=dev)
        ids, wts = _make_routing(
            T,
            e_total,
            TOPK,
            seed=seed + 100 + 1000 * ridx + rank,
            device=dev,
            mode=mode,
            rank=rank,
            e_local=E_TOTAL,
        )
        out = mm(x, ids, wts).clone()
        torch.cuda.synchronize()

        assert torch.isfinite(out).all(), f"rank {rank} round {ridx}: NaN/Inf"
        if T > 0:
            ref = _dequant_reference(x, ids, wts, fp8_full, e_total)
            cos = _cos(out, ref)
            assert cos > 0.997, f"rank {rank} round {ridx}: cos {cos:.5f}"
    import torch.distributed as dist

    dist.barrier()


@requires_dist
def test_dist_full_pipeline_even():
    _dist_forward_and_check("fp8", "fp8", lambda r: T_CAP)


@requires_dist
def test_dist_full_pipeline_uneven_tokens():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: 0 if r == 1 else max(T_CAP - 13 * r, 1),
        seed=60,
        t_of_rank_next=lambda r: T_CAP,
    )


@requires_dist
def test_dist_fp8_payload_and_combine():
    _dist_forward_and_check("fp8", "fp8", lambda r: T_CAP, seed=70)
    _dist_forward_and_check("bf16", "bf16", lambda r: T_CAP, seed=70)


@requires_dist
def test_dist_all_remote():
    _dist_forward_and_check("fp8", "fp8", lambda r: T_CAP, seed=75, mode="all_remote")


@requires_dist
def test_dist_fc1_fused():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: 0 if r == 1 else T_CAP,
        seed=85,
        t_of_rank_next=lambda r: T_CAP,
        fuse_fc1_epilogue=True,
    )

    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world
    pipe_f, mm_f, _, _ = _build(
        fuse_fc1_epilogue=True,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    dev = pipe_f.device
    x = _make_x(T_CAP, seed=181 + rank, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, TOPK, seed=182 + rank, device=dev)
    out_f = mm_f(x, ids, wts).clone()
    torch.cuda.synchronize()
    _, mm_u, _, _ = _build(
        device_index=rank, ep=world, rank=rank, comm=comm, e_total=e_total
    )
    out_u = mm_u(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_f, out_u), (
        f"rank {rank}: dist fc1-fused != unfused (bitwise)"
    )
    import torch.distributed as dist

    dist.barrier()


@requires_dist
def test_dist_dedup():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: T_CAP,
        seed=85,
        dedup=True,
        t_of_rank_next=lambda r: 0 if r == 0 else T_CAP - 7 * r,
    )


@requires_dist
def test_dist_dedup_all_remote():
    _dist_forward_and_check(
        "fp8", "fp8", lambda r: T_CAP, seed=95, mode="all_remote", dedup=True
    )


@requires_dist
def test_dist_grouped_combine_accuracy():
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world
    # Both pipes are collectives: construct in the same order on every rank.
    _, mm_r, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    _, mm_g, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        grouped_combine=True,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    mm_g.configure_workspace()
    dev = torch.device("cuda", rank)
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13, w2 = _make_weights(e_total, seed=7, device=dev)
    fp8_full = transform_weights_for_sm90_push(w13, w2)

    cases = [
        ("random", T_CAP),
        ("hot", T_CAP),
        ("all_remote", T_CAP),
        ("random", 0 if rank == 1 else max(T_CAP // 2, 1)),  # uneven + empty
    ]
    for ridx, (mode, T) in enumerate(cases):
        x = _make_x(T, seed=300 + 10 * ridx + rank, device=dev)
        ids, wts = _make_routing(
            T,
            e_total,
            TOPK,
            seed=400 + 10 * ridx + rank,
            device=dev,
            mode=mode,
            rank=rank,
            e_local=E_TOTAL,
        )
        out_r = mm_r(x, ids, wts).clone()
        torch.cuda.synchronize()
        out_g = mm_g(x, ids, wts).clone()
        torch.cuda.synchronize()
        assert torch.isfinite(out_g).all(), f"rank {rank} {mode}: NaN/Inf"
        if T > 0:
            ref = _dequant_reference(x, ids, wts, fp8_full, e_total)
            err_r, err_g = _err(out_r, ref), _err(out_g, ref)
            cos_g = _cos(out_g, ref)
            assert cos_g > 0.997, f"rank {rank} {mode}: grouped cos {cos_g:.5f}"
            assert err_g <= err_r * 1.05 + 1e-7, (
                f"rank {rank} {mode}: grouped err {err_g:.5f} > "
                f"1.05 * per-route {err_r:.5f}"
            )
    dist.barrier()


@requires_dist
def test_dist_dedup_grouped_random():
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world
    # All pipes are collectives: construct in the same order on every rank.
    _, mm_r, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    _, mm_g, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        grouped_combine=True,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    _, mm_dg, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        dedup=True,
        grouped_combine=True,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    mm_dg.configure_workspace()
    dev = torch.device("cuda", rank)
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13, w2 = _make_weights(e_total, seed=7, device=dev)
    fp8_full = transform_weights_for_sm90_push(w13, w2)

    x = _make_x(T_CAP, seed=500 + rank, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, TOPK, seed=600 + rank, device=dev)
    out_r = mm_r(x, ids, wts).clone()
    torch.cuda.synchronize()
    out_g = mm_g(x, ids, wts).clone()
    torch.cuda.synchronize()
    out_dg = mm_dg(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_dg, out_g), (
        f"rank {rank}: dedup+grouped != grouped-only (bitwise) -- dedup is "
        "no longer transparent to the combine stage"
    )
    ref = _dequant_reference(x, ids, wts, fp8_full, e_total)
    err_r, err_dg = _err(out_r, ref), _err(out_dg, ref)
    cos_dg = _cos(out_dg, ref)
    assert cos_dg > 0.997, f"rank {rank}: dedup+grouped cos {cos_dg:.5f}"
    assert err_dg <= err_r * 1.05 + 1e-7, (
        f"rank {rank}: dedup+grouped err {err_dg:.5f} > 1.05 * per-route {err_r:.5f}"
    )
    dist.barrier()


@requires_dist
def test_dist_dedup_grouped_hot():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: T_CAP,
        seed=105,
        mode="hot",
        dedup=True,
        grouped_combine=True,
    )


@requires_dist
def test_dist_dedup_grouped_all_remote():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: T_CAP,
        seed=115,
        mode="all_remote",
        dedup=True,
        grouped_combine=True,
    )


@requires_dist
def test_dist_dedup_grouped_uneven():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: 0 if r == 1 else max(T_CAP - 13 * r, 1),
        seed=125,
        dedup=True,
        grouped_combine=True,
        t_of_rank_next=lambda r: T_CAP,
    )


@requires_dist
def test_dist_dedup_grouped_fused_bitwise_equal():
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world
    # Both pipes are collectives: construct in the same order on every rank.
    _, mm_dg, _, _ = _build(
        dedup=True,
        grouped_combine=True,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    _, mm_all, _, _ = _build(
        dedup=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
    )
    dev = torch.device("cuda", rank)
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13, w2 = _make_weights(e_total, seed=7, device=dev)
    fp8_full = transform_weights_for_sm90_push(w13, w2)

    x = _make_x(T_CAP, seed=700 + rank, device=dev)
    ids, wts = _make_routing(T_CAP, e_total, TOPK, seed=800 + rank, device=dev)
    out_dg = mm_dg(x, ids, wts).clone()
    torch.cuda.synchronize()
    out_all = mm_all(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out_all, out_dg), (
        f"rank {rank}: dedup+grouped+fused != dedup+grouped (bitwise) -- "
        "the FC1 fusion is no longer transparent under the joint flags"
    )
    ref = _dequant_reference(x, ids, wts, fp8_full, e_total)
    cos = _cos(out_all, ref)
    assert cos > 0.997, f"rank {rank}: dedup+grouped+fused cos {cos:.5f}"
    dist.barrier()


@requires_dist
def test_dist_dedup_grouped_fused_uneven():
    _dist_forward_and_check(
        "fp8",
        "fp8",
        lambda r: 0 if r == 1 else max(T_CAP - 13 * r, 1),
        seed=135,
        dedup=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
        t_of_rank_next=lambda r: T_CAP,
    )


def _dist_fingerprint_mismatch_impl(cfg_of_rank):
    """Layout-affecting config disagreement must fail fast on every rank."""
    import torch.distributed as dist

    from flashinfer.fused_moe.sm90_push_a2a import _Sm90PushPipe

    rank, world, comm = _dist_setup()
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        _Sm90PushPipe(
            ep_size=world,
            rank=rank,
            num_local_experts=E_TOTAL,
            hidden_size=H,
            top_k=TOPK,
            token_capacity=T_CAP,
            device_index=rank,
            config=cfg_of_rank(rank),
            comm_backend=comm,
        )
    dist.barrier()


@requires_dist
def test_dist_fingerprint_mismatch_grouped():
    from flashinfer.fused_moe import Sm90PushConfig

    _dist_fingerprint_mismatch_impl(lambda r: Sm90PushConfig(grouped_combine=(r == 0)))


@requires_dist
def test_dist_fingerprint_mismatch_dedup():
    from flashinfer.fused_moe import Sm90PushConfig

    _dist_fingerprint_mismatch_impl(lambda r: Sm90PushConfig(dedup_dispatch=(r == 0)))


@requires_dist
def test_dist_round_skew():
    _dist_forward_and_check("fp8", "fp8", lambda r: T_CAP, seed=80, skew_rank=0)


def _dist_graph_replay_impl(
    grouped_combine: bool, dedup: bool = False, fuse_fc1_epilogue: bool = False
):
    """Collective full-forward CUDA graph: capture once, replay bitwise-equal."""
    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world
    pipe, mm, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
        dedup=dedup,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    dev = pipe.device

    xs = [_make_x(T_CAP, seed=91 + 10 * i + rank, device=dev) for i in range(2)]
    routing = [
        _make_routing(T_CAP, e_total, TOPK, seed=991 + 10 * i + rank, device=dev)
        for i in range(2)
    ]
    # eager references first (each is one full-rank collective round)
    eager = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        eager.append(mm(x, ids, wts).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(xs[0])
    static_ids = torch.empty_like(routing[0][0])
    static_w = torch.empty_like(routing[0][1])
    static_x.copy_(xs[0])
    static_ids.copy_(routing[0][0])
    static_w.copy_(routing[0][1])

    for _ in range(2):  # warmup on a side stream, as the capture will run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mm(static_x, static_ids, static_w)
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = mm(static_x, static_ids, static_w)

    outs = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_w.copy_(wts)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(static_out.clone())

    assert not torch.equal(outs[0], outs[1]), "replay ignored the input change"
    for i in range(2):
        assert torch.equal(outs[i], eager[i]), (
            f"rank {rank}: graph replay {i} != eager (bitwise)"
        )
    import torch.distributed as dist

    dist.barrier()


@requires_dist
def test_dist_graph_replay():
    _dist_graph_replay_impl(grouped_combine=False)


@requires_dist
def test_dist_graph_replay_grouped():
    _dist_graph_replay_impl(grouped_combine=True)


@requires_dist
def test_dist_dedup_grouped_graph_replay():
    _dist_graph_replay_impl(grouped_combine=True, dedup=True)


@requires_dist
def test_dist_dedup_grouped_fused_graph_replay():
    _dist_graph_replay_impl(grouped_combine=True, dedup=True, fuse_fc1_epilogue=True)


def _dist_soak_impl(
    grouped_combine: bool, dedup: bool = False, fuse_fc1_epilogue: bool = False
):
    """Multi-rank protocol soak (the real gate for a NEW sync protocol):"""
    import random as _random

    import torch.distributed as dist

    rounds = int(os.environ.get("SM90_PUSH_SOAK_ROUNDS", "100"))
    assert rounds >= 1, "SM90_PUSH_SOAK_ROUNDS must be >= 1 (0 passes vacuously)"
    rank, world, comm = _dist_setup()
    e_total = E_TOTAL * world
    e_local = E_TOTAL
    pipe, mm, _, _ = _build(
        payload_dtype="fp8",
        combine_dtype="fp8",
        device_index=rank,
        ep=world,
        rank=rank,
        comm=comm,
        e_total=e_total,
        dedup=dedup,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    dev = pipe.device
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13, w2 = _make_weights(e_total, seed=7, device=dev)
    fp8_full = transform_weights_for_sm90_push(w13, w2)

    rng = _random.Random(4242 + rank)  # rank-decorrelated T / skew
    t_choices = [0, 1, 13, T_CAP // 2, T_CAP]
    modes = ["random", "hot", "all_remote"]
    pending = []  # every round's stream-ordered output snapshot until the drain
    for r in range(rounds):
        T = rng.choice(t_choices)
        mode = modes[r % len(modes)]  # same pattern class on all ranks
        if rng.random() < 0.2:
            torch.cuda._sleep(int(rng.random() * 1e8))  # up to ~50ms stream skew
        x = _make_x(T, seed=3000 + 17 * r + rank, device=dev)
        ids, wts = _make_routing(
            T,
            e_total,
            TOPK,
            seed=5000 + 31 * r + rank,
            device=dev,
            mode=mode,
            rank=rank,
            e_local=e_local,
        )
        out = mm(x, ids, wts)
        pending.append((r, mode, x, ids, wts, out.clone()))
        if r % 10 == 9 or r == rounds - 1:
            torch.cuda.synchronize()
            for rr, mmode, xx, iids, wwts, oo in pending:
                if xx.shape[0] == 0:
                    continue
                assert torch.isfinite(oo).all(), f"rank {rank} soak round {rr}: NaN/Inf"
                ref = _dequant_reference(xx, iids, wwts, fp8_full, e_total)
                cos = _cos(oo, ref)
                assert cos > 0.997, (
                    f"rank {rank} soak round {rr} ({mmode}, T={xx.shape[0]}): "
                    f"cos {cos:.5f}"
                )
            pending.clear()
            dist.barrier()  # re-align hosts; keeps queue depth bounded
    dist.barrier()


@requires_dist
def test_dist_soak():
    _dist_soak_impl(grouped_combine=False)


@requires_dist
def test_dist_soak_grouped():
    _dist_soak_impl(grouped_combine=True)


@requires_dist
def test_dist_dedup_grouped_soak():
    _dist_soak_impl(grouped_combine=True, dedup=True)


@requires_dist
def test_dist_dedup_grouped_fused_soak():
    _dist_soak_impl(grouped_combine=True, dedup=True, fuse_fc1_epilogue=True)
