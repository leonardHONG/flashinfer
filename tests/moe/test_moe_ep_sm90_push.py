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

Public-entry gates for the SM90 push MegaMoE EP backend.
"""

import os

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

# same small-but-nontrivial shapes as the low-level suite
H, I, E_TOTAL, TOPK, T_CAP = 512, 768, 4, 2, 64

# windows are peer-mapped VMM allocations; keep pipes alive to process exit
_KEEP_ALIVE = []


def _make_weights(E, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H, generator=g) * (H**-0.5)).to(
        device=device, dtype=torch.bfloat16
    )
    w2 = (torch.randn(E, H, I, generator=g) * (I**-0.5)).to(
        device=device, dtype=torch.bfloat16
    )
    return w13, w2


def _make_inputs(T, e_total, top_k, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(T, H, generator=g).to(device=device, dtype=torch.bfloat16)
    ids = (
        torch.randn(T, e_total, generator=g).topk(top_k, dim=1).indices.to(torch.int32)
    )
    w = torch.rand(T, top_k, generator=g) + 0.1
    w = w / w.sum(dim=1, keepdim=True)
    return x, ids.to(device), w.to(device=device, dtype=torch.float32)


def _compute_config(e_total, top_k):
    from flashinfer.fused_moe import (
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=e_total, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
        experts=ExpertConfig(intermediate_size=I),
    )


def _build_layer(world, rank, e_total, device, seed=7, fuse_fc1_epilogue=True):
    """Public construction: weights into a MoEWeightPack, then MoEEpLayer."""
    from flashinfer.fused_moe import MoEWeightPack
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpLayer,
        Sm90PushEpConfig,
    )

    e_local = e_total // world
    w13, w2 = _make_weights(e_total, seed=seed, device=device)
    lo, hi = rank * e_local, (rank + 1) * e_local
    pack = MoEWeightPack()
    pack.prepare_for(
        Sm90PushEpConfig.WEIGHT_VIEW_KEY,
        Sm90PushEpConfig.prepare_weights(
            w13[lo:hi],
            w2[lo:hi],
            fuse_fc1_epilogue=fuse_fc1_epilogue,
        ),
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=world, rank=rank),
        fleet_params=FleetParams(
            num_experts=e_total, max_tokens_per_rank=T_CAP, token_hidden_size=H
        ),
        backend=Sm90PushEpConfig(fuse_fc1_epilogue=fuse_fc1_epilogue),
        compute_config=_compute_config(e_total, TOPK),
        weights=pack,
    )
    _KEEP_ALIVE.append(layer)
    return layer, w13, w2


def _dequant_reference(x, ids, wts, w13, w2, e_total):
    """fp32 oracle on dequantized fp8-blockscale weights."""
    from flashinfer.fused_moe import transform_weights_for_sm90_push

    w13_fp8, w13_sf, w2_fp8, w2_sf = transform_weights_for_sm90_push(w13, w2)
    w13d = torch.stack(
        [dequant_weight_128x128(w13_fp8[e], w13_sf[e]) for e in range(e_total)]
    )
    w2d = torch.stack(
        [dequant_weight_128x128(w2_fp8[e], w2_sf[e]) for e in range(e_total)]
    )
    return reference_moe(x, w13d, w2d, ids, wts)


def _cos(a, b):
    return float(
        torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        )
    )


def test_moe_ep_sm90_layer_requires_compute_and_weights():
    from flashinfer.fused_moe import MoEWeightPack
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpLayer,
        NcclEpConfig,
        Sm90PushEpConfig,
    )

    bs = BootstrapConfig(world_size=1, rank=0)
    fp = FleetParams(
        num_experts=E_TOTAL, max_tokens_per_rank=T_CAP, token_hidden_size=H
    )
    # whole-layer backend without compute_config/weights: rejected at once
    with pytest.raises(ValueError, match="whole-layer"):
        MoEEpLayer(bootstrap=bs, fleet_params=fp, backend=Sm90PushEpConfig())
    # split backend WITH compute_config: would be silently ignored -> reject
    with pytest.raises(ValueError, match="whole-layer"):
        MoEEpLayer(
            bootstrap=bs,
            fleet_params=fp,
            backend=NcclEpConfig(),
            compute_config=_compute_config(E_TOTAL, TOPK),
            weights=MoEWeightPack(),
        )


def test_moe_ep_sm90_backend_config_errors(monkeypatch):
    """Config/weight errors surface as the guarded phases' RuntimeError."""
    import dataclasses

    import flashinfer.fused_moe.sm90_push_a2a as _low
    from flashinfer.fused_moe import MoEWeightPack, QuantConfig, QuantVariant
    from flashinfer.moe_ep import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.split_backends.sm90_push import (
        _Sm90PushEpBackend,
        Sm90PushEpConfig,
    )

    bs = BootstrapConfig(world_size=1, rank=0)
    fp = FleetParams(
        num_experts=E_TOTAL, max_tokens_per_rank=T_CAP, token_hidden_size=H
    )
    good = _compute_config(E_TOTAL, TOPK)

    with pytest.raises(RuntimeError, match="init_timeout_s"):
        _Sm90PushEpBackend(
            Sm90PushEpConfig(init_timeout_s=0), bs, fp, good, MoEWeightPack()
        )

    bad_quant = dataclasses.replace(good, quant=QuantConfig(variant=QuantVariant.Nvfp4))
    with pytest.raises(RuntimeError, match="DeepSeekFp8"):
        _Sm90PushEpBackend(Sm90PushEpConfig(), bs, fp, bad_quant, MoEWeightPack())

    # single-rank stub comm: the guarded phase runs without a live group
    monkeypatch.setattr(
        _low, "_default_comm_backend", lambda n: _low._SingleRankBackend()
    )
    bs4 = BootstrapConfig(world_size=3, rank=0)
    fp3 = FleetParams(num_experts=4, max_tokens_per_rank=T_CAP, token_hidden_size=H)
    with pytest.raises(RuntimeError, match="topology/fingerprint mismatch"):
        _Sm90PushEpBackend(Sm90PushEpConfig(), bs4, fp3, good, MoEWeightPack())

    # missing weight view
    with pytest.raises(RuntimeError, match="sm90_push_fp8_block"):
        _Sm90PushEpBackend(Sm90PushEpConfig(), bs, fp, good, MoEWeightPack())

    # interleave tag disagreeing with fuse_fc1_epilogue (CPU-fabricated view)
    pack = MoEWeightPack()
    pack.prepare_for(
        Sm90PushEpConfig.WEIGHT_VIEW_KEY,
        {
            "w13_fp8": torch.zeros(E_TOTAL, 2 * I, H, dtype=torch.float8_e4m3fn),
            "w13_sf": torch.zeros(E_TOTAL, 2 * I // 128, H // 128),
            "w2_fp8": torch.zeros(E_TOTAL, H, I, dtype=torch.float8_e4m3fn),
            "w2_sf": torch.zeros(E_TOTAL, H // 128, I // 128),
            "w13_interleaved": torch.tensor(False),
        },
    )
    with pytest.raises(RuntimeError, match="interleave"):
        _Sm90PushEpBackend(Sm90PushEpConfig(), bs, fp, good, pack)


@requires_sm90
def test_moe_ep_sm90_ep1_forward_bf16():
    from flashinfer.moe_ep import MoEEpTensors

    dev = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, E_TOTAL, dev)
    x, ids, wts = _make_inputs(T_CAP, E_TOTAL, TOPK, seed=11, device=dev)

    out = layer(MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=wts))
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16, "public output must match hidden_states dtype"
    assert out.shape == (T_CAP, H)
    assert torch.isfinite(out.float()).all()

    ref = _dequant_reference(x, ids, wts, w13, w2, E_TOTAL)
    assert _cos(out, ref) > 0.997

    # low-level fp32 anchor: same flags, same weights, fp32 reduce output
    from flashinfer.fused_moe.sm90_push_a2a import (
        _Sm90PushMoERunner,
        _Sm90PushPipe,
        Sm90PushConfig,
        make_sm90_push_weights,
    )

    pipe = _Sm90PushPipe(
        ep_size=1,
        rank=0,
        num_local_experts=E_TOTAL,
        hidden_size=H,
        top_k=TOPK,
        token_capacity=T_CAP,
        device_index=0,
        config=Sm90PushConfig(
            dedup_dispatch=True, grouped_combine=True, fuse_fc1_epilogue=True
        ),
    )
    _KEEP_ALIVE.append(pipe)
    mm = _Sm90PushMoERunner(
        pipe, make_sm90_push_weights(w13, w2, interleave_gate_up=True)
    )
    out_f32 = mm(x, ids, wts).clone()
    torch.cuda.synchronize()
    assert torch.equal(out, out_f32.to(torch.bfloat16)), (
        "public bf16 out != RN(bf16) of the fp32 reference reduction (bitwise)"
    )


@requires_sm90
def test_moe_ep_sm90_ep1_staging_errors():
    from flashinfer.moe_ep import MoEEpTensors

    dev = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, E_TOTAL, dev, seed=8)
    x, ids, wts = _make_inputs(T_CAP, E_TOTAL, TOPK, seed=21, device=dev)

    with pytest.raises(ValueError, match="BF16"):
        layer(MoEEpTensors(hidden_states=x.half(), topk_ids=ids, topk_weights=wts))
    with pytest.raises(ValueError, match="int32"):
        layer(MoEEpTensors(hidden_states=x, topk_ids=ids.long(), topk_weights=wts))
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        big_x, big_ids, big_w = _make_inputs(
            T_CAP + 1, E_TOTAL, TOPK, seed=22, device=dev
        )
        layer(MoEEpTensors(hidden_states=big_x, topk_ids=big_ids, topk_weights=big_w))


@requires_sm90
def test_moe_ep_sm90_ep1_graph_replay():
    from flashinfer.moe_ep import MoEEpTensors

    dev = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, E_TOTAL, dev, seed=9)

    xs, routing = [], []
    for i in range(2):
        x, ids, wts = _make_inputs(T_CAP, E_TOTAL, TOPK, seed=31 + i, device=dev)
        xs.append(x)
        routing.append((ids, wts))
    eager = []
    for x, (ids, wts) in zip(xs, routing, strict=True):
        eager.append(
            layer(MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=wts)).clone()
        )
        torch.cuda.synchronize()

    static_x = torch.empty_like(xs[0]).copy_(xs[0])
    static_ids = torch.empty_like(routing[0][0]).copy_(routing[0][0])
    static_w = torch.empty_like(routing[0][1]).copy_(routing[0][1])
    t = MoEEpTensors(hidden_states=static_x, topk_ids=static_ids, topk_weights=static_w)

    for _ in range(2):  # warmup on a side stream, as the capture will run
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            layer(t)
        torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = layer(t)

    outs = []
    for i, (x, (ids, wts)) in enumerate(zip(xs, routing, strict=True)):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_w.copy_(wts)
        graph.replay()
        torch.cuda.synchronize()
        outs.append(static_out.clone())
        assert torch.equal(outs[-1], eager[i]), (
            f"public graph replay {i} != public eager (bitwise)"
        )
    assert not torch.equal(outs[0], outs[1]), "public replay ignored input changes"


@requires_sm90
def test_moe_ep_sm90_ep1_output_not_aliased():
    from flashinfer.moe_ep import MoEEpTensors

    dev = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, E_TOTAL, dev, seed=10)
    x1, ids1, wt1 = _make_inputs(T_CAP, E_TOTAL, TOPK, seed=51, device=dev)
    x2, ids2, wt2 = _make_inputs(T_CAP, E_TOTAL, TOPK, seed=52, device=dev)
    out1 = layer(MoEEpTensors(hidden_states=x1, topk_ids=ids1, topk_weights=wt1))
    torch.cuda.synchronize()
    snap = out1.clone()
    out2 = layer(MoEEpTensors(hidden_states=x2, topk_ids=ids2, topk_weights=wt2))
    torch.cuda.synchronize()
    assert torch.equal(out1, snap), "second forward overwrote the first return"
    assert not torch.equal(out1, out2), "distinct inputs must give distinct outputs"


@requires_sm90
def test_moe_ep_sm90_ep1_module_contract():
    dev = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, E_TOTAL, dev, seed=12)
    sd = layer.state_dict()
    for key in (
        "sm90_push_w13_fp8",
        "sm90_push_w13_sf",
        "sm90_push_w2_fp8",
        "sm90_push_w2_sf",
        "sm90_push_w13_interleaved",
    ):
        assert key in sd, f"weight view {key} missing from state_dict"
    assert sd["sm90_push_w13_interleaved"].dtype == torch.bool
    assert sd["sm90_push_w13_interleaved"].numel() == 1
    assert bool(sd["sm90_push_w13_interleaved"].item()) is True
    with pytest.raises(RuntimeError, match="cannot be moved"):
        layer.to("cpu")
    with pytest.raises(RuntimeError, match="cannot be moved"):
        layer.half()
    layer.to(dev)
    layer.to("cuda")
    layer.load_state_dict(sd)
    with pytest.raises(RuntimeError, match="assign=True"):
        layer.load_state_dict(sd, assign=True)

    layer_plain, _, _ = _build_layer(
        1, 0, E_TOTAL, dev, seed=13, fuse_fc1_epilogue=False
    )
    sd_bad = layer_plain.state_dict()
    before = {
        key: value.clone()
        for key, value in layer.state_dict().items()
        if key.startswith("sm90_push_")
    }
    with pytest.raises(RuntimeError, match="interleave_gate_up"):
        layer.load_state_dict(sd_bad)
    with pytest.raises(RuntimeError, match="interleave_gate_up"):
        layer.load_state_dict(sd_bad, strict=False)
    for key, value in before.items():
        assert torch.equal(layer.state_dict()[key], value)

    sd_missing = dict(sd)
    del sd_missing["sm90_push_w13_interleaved"]
    with pytest.raises(RuntimeError, match="without sm90_push_w13_interleaved"):
        layer.load_state_dict(sd_missing, strict=False)
    for key, value in before.items():
        assert torch.equal(layer.state_dict()[key], value)


def test_moe_ep_sm90_layer_tag_state_cpu():
    import torch.nn as nn

    from flashinfer.moe_ep import Sm90PushEpConfig
    from flashinfer.moe_ep.layer import MoEEpLayer

    class _FakePipe:
        device = torch.device("cpu")

    class _FakeBackend:
        _pipe = _FakePipe()
        w13_interleaved = True

        def weight_buffers(self):
            return {"w13_fp8": self._w, "w13_sf": self._s}

        _w = torch.zeros(4, 8, dtype=torch.uint8)
        _s = torch.zeros(4, dtype=torch.float32)

    layer = MoEEpLayer.__new__(MoEEpLayer)
    nn.Module.__init__(layer)
    layer._backend = Sm90PushEpConfig()  # fuse_fc1_epilogue=True default
    layer._fleet = None
    layer._is_whole_layer = True
    layer._whole_layer = _FakeBackend()
    for name, tensor in layer._whole_layer.weight_buffers().items():
        layer.register_buffer(f"sm90_push_{name}", tensor, persistent=True)
    layer.register_buffer(
        "sm90_push_w13_interleaved", torch.tensor(True), persistent=True
    )

    sd = layer.state_dict()
    assert bool(sd["sm90_push_w13_interleaved"].item()) is True
    layer.load_state_dict(sd)
    layer._apply(lambda t: t)
    before = {name: tensor.clone() for name, tensor in layer._buffers.items()}
    pointers = {name: tensor.data_ptr() for name, tensor in layer._buffers.items()}
    with pytest.raises(RuntimeError):
        layer._apply(lambda t: t.clone())
    with pytest.raises(RuntimeError):
        layer.bfloat16()
    for name, tensor in layer._buffers.items():
        assert tensor.data_ptr() == pointers[name]
        assert torch.equal(tensor, before[name])


@requires_dist
def test_dist_moe_ep_sm90_independent_ep1():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    layer, _, _ = _build_layer(1, 0, E_TOTAL, dev, seed=101 + rank)
    from flashinfer.moe_ep import MoEEpTensors

    x, ids, wts = _make_inputs(T_CAP, E_TOTAL, TOPK, seed=111 + rank, device=dev)
    out = layer(MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=wts))
    torch.cuda.synchronize()
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()
    dist.barrier()


@requires_dist
def test_dist_moe_ep_sm90_forward():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    e_total = E_TOTAL * world
    layer, w13, w2 = _build_layer(world, rank, e_total, dev)
    from flashinfer.moe_ep import MoEEpTensors

    x, ids, wts = _make_inputs(T_CAP, e_total, TOPK, seed=41 + rank, device=dev)
    out = layer(MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=wts))
    torch.cuda.synchronize()
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all(), f"rank {rank}: NaN/Inf"
    ref = _dequant_reference(x, ids, wts, w13, w2, e_total)
    cos = _cos(out, ref)
    assert cos > 0.997, f"rank {rank}: public-entry cos {cos:.5f}"
    dist.barrier()
