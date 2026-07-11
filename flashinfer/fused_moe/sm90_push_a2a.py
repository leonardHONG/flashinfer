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

SM90 push-based MegaMoE (single-node NVLink expert parallelism).
"""

from __future__ import annotations

import atexit
import contextlib
import weakref
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import torch

from ..api_logging import flashinfer_api
from ..jit.fused_moe import gen_sm90_push_a2a_module
from ..trace.templates.sm90_push_moe import transform_weights_for_sm90_push_trace

__all__ = [
    "Sm90PushPayload",
    "Sm90PushCombine",
    "Sm90PushConfig",
    "Sm90PushWeights",
    "make_sm90_push_weights",
    "transform_weights_for_sm90_push",
]


class Sm90PushPayload(Enum):
    """Dispatch payload dtype: FP8 quantizes 1x128 at the source; BF16 is the debug anchor."""

    FP8 = "fp8"
    BF16 = "bf16"


class Sm90PushCombine(Enum):
    """Combine partial dtype: FP8 halves combine ingress; BF16 is the debug anchor."""

    FP8 = "fp8"
    BF16 = "bf16"


@dataclass(frozen=True)
class Sm90PushConfig:
    """Construction-time knobs (frozen; the hot path takes no strings)."""

    payload_dtype: Sm90PushPayload = Sm90PushPayload.FP8
    combine_dtype: Sm90PushCombine = Sm90PushCombine.FP8
    fuse_act: bool = True
    capacity_factor: float = 1.0
    dedup_dispatch: bool = False
    grouped_combine: bool = False
    fuse_fc1_epilogue: bool = False


# Window views carry C++ deleters that must not run during interpreter
# finalization: atexit drains them early; the window itself is left to the
# OS since a peer may still map it at exit.
_LIVE_PIPES: list = []

# Deliberate process-lifetime keepalive (~100 B/view): pack_strided_memory's
# ctypes DLManagedTensor wrapper must outlive every derived view's storage,
# or torch calls the deleter through a freed struct at storage destruction.
_CAPSULE_KEEPALIVE: list = []


def _drain_live_pipes() -> None:
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    for ref in _LIVE_PIPES:
        pipe = ref()
        if pipe is not None:
            pipe._release_window_views()
    _LIVE_PIPES.clear()


atexit.register(_drain_live_pipes)


def _record_stage(name: str, enabled: bool):
    """Host-side stage range (a few us/round when enabled) for profiling."""
    return torch.profiler.record_function(name) if enabled else contextlib.nullcontext()


def _align(x: int, a: int = 128) -> int:
    return (x + a - 1) // a * a


class _SingleRankBackend:
    """Trivial comm backend for ep_size == 1 (no MPI / torch.distributed)."""

    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def allgather(self, data):
        return [data]

    def bcast(self, data, root: int = 0):
        return data

    def barrier(self) -> None:
        pass

    def Split(self, color: int, key: int):
        return self


def _default_comm_backend(ep_size: int):
    if ep_size == 1:
        return _SingleRankBackend()
    import torch.distributed as dist

    if dist.is_initialized():
        from ..comm.mnnvl import TorchDistBackend

        return TorchDistBackend()
    from ..comm.mnnvl import MPIBackend

    return MPIBackend()


def _run_guarded_phase(comm, rank: int, name: str, fn):
    """Run fn locally, then unconditionally allgather per-rank (error, payload) reports."""
    err, payload = None, None
    try:
        payload = fn()
    except Exception as exc:
        err = f"rank {rank}: {type(exc).__name__}: {exc}"
    reports = comm.allgather((err, payload))
    failures = [e for e, _ in reports if e is not None]
    if failures:
        raise RuntimeError(
            f"sm90_push init phase '{name}' failed on "
            f"{len(failures)}/{len(reports)} rank(s); all ranks abort "
            f"together: " + " | ".join(failures)
        )
    return [p for _, p in reports]


def _run_phase0_handshake(
    comm, reported_world: int, reported_rank: int, fingerprint, validate=None
):
    comm_rank = comm.Get_rank()

    def _probe():
        if validate is not None:
            validate()
        return reported_world, reported_rank, fingerprint

    reports = _run_guarded_phase(comm, comm_rank, "validate", _probe)
    comm_world = len(reports)
    bad = [
        i
        for i, (world, rank, peer_fingerprint) in enumerate(reports)
        if world != comm_world or rank != i or peer_fingerprint != fingerprint
    ]
    if bad:
        raise RuntimeError(
            f"sm90_push topology/fingerprint mismatch at rank(s) {bad}: "
            f"communicator size={comm_world}, local reported rank/world="
            f"({reported_rank}, {reported_world})"
        )
    return reports


class _Sm90PushPipe:
    """Symmetric-window owner + device round protocol for the SM90 push path."""

    def __init__(
        self,
        *,
        ep_size: int,
        rank: int,
        num_local_experts: int,
        hidden_size: int,
        top_k: int,
        token_capacity: int,
        device_index: int,
        config: Sm90PushConfig | None = None,
        comm_backend=None,
        out_dtype: torch.dtype = torch.float32,
        allow_unverified_p2p: bool = False,
    ):
        if config is None:
            config = Sm90PushConfig()
        if ep_size < 1 or ep_size > 32:
            raise ValueError(f"ep_size must be in [1, 32], got {ep_size}")
        comm = (
            comm_backend if comm_backend is not None else _default_comm_backend(ep_size)
        )
        comm_size, comm_rank = comm.Get_size(), comm.Get_rank()

        def _validate_arguments():
            if not isinstance(config, Sm90PushConfig):
                raise ValueError(
                    f"config must be Sm90PushConfig, got {type(config).__name__}"
                )
            if not isinstance(config.payload_dtype, Sm90PushPayload):
                raise ValueError(
                    f"payload_dtype must be Sm90PushPayload, got {config.payload_dtype!r}"
                )
            if not isinstance(config.combine_dtype, Sm90PushCombine):
                raise ValueError(
                    f"combine_dtype must be Sm90PushCombine, got {config.combine_dtype!r}"
                )
            if out_dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(
                    f"out_dtype must be torch.float32 or torch.bfloat16, got {out_dtype}"
                )
            if ep_size < 1 or ep_size > 32:
                raise ValueError(f"ep_size must be in [1, 32], got {ep_size}")
            if num_local_experts < 1:
                raise ValueError(
                    f"num_local_experts must be >= 1, got {num_local_experts}"
                )
            if hidden_size < 128 or hidden_size % 128 != 0:
                raise ValueError(
                    f"hidden_size must be a positive multiple of 128, got {hidden_size}"
                )
            if top_k not in (1, 2, 4, 6, 8):
                raise ValueError(f"top_k must be one of (1, 2, 4, 6, 8), got {top_k}")
            if token_capacity < 1:
                raise ValueError(f"token_capacity must be >= 1, got {token_capacity}")
            if not (0.0 < config.capacity_factor <= 1.0):
                raise ValueError(
                    f"capacity_factor must be in (0, 1], got {config.capacity_factor}"
                )
            if config.grouped_combine and config.combine_dtype != Sm90PushCombine.FP8:
                raise ValueError(
                    "grouped_combine requires combine_dtype=Sm90PushCombine.FP8"
                )
            if config.fuse_fc1_epilogue and not config.fuse_act:
                raise ValueError("fuse_fc1_epilogue=True requires fuse_act=True")
            return None

        argument_fingerprint = (
            num_local_experts,
            hidden_size,
            top_k,
            token_capacity,
            str(out_dtype),
            repr(getattr(config, "payload_dtype", None)),
            repr(getattr(config, "combine_dtype", None)),
            getattr(config, "fuse_act", None),
            getattr(config, "capacity_factor", None),
            getattr(config, "dedup_dispatch", None),
            getattr(config, "grouped_combine", None),
            getattr(config, "fuse_fc1_epilogue", None),
            bool(allow_unverified_p2p),
        )
        _run_phase0_handshake(
            comm,
            ep_size,
            rank,
            argument_fingerprint,
            _validate_arguments,
        )
        ep_size, rank = comm_size, comm_rank

        from ..comm import pack_strided_memory
        from ..comm.mnnvl import SymmDeviceMemory

        self.ep, self.rank = ep_size, rank
        self.E, self.H, self.K, self.token_capacity = (
            num_local_experts,
            hidden_size,
            top_k,
            token_capacity,
        )
        self.config = config
        self.device_index = device_index

        max_recv_routes = ep_size * token_capacity * top_k
        if max_recv_routes > 2**31 - 1:
            raise ValueError(
                f"ep_size * token_capacity * top_k = {max_recv_routes} overflows the "
                "packed reservation counter"
            )
        dedup = config.dedup_dispatch
        self.meta_rows = max(int(config.capacity_factor * max_recv_routes), 1)
        self.pool_rows = (
            max(int(config.capacity_factor * ep_size * token_capacity), 1)
            if dedup
            else self.meta_rows
        )
        self.m_ws = max_recv_routes
        # compute-row capacity: GEMM rows == meta records this rank can hold
        self.m_cap = self.meta_rows
        fp8_payload = config.payload_dtype == Sm90PushPayload.FP8
        self.bytes_per_row = hidden_size if fp8_payload else 2 * hidden_size
        nkb = hidden_size // 128
        E, eps = num_local_experts, ep_size

        off = 0
        self.pool_offset = off
        off = _align(off + self.pool_rows * self.bytes_per_row)
        self.pool_sc_offset = off
        off = _align(off + (self.pool_rows * nkb * 4 if fp8_payload else 0))
        self.pool_meta_offset = off
        off = _align(off + self.meta_rows * 16)
        self.pool_head_offset = off
        off = _align(off + 8)
        self.base_cells_offset = off
        off = _align(off + eps * 8)
        self.count_cells_offset = off
        off = _align(off + E * eps * 8)
        self.cdone_cells_offset = off
        off = _align(off + eps * 8)
        self.ack_cells_offset = off
        off = _align(off + eps * 8)
        fp8_combine = config.combine_dtype == Sm90PushCombine.FP8
        self.combine_slots = ep_size if config.grouped_combine else top_k
        cslots = self.combine_slots
        self.combine_offset = off
        off = _align(
            off + (0 if fp8_combine else token_capacity * top_k * hidden_size * 2)
        )
        self.cfp8_offset = off
        off = _align(
            off + (token_capacity * cslots * hidden_size if fp8_combine else 0)
        )
        self.csc_offset = off
        off = _align(off + (token_capacity * cslots * nkb * 4 if fp8_combine else 0))
        self.total_bytes = off

        self._comm = comm

        def _phase(name, fn):
            return _run_guarded_phase(comm, rank, name, fn)

        def _phase_a_probe():
            if not torch.cuda.is_available():
                raise RuntimeError("sm90_push requires CUDA")
            major, _minor = torch.cuda.get_device_capability(device_index)
            if major != 9:
                raise RuntimeError(
                    f"sm90_push requires an SM90 (Hopper) device, got SM{major}x"
                )
            props = torch.cuda.get_device_properties(device_index)
            uuid_str = str(getattr(props, "uuid", ""))
            if "MIG" in props.name or uuid_str.startswith("MIG"):
                raise RuntimeError(
                    "sm90_push does not support MIG slices (the protocol "
                    f"needs whole-GPU NVLink peer mapping); got {props.name}"
                )
            return None

        _phase("validate-device", _phase_a_probe)

        def _phase_b_jit():
            import socket

            self.module = gen_sm90_push_a2a_module().build_and_load()
            props = torch.cuda.get_device_properties(device_index)
            return (socket.gethostname(), device_index, str(getattr(props, "uuid", "")))

        topo = _phase("a2a-jit", _phase_b_jit)

        def _phase_c_peers():
            import warnings

            hosts = {h for h, _, _ in topo}
            if len(hosts) > 1:
                raise RuntimeError(
                    f"sm90_push is single-node only; EP group spans {hosts}"
                )
            my_uuid = topo[rank][2]
            for peer_rank, (_, peer_dev, peer_uuid) in enumerate(topo):
                if peer_rank == rank:
                    continue
                if peer_uuid and peer_uuid == my_uuid:
                    raise RuntimeError(
                        f"rank {rank} and peer rank {peer_rank} report the SAME "
                        f"physical GPU ({peer_uuid}); one GPU cannot host two "
                        "EP ranks of the push protocol"
                    )
                probeable = (
                    peer_dev != device_index and peer_dev < torch.cuda.device_count()
                )
                if not probeable:
                    msg = (
                        f"cannot probe P2P capability between rank {rank} "
                        f"(device {device_index}, {my_uuid or 'uuid?'}) and peer "
                        f"rank {peer_rank} ({peer_uuid or 'uuid?'}): the peer's "
                        f"device index {peer_dev} does not name that GPU in this "
                        "process (per-rank CUDA_VISIBLE_DEVICES masking). "
                        "Unverified: cudaDeviceCanAccessPeer and NVLink-native "
                        "system-scope atomics."
                    )
                    if not allow_unverified_p2p:
                        raise RuntimeError(
                            msg + " Expose all EP GPUs to every rank, or opt in "
                            "explicitly with allow_unverified_p2p=True "
                            "(Sm90PushEpConfig.allow_unverified_p2p)."
                        )
                    warnings.warn(
                        "sm90_push: proceeding with UNVERIFIED peer-to-peer "
                        "capability -- " + msg,
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    continue
                if not torch.cuda.can_device_access_peer(device_index, peer_dev):
                    raise RuntimeError(
                        f"device {device_index} cannot P2P-access peer rank "
                        f"{peer_rank}'s device {peer_dev}"
                    )
                if not self.module.sm90_push_p2p_native_atomics(device_index, peer_dev):
                    raise RuntimeError(
                        f"no NVLink-native system-scope atomics between device "
                        f"{device_index} and peer device {peer_dev} (PCIe-only "
                        "P2P cannot run the push protocol)"
                    )
            return None

        _phase("peer-topology", _phase_c_peers)

        def _phase_d_window():
            self.symm = SymmDeviceMemory(
                buf_size=self.total_bytes,
                group_size=eps,
                group_rank=rank,
                device_idx=device_index,  # SymmDeviceMemory's own kwarg name
                comm_backend_for_handle_transfer=comm,
                enable_multicast=False,
                allocate_signal_pads=False,  # the protocol never uses signal pads
            )
            self.peer_bases = self.symm.get_buffer_ptrs_dev()
            wrapper = getattr(self.peer_bases, "_capsule_wrapper", None)
            if wrapper is not None:  # same keepalive hazard as view()
                _CAPSULE_KEEPALIVE.append(wrapper)
            base = self.symm.get_unicast_ptr(rank)
            dv = torch.device("cuda", device_index)
            self.device = dv

            def view(offset: int, nbytes: int, dtype: torch.dtype) -> torch.Tensor:
                t = pack_strided_memory(
                    base + offset, nbytes, nbytes, 1, dtype, device_index
                )
                wrapper = getattr(t, "_capsule_wrapper", None)
                if wrapper is not None:  # see _CAPSULE_KEEPALIVE
                    _CAPSULE_KEEPALIVE.append(wrapper)
                t = t.reshape(-1)
                # dlpack labels bf16 as fp16 (bits are correct); re-label
                if t.dtype != dtype:
                    t = t.view(dtype)
                return t

            if not fp8_combine:
                self.combine_t = view(
                    self.combine_offset,
                    token_capacity * top_k * hidden_size * 2,
                    torch.bfloat16,
                ).reshape(token_capacity, top_k, hidden_size)
            else:
                self.csc_t = view(
                    self.csc_offset, token_capacity * cslots * nkb * 4, torch.float32
                )

            n_total_experts = E * eps
            self._round = torch.zeros(1, dtype=torch.int32, device=dv)
            if dedup:
                self._lc = torch.zeros(
                    n_total_experts + eps, dtype=torch.int32, device=dv
                )
                self._pc = self._lc[n_total_experts:]
                self._ploff = torch.empty(
                    token_capacity * top_k, dtype=torch.int32, device=dv
                )
                self._pkeybase = torch.empty(eps, dtype=torch.int32, device=dv)
            else:
                self._lc = torch.zeros(n_total_experts, dtype=torch.int32, device=dv)
            self._done = torch.zeros(n_total_experts, dtype=torch.int32, device=dv)
            self._keybase = torch.empty(n_total_experts, dtype=torch.int32, device=dv)
            self._loff = torch.empty(
                token_capacity * top_k, dtype=torch.int32, device=dv
            )
            self._rows_per_src = torch.empty(eps, dtype=torch.int32, device=dv)
            self._cdone_local = torch.empty(eps, dtype=torch.int32, device=dv)
            if fp8_combine and config.grouped_combine:
                self._grp_cnt = torch.empty(
                    eps * token_capacity, dtype=torch.int32, device=dv
                )
                self._grp_rows = torch.empty(
                    eps * token_capacity * top_k, dtype=torch.int32, device=dv
                )
                self._grp_list = torch.empty(
                    eps * token_capacity, dtype=torch.int32, device=dv
                )
                self._n_groups = torch.zeros(1, dtype=torch.int32, device=dv)
                self._groups_per_src = torch.empty(eps, dtype=torch.int32, device=dv)
            self._offsets = torch.empty(E + 1, dtype=torch.int64, device=dv)
            self._seg_src_base = torch.empty(
                n_total_experts, dtype=torch.int32, device=dv
            )
            self._seg_out_base = torch.empty(
                n_total_experts + 1, dtype=torch.int32, device=dv
            )
            self._pad_base = torch.empty(E, dtype=torch.int32, device=dv)
            self._m_dev = torch.zeros(1, dtype=torch.int32, device=dv)
            self._p_dev = torch.zeros(1, dtype=torch.int32, device=dv)
            self._next_row = torch.zeros(1, dtype=torch.int32, device=dv)
            self._out = torch.empty(
                token_capacity, hidden_size, dtype=out_dtype, device=dv
            )

            whole = pack_strided_memory(
                base, self.total_bytes, self.total_bytes, 1, torch.uint8, device_index
            )
            whole.zero_()
            torch.cuda.synchronize()
            return None

        _phase("window+scratch", _phase_d_window)
        self._round_open = False  # host-side single-inflight misuse guard
        _LIVE_PIPES.append(weakref.ref(self))

    def _release_window_views(self) -> None:
        """Drop the dlpack-wrapped window views (atexit; see _drain_live_pipes)."""
        self.combine_t = None
        self.csc_t = None
        self.peer_bases = None

    # 21 layout scalars forwarded to every kernel launch (mirrors LAYOUT_PARAMS)
    def _layout_args(self):
        return (
            self.peer_bases,
            self.ep,
            self.rank,
            self.E,
            self.pool_rows,
            self.meta_rows,
            self.bytes_per_row,
            self.H,
            self.K,
            self.token_capacity,
            self.pool_offset,
            self.pool_sc_offset,
            self.pool_meta_offset,
            self.pool_head_offset,
            self.base_cells_offset,
            self.count_cells_offset,
            self.cdone_cells_offset,
            self.ack_cells_offset,
            self.combine_offset,
            self.cfp8_offset,
            self.csc_offset,
        )

    def proto_begin_round(self) -> None:
        # single-inflight: an unacked prior round means misuse
        if self._round_open:
            raise RuntimeError(
                "sm90_push pipe: previous round was never acked "
                "(proto_ack); the pipe runs ONE round at a time"
            )
        self._round_open = True
        if self.config.combine_dtype == Sm90PushCombine.FP8:
            self.csc_t.zero_()
        else:
            self.combine_t.zero_()
        self.module.sm90_push_bump_tag(self._round)
        self.module.sm90_push_wait_acks(*self._layout_args(), self._round)

    def proto_dispatch(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor
    ) -> None:
        fp8 = self.config.payload_dtype == Sm90PushPayload.FP8
        if self.config.dedup_dispatch:
            fn = (
                self.module.sm90_push_dispatch_dedup_fp8
                if fp8
                else self.module.sm90_push_dispatch_dedup
            )
            fn(
                x,
                topk_ids,
                topk_w,
                *self._layout_args(),
                self._lc,
                self._pc,
                self._loff,
                self._ploff,
                self._done,
                self._keybase,
                self._pkeybase,
                self._round,
            )
            return
        fn = (
            self.module.sm90_push_dispatch_fp8
            if fp8
            else self.module.sm90_push_dispatch
        )
        fn(
            x,
            topk_ids,
            topk_w,
            *self._layout_args(),
            self._lc,
            self._loff,
            self._done,
            self._keybase,
            self._round,
        )

    def proto_wait_prefix(self) -> None:
        self.module.sm90_push_wait_prefix(
            *self._layout_args(),
            self._round,
            self._rows_per_src,
            self._offsets,
            self._seg_src_base,
            self._seg_out_base,
            self._pad_base,
            self._m_dev,
            self._p_dev,
            self._next_row,
            self.m_ws,
        )

    def proto_compact(
        self,
        a_fp8: torch.Tensor,
        sfa: torch.Tensor,
        meta: torch.Tensor,
        row_expert: torch.Tensor,
    ) -> None:
        self.module.sm90_push_compact(
            a_fp8,
            sfa,
            meta,
            row_expert,
            *self._layout_args(),
            self._offsets,
            self._seg_src_base,
            self._seg_out_base,
            self._pad_base,
            self._m_dev,
            self._p_dev,
            self._next_row,
        )

    def proto_silu_mul_quant(
        self,
        h: torch.Tensor,
        a_fp8: torch.Tensor,
        sfa: torch.Tensor,
        row_expert: torch.Tensor,
    ) -> None:
        self.module.sm90_silu_mul_quant_grouped(
            a_fp8,
            sfa,
            h,
            self._offsets,
            self._pad_base,
            self._m_dev,
            self._p_dev,
            row_expert,
            h.shape[0],
        )

    def proto_combine(self, y: torch.Tensor, meta: torch.Tensor) -> None:
        if (
            self.config.combine_dtype == Sm90PushCombine.FP8
            and self.config.grouped_combine
        ):
            self.module.sm90_push_combine_fp8_grouped(
                y,
                meta,
                *self._layout_args(),
                self._m_dev,
                self._grp_cnt,
                self._grp_rows,
                self._grp_list,
                self._n_groups,
                self._groups_per_src,
                self._cdone_local,
                self._round,
            )
            return
        fn = (
            self.module.sm90_push_combine_fp8
            if self.config.combine_dtype == Sm90PushCombine.FP8
            else self.module.sm90_push_combine
        )
        fn(
            y,
            meta,
            *self._layout_args(),
            self._m_dev,
            self._rows_per_src,
            self._cdone_local,
            self._round,
        )

    def proto_wait_combine(self) -> None:
        self.module.sm90_push_wait_combine(*self._layout_args(), self._round)

    def proto_reduce(self, num_tokens: int) -> torch.Tensor:
        """Reduce this rank's combine inbox into the persistent output buffer."""
        if self.config.combine_dtype == Sm90PushCombine.FP8:
            fn = (
                self.module.sm90_combine_reduce_fp8_grouped
                if self.config.grouped_combine
                else self.module.sm90_combine_reduce_fp8
            )
        else:
            fn = self.module.sm90_combine_reduce
        fn(self._out, *self._layout_args(), num_tokens)
        return self._out[:num_tokens]

    def proto_ack(self) -> None:
        """End the round: reset pool head + scratch, release the ack cells."""
        self.module.sm90_push_ack(
            *self._layout_args(), self._round, self._lc, self._done
        )
        self._round_open = False


def _per_block_cast_128x128(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, K) -> e4m3 (N, K) + f32 scales (N/128, K/128), 128x128 blockwise."""
    N, K = w.shape
    if N % 128 != 0 or K % 128 != 0:
        raise ValueError(f"weight ({N}, {K}) must be 128-aligned")
    t = w.float().reshape(N // 128, 128, K // 128, 128)
    amax = t.abs().amax(dim=(1, 3))
    sc = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    q = (
        (t / sc[:, None, :, None])
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .reshape(N, K)
    )
    return q, sc


def _mxfp8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """OCP MXFP8 quantize->dequantize round-trip (test-only value-domain simulation)."""
    N, K = w.shape
    if K % 32 != 0:
        raise ValueError(f"MXFP8 requires K % 32 == 0, got {K}")
    b = w.float().reshape(N, K // 32, 32)
    amax = b.abs().amax(-1)
    _, e = torch.frexp(amax)  # frexp, not log2: exact at power-of-2 boundaries
    shared = (e - 1) - 8  # floor(log2(amax)) - emax(e4m3)
    shared = torch.where(amax > 0, shared, torch.full_like(shared, -127))
    shared = shared.clamp(-127, 127)
    scale = torch.ldexp(torch.ones_like(amax), shared)
    payload = (b / scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (payload.float() * scale.unsqueeze(-1)).reshape(N, K)


_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def _nvfp4_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """NVFP4 quantize->dequantize round-trip (test-only value-domain simulation)."""
    N, K = w.shape
    if K % 16 != 0:
        raise ValueError(f"NVFP4 requires K % 16 == 0, got {K}")
    w32 = w.float()
    alpha = (w32.abs().max() / (448.0 * 6.0)).clamp(min=1e-12)
    b = w32.reshape(N, K // 16, 16)
    sf = (b.abs().amax(-1) / 6.0 / alpha).clamp(max=448.0).to(torch.float8_e4m3fn)
    eff = sf.float() * alpha
    eff = torch.where(eff > 0, eff, torch.ones_like(eff))
    v = (b / eff.unsqueeze(-1)).clamp(-6.0, 6.0)
    mids = torch.tensor(_E2M1_MIDS, device=w.device)
    lut = torch.tensor(_E2M1_LUT, device=w.device)
    mag = lut[torch.bucketize(v.abs().contiguous(), mids)]
    snapped = torch.where(v < 0, -mag, mag)
    return (snapped * eff.unsqueeze(-1)).reshape(N, K)


@flashinfer_api(trace=transform_weights_for_sm90_push_trace)
def transform_weights_for_sm90_push(
    w13: torch.Tensor,
    w2: torch.Tensor,
    weight_format: str = "bf16",
    interleave_gate_up: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert dense BF16 weights into FP8 block-scale layout."""
    if weight_format != "bf16":
        raise ValueError(
            f"unsupported weight_format {weight_format!r}; "
            "SM90 push accepts BF16 checkpoint weights only"
        )
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("w13/w2 must be BF16 checkpoint tensors")
    if w13.ndim != 3 or w2.ndim != 3 or w13.shape[0] != w2.shape[0]:
        raise ValueError("w13/w2 must be (E, 2I, H) / (E, H, I)")
    E, two_i, H = w13.shape
    _, H2, I = w2.shape
    if H2 != H or two_i != 2 * I:
        raise ValueError(
            f"inconsistent weight shapes: w13 {tuple(w13.shape)}, w2 {tuple(w2.shape)}"
        )
    if interleave_gate_up and I % 128 != 0:
        raise ValueError(f"interleave_gate_up needs I % 128 == 0, got I={I}")
    dev = w13.device
    w13_fp8 = torch.empty(E, two_i, H, device=dev, dtype=torch.float8_e4m3fn)
    w13_sf = torch.empty(E, two_i // 128, H // 128, device=dev, dtype=torch.float32)
    w2_fp8 = torch.empty(E, H, I, device=dev, dtype=torch.float8_e4m3fn)
    w2_sf = torch.empty(E, H // 128, I // 128, device=dev, dtype=torch.float32)
    for e in range(E):
        a, b = w13[e], w2[e]
        q, s = _per_block_cast_128x128(a)
        w13_fp8[e].copy_(q)
        w13_sf[e].copy_(s)
        q, s = _per_block_cast_128x128(b)
        w2_fp8[e].copy_(q)
        w2_sf[e].copy_(s)
    if interleave_gate_up:
        nb = I // 128
        w13_fp8 = (
            w13_fp8.reshape(E, 2, nb, 128, H)
            .transpose(1, 2)
            .reshape(E, two_i, H)
            .contiguous()
        )
        w13_sf = (
            w13_sf.reshape(E, 2, nb, H // 128)
            .transpose(1, 2)
            .reshape(E, two_i // 128, H // 128)
            .contiguous()
        )
    return w13_fp8, w13_sf, w2_fp8, w2_sf


@dataclass(frozen=True)
class Sm90PushWeights:
    """Transformed SM90-push weights WITH their layout tag."""

    w13_fp8: torch.Tensor
    w13_sf: torch.Tensor
    w2_fp8: torch.Tensor
    w2_sf: torch.Tensor
    w13_interleaved: bool = False

    def __post_init__(self):
        if (
            self.w13_fp8.dtype != torch.float8_e4m3fn
            or self.w2_fp8.dtype != torch.float8_e4m3fn
        ):
            raise ValueError("Sm90PushWeights payloads must be float8_e4m3fn")
        if self.w13_sf.dtype != torch.float32 or self.w2_sf.dtype != torch.float32:
            raise ValueError("Sm90PushWeights scales must be float32")
        if self.w13_fp8.ndim != 3:
            raise ValueError("w13_fp8 must be (E, 2I, H)")
        two_i = self.w13_fp8.shape[1]
        if two_i <= 0 or two_i % 256 != 0:
            raise ValueError(
                f"w13 second dim (2I) must be a positive multiple of 256, got {two_i}"
            )


def make_sm90_push_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    weight_format: str = "bf16",
    interleave_gate_up: bool = False,
) -> Sm90PushWeights:
    """:func:`transform_weights_for_sm90_push` + layout tag, in one step."""
    w13_fp8, w13_sf, w2_fp8, w2_sf = transform_weights_for_sm90_push(
        w13, w2, weight_format=weight_format, interleave_gate_up=interleave_gate_up
    )
    return Sm90PushWeights(
        w13_fp8=w13_fp8,
        w13_sf=w13_sf,
        w2_fp8=w2_fp8,
        w2_sf=w2_sf,
        w13_interleaved=interleave_gate_up,
    )


class _Sm90PushMoERunner:
    """Full SM90 push MegaMoE forward over a :class:`_Sm90PushPipe`."""

    def __init__(
        self,
        pipe: _Sm90PushPipe,
        weights: "Sm90PushWeights | torch.Tensor | None" = None,
        w13_sf: torch.Tensor | None = None,
        w2_fp8: torch.Tensor | None = None,
        w2_sf: torch.Tensor | None = None,
        *,
        w13_fp8: torch.Tensor | None = None,
        w13_interleaved: bool | None = None,
    ):
        """Full forward executor; weights via Sm90PushWeights bundle or raw tensors."""
        E, H = pipe.E, pipe.H
        self.pipe = pipe
        self._poisoned = False
        self.record_stages = False  # per-stage profiler ranges (see _record_stage)

        # weights are per-rank state: form AND content checks run guarded
        def _local_init():
            nonlocal weights, w13_sf, w2_fp8, w2_sf, w13_fp8, w13_interleaved
            if w13_fp8 is not None:  # legacy keyword form: w13_fp8= names arg 1
                if weights is not None:
                    raise ValueError(
                        "pass the weights either positionally/as a bundle OR "
                        "via the legacy w13_fp8= keyword, not both"
                    )
                weights = w13_fp8
            if weights is None:
                raise ValueError(
                    "weights are required: pass an Sm90PushWeights bundle or "
                    "the four raw tensors (w13_fp8, w13_sf, w2_fp8, w2_sf)"
                )
            if isinstance(weights, Sm90PushWeights):
                if w13_sf is not None or w2_fp8 is not None or w2_sf is not None:
                    raise ValueError(
                        "pass EITHER an Sm90PushWeights bundle OR the four raw "
                        "tensors, not both"
                    )
                if (
                    w13_interleaved is not None
                    and w13_interleaved != weights.w13_interleaved
                ):
                    raise ValueError(
                        f"explicit w13_interleaved={w13_interleaved} contradicts "
                        f"the Sm90PushWeights tag ({weights.w13_interleaved}); "
                        f"drop the kwarg -- the bundle already knows its layout"
                    )
                w13_fp8, w13_sf = weights.w13_fp8, weights.w13_sf
                w2_fp8, w2_sf = weights.w2_fp8, weights.w2_sf
                w13_interleaved = weights.w13_interleaved
            else:
                w13_fp8 = weights
                if w13_sf is None or w2_fp8 is None or w2_sf is None:
                    raise ValueError(
                        "raw-tensor form requires all four tensors "
                        "(w13_fp8, w13_sf, w2_fp8, w2_sf)"
                    )
                if w13_interleaved is None:
                    w13_interleaved = False
            if pipe.config.fuse_fc1_epilogue != w13_interleaved:
                raise ValueError(
                    f"fuse_fc1_epilogue={pipe.config.fuse_fc1_epilogue} requires "
                    f"w13_interleaved={pipe.config.fuse_fc1_epilogue} (transform "
                    f"with interleave_gate_up={pipe.config.fuse_fc1_epilogue}); "
                    f"got w13_interleaved={w13_interleaved}"
                )
            if (
                w13_fp8.dtype != torch.float8_e4m3fn
                or w2_fp8.dtype != torch.float8_e4m3fn
            ):
                raise ValueError(
                    "weights must be pre-quantized fp8 "
                    "(use transform_weights_for_sm90_push)"
                )
            if w13_fp8.ndim != 3 or w13_fp8.shape[0] != E or w13_fp8.shape[2] != H:
                raise ValueError(f"w13_fp8 must be (E={E}, 2I, H={H})")
            two_i = w13_fp8.shape[1]
            if two_i <= 0 or two_i % 256 != 0:
                raise ValueError(
                    f"w13 second dim (2I) must be a positive multiple of 256 "
                    f"(I % 128 == 0), got {two_i}"
                )
            if w13_sf.dtype != torch.float32 or w2_sf.dtype != torch.float32:
                raise ValueError(
                    f"weight scales must be float32 (the kernels reinterpret "
                    f"the buffers as float*), got w13_sf={w13_sf.dtype}, "
                    f"w2_sf={w2_sf.dtype}"
                )
            i_size = two_i // 2
            if tuple(w2_fp8.shape) != (E, H, i_size):
                raise ValueError(f"w2_fp8 must be (E, H, I) = ({E}, {H}, {i_size})")
            if tuple(w13_sf.shape) != (E, two_i // 128, H // 128) or tuple(
                w2_sf.shape
            ) != (E, H // 128, i_size // 128):
                raise ValueError("weight scale shapes do not match the fp8 weights")
            for name, t in (
                ("w13_fp8", w13_fp8),
                ("w13_sf", w13_sf),
                ("w2_fp8", w2_fp8),
                ("w2_sf", w2_sf),
            ):
                if t.device != pipe.device:
                    raise ValueError(f"{name} must be on {pipe.device}, got {t.device}")
                if not t.is_contiguous():
                    raise ValueError(f"{name} must be contiguous")
            self.w13_interleaved = w13_interleaved
            self.I = i_size
            self.w13_fp8, self.w13_sf = w13_fp8, w13_sf
            self.w2_fp8, self.w2_sf = w2_fp8, w2_sf
            self._init_gemm_resources()
            return None

        _run_guarded_phase(
            pipe._comm, getattr(pipe, "rank", 0), "weights+gemm-resources", _local_init
        )

    def _init_gemm_resources(self) -> None:
        """Local (collective-free) resource construction; see __init__."""
        from ..gemm.gemm_base import create_fp8_blockscale_gemm_runner_sm90

        pipe = self.pipe
        E, H = pipe.E, pipe.H
        two_i = 2 * self.I
        self.runner = create_fp8_blockscale_gemm_runner_sm90()

        m_cap, m_ws = pipe.m_cap, pipe.m_ws
        # TMA DECLARED-ROW BOUND: cover both the 128-row tile boundary and
        # the frozen A-descriptor declaration align4(m_ws) -- TMA clamps to
        # the DECLARATION, so a smaller allocation would be real OOB traffic.
        m_buf = max((m_cap + 127) // 128 * 128, (m_ws + 3) // 4 * 4)
        dv = pipe.device
        p_ws = max((m_ws + E * 31) // 32 * 32, 1)  # sfa stride (padded m_ws)
        self.a1 = torch.empty(m_buf, H, dtype=torch.uint8, device=dv)
        self.sfa1 = torch.empty((H // 128) * p_ws + 128, dtype=torch.float32, device=dv)
        self.meta = torch.empty(m_buf, 4, dtype=torch.int32, device=dv)
        self.row_expert = torch.empty(m_buf, dtype=torch.int32, device=dv)
        self.h = (
            None
            if pipe.config.fuse_fc1_epilogue
            else torch.empty(m_buf, two_i, dtype=torch.bfloat16, device=dv)
        )
        self.a2 = torch.empty(m_buf, self.I, dtype=torch.uint8, device=dv)
        self.sfa2 = torch.empty(
            (self.I // 128) * p_ws + 128, dtype=torch.float32, device=dv
        )
        self.y = torch.empty(m_buf, H, dtype=torch.bfloat16, device=dv)
        self._g = None  # lazy: only the unfused (fuse_act=False) path needs it

        self._workspace: torch.Tensor | None = None
        self.configure_workspace()
        self._prepare_gemm_jit()

    def configure_workspace(self) -> None:
        """(Re)apply this pipeline's workspace state to its runner."""
        pipe = self.pipe
        two_i = 2 * self.I
        sz = self.runner.get_moe_workspace_size(
            pipe.token_capacity * pipe.K,
            max(two_i, pipe.H),
            max(pipe.H, self.I),
            pipe.ep,
            pipe.E,
            True,
            True,
        )
        self._workspace = torch.empty(
            max(int(sz), 1), device=pipe.device, dtype=torch.uint8
        )
        self.runner.configure_workspace(self._workspace)

    _FC1_FUSED_FAIL_HELP = (
        "moe_gemm_fc1_fused failed (the fused DeepGEMM variant is "
        "JIT-compiled via in-process nvcc). Likely causes and fixes: "
        "(1) stale/corrupt DeepGEMM disk cache -- clear "
        "~/.tensorrt_llm/cache (or $TRTLLM_DG_CACHE_DIR) and retry; "
        "(2) nvcc not reachable (CUDA_HOME) or DeepGEMM JIT disabled "
        "(TRTLLM_DG_ENABLED=0). To run WITHOUT this experiment, rebuild "
        "with Sm90PushConfig(fuse_fc1_epilogue=False) and non-interleaved "
        "weights (interleave_gate_up=False) -- the unfused FA path is the "
        "maintained anchor."
    )

    def _prepare_gemm_jit(self) -> None:
        """Compile EVERY DeepGEMM kernel this config uses AT CONSTRUCTION."""
        pipe = self.pipe
        offsets0 = torch.zeros(pipe.E + 1, dtype=torch.int64, device=pipe.device)
        if pipe.config.fuse_fc1_epilogue:
            try:
                self.runner.moe_gemm_fc1_fused(
                    self.a2,
                    self.sfa2,
                    self.a1.view(torch.float8_e4m3fn),
                    self.w13_fp8,
                    offsets0,
                    2 * self.I,
                    pipe.H,
                    self.sfa1,
                    self.w13_sf,
                    True,
                )
            except Exception as exc:
                raise RuntimeError(
                    self._FC1_FUSED_FAIL_HELP + f" Underlying error: {exc}"
                ) from exc
        else:
            self.runner.moe_gemm(
                self.h,
                self.a1.view(torch.float8_e4m3fn),
                self.w13_fp8,
                offsets0,
                2 * self.I,
                pipe.H,
                self.sfa1,
                self.w13_sf,
                True,
            )
        self.runner.moe_gemm(
            self.y,
            self.a2.view(torch.float8_e4m3fn),
            self.w2_fp8,
            offsets0,
            pipe.H,
            self.I,
            self.sfa2,
            self.w2_sf,
            True,
        )
        torch.cuda.synchronize()

    def _g_buf(self) -> torch.Tensor:
        if self._g is None:
            # sized off a2 (same m_buf rows); h may be None in fused mode
            self._g = torch.empty(
                self.a2.shape[0], self.I, dtype=torch.bfloat16, device=self.pipe.device
            )
        return self._g

    def forward(
        self, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor
    ) -> torch.Tensor:
        """x (T, H) bf16, topk_ids (T, K) int32 (negative = masked route),
        topk_w (T, K) fp32; returns (T, H) in the pipe's out_dtype as a view
        of a persistent buffer, valid until the next forward."""
        if self._poisoned:
            raise RuntimeError(
                "sm90_push pipe is poisoned by an earlier mid-round failure; "
                "rebuild _Sm90PushPipe/_Sm90PushMoERunner"
            )
        pipe = self.pipe
        T = x.shape[0]
        if x.ndim != 2 or x.shape[1] != pipe.H:
            raise ValueError(f"x must be (T, {pipe.H})")
        if pipe.token_capacity < T:
            raise ValueError(f"T {T} exceeds token_capacity {pipe.token_capacity}")
        if topk_ids.shape != (T, pipe.K) or topk_w.shape != (T, pipe.K):
            raise ValueError(f"routing must be (T, {pipe.K})")
        # CUDA-GRAPH RULE: no implicit .to()/.contiguous() fixups -- a
        # materialized copy would freeze into the capture
        for name, t, dt in (
            ("x", x, torch.bfloat16),
            ("topk_ids", topk_ids, torch.int32),
            ("topk_w", topk_w, torch.float32),
        ):
            if t.dtype != dt:
                raise ValueError(f"{name} must be {dt}, got {t.dtype}")
            if t.device != pipe.device:
                raise ValueError(f"{name} must be on {pipe.device}, got {t.device}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        nv = self.record_stages
        try:
            with _record_stage("begin_round", nv):
                pipe.proto_begin_round()
            with _record_stage("dispatch", nv):
                pipe.proto_dispatch(x, topk_ids, topk_w)
            with _record_stage("wait_prefix", nv):
                pipe.proto_wait_prefix()
            with _record_stage("compact", nv):
                pipe.proto_compact(self.a1, self.sfa1, self.meta, self.row_expert)
            with _record_stage("fc1", nv):
                if pipe.config.fuse_fc1_epilogue:
                    try:
                        self.runner.moe_gemm_fc1_fused(
                            self.a2,
                            self.sfa2,
                            self.a1.view(torch.float8_e4m3fn),
                            self.w13_fp8,
                            pipe._offsets,
                            2 * self.I,
                            pipe.H,
                            self.sfa1,
                            self.w13_sf,
                            True,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            self._FC1_FUSED_FAIL_HELP + f" Underlying error: {exc}"
                        ) from exc
                else:
                    self.runner.moe_gemm(
                        self.h,
                        self.a1.view(torch.float8_e4m3fn),
                        self.w13_fp8,
                        pipe._offsets,
                        2 * self.I,
                        pipe.H,
                        self.sfa1,
                        self.w13_sf,
                        True,
                    )
            if not pipe.config.fuse_fc1_epilogue:
                with _record_stage("act_quant", nv):
                    if pipe.config.fuse_act:
                        pipe.proto_silu_mul_quant(
                            self.h, self.a2, self.sfa2, self.row_expert
                        )
                    else:
                        g = self._g_buf()
                        pipe.module.sm90_silu_mul_gated(
                            g, self.h, pipe._m_dev, g.shape[0]
                        )
                        pipe.module.sm90_quant_grouped(
                            self.a2,
                            self.sfa2,
                            g,
                            pipe._offsets,
                            pipe._pad_base,
                            pipe._m_dev,
                            pipe._p_dev,
                            self.row_expert,
                            g.shape[0],
                        )
            with _record_stage("fc2", nv):
                self.runner.moe_gemm(
                    self.y,
                    self.a2.view(torch.float8_e4m3fn),
                    self.w2_fp8,
                    pipe._offsets,
                    pipe.H,
                    self.I,
                    self.sfa2,
                    self.w2_sf,
                    True,
                )
            with _record_stage("combine", nv):
                pipe.proto_combine(self.y, self.meta)
            with _record_stage("wait_combine", nv):
                pipe.proto_wait_combine()
            with _record_stage("reduce", nv):
                out = pipe.proto_reduce(T)
            with _record_stage("ack", nv):
                pipe.proto_ack()
        except Exception:
            self._poisoned = True
            raise
        return out

    __call__ = forward
