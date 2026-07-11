"""
SM90 push-based MegaMoE whole-layer EP backend.

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

Unlike the split comm backends (nccl_ep / nixl_ep), which expose
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Tuple

if TYPE_CHECKING:
    import torch

    from ...fused_moe.api import MoEConfig, MoEWeightPack
    from ..config import BootstrapConfig, FleetParams
    from ..tensors import MoEEpTensors


@dataclass
class Sm90PushEpConfig:
    """Whole-layer SM90 push backend selector for MoEEpLayer; defaults enable
    dedup dispatch, grouped combine and the fused FC1 epilogue. Constraints
    and lifecycle: docs/sm90_push_megamoe.md."""

    backend_name: str = "sm90_push"
    capacity_factor: float = 1.0
    dedup_dispatch: bool = True
    grouped_combine: bool = True
    fuse_fc1_epilogue: bool = True
    # Wire dtypes; "bf16" variants are debug anchors, "fp8" is production.
    payload_dtype: str = "fp8"
    combine_dtype: str = "fp8"
    ep_group: Optional[Any] = None
    #: CUDA device index for this rank. None = torch.cuda.current_device().
    device_index: Optional[int] = None
    zero_copy_output: bool = False
    allow_unverified_p2p: bool = False
    init_timeout_s: float = 600.0

    #: MoEWeightPack native-view key this backend consumes.
    WEIGHT_VIEW_KEY: ClassVar[str] = "sm90_push_fp8_block"

    @staticmethod
    def prepare_weights(
        w13_bf16: "torch.Tensor",
        w2_bf16: "torch.Tensor",
        *,
        fuse_fc1_epilogue: bool = True,
        device: Optional["torch.device"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """Build the sm90_push_fp8_block weight view from dense BF16 checkpoints
        (the only public format; register via MoEWeightPack.prepare_for)."""
        from ...fused_moe.prepare import prepare_sm90_push_fp8_block_weights

        return prepare_sm90_push_fp8_block_weights(
            w13_bf16,
            w2_bf16,
            interleave_gate_up=fuse_fc1_epilogue,
            device=device,
        )


class _Sm90PushEpBackend:
    """Private whole-layer executor behind ``Sm90PushEpConfig``."""

    def __init__(
        self,
        config: Sm90PushEpConfig,
        bootstrap: "BootstrapConfig",
        fleet_params: "FleetParams",
        compute_config: "MoEConfig",
        weight_pack: "MoEWeightPack",
    ) -> None:
        import torch

        from ...fused_moe.api import MoEConfig, MoEWeightPack, QuantVariant
        from ...fused_moe.sm90_push_a2a import (
            _Sm90PushMoERunner,
            _Sm90PushPipe,
            Sm90PushCombine,
            Sm90PushConfig,
            Sm90PushPayload,
            _default_comm_backend,
            _run_phase0_handshake,
            _run_guarded_phase,
        )
        from ...tllm_enums import ActivationType

        bootstrap_world, bootstrap_rank = bootstrap.world_size, bootstrap.rank
        if config.ep_group is not None:
            from ...comm.mnnvl import TorchDistBackend

            comm = TorchDistBackend(group=config.ep_group)
        else:
            comm = _default_comm_backend(bootstrap_world)
        try:
            timeout_value = float(config.init_timeout_s)
            timeout_valid = math.isfinite(timeout_value) and timeout_value > 0
        except (TypeError, ValueError):
            timeout_value = 0.0
            timeout_valid = False
        timeout_error = None
        if timeout_valid and hasattr(comm, "set_timeout"):
            try:
                comm.set_timeout(timeout_value)
            except Exception as exc:
                timeout_error = f"{type(exc).__name__}: {exc}"
        world, rank = comm.Get_size(), comm.Get_rank()
        routing = getattr(compute_config, "routing", None)
        experts = getattr(compute_config, "experts", None)
        backend_fingerprint = (
            type(compute_config).__name__,
            type(weight_pack).__name__,
            getattr(fleet_params, "num_experts", None),
            getattr(fleet_params, "max_tokens_per_rank", None),
            getattr(fleet_params, "token_hidden_size", None),
            getattr(routing, "num_experts", None),
            getattr(routing, "top_k", None),
            getattr(experts, "intermediate_size", None),
            config.capacity_factor,
            config.dedup_dispatch,
            config.grouped_combine,
            config.fuse_fc1_epilogue,
            config.payload_dtype,
            config.combine_dtype,
            config.zero_copy_output,
            config.allow_unverified_p2p,
            repr(config.init_timeout_s),
        )

        def _validate_phase0():
            if not timeout_valid:
                raise ValueError(
                    "init_timeout_s must be finite and positive, got "
                    f"{config.init_timeout_s}"
                )
            if timeout_error is not None:
                raise RuntimeError(f"failed to apply init_timeout_s: {timeout_error}")

        _run_phase0_handshake(
            comm,
            bootstrap_world,
            bootstrap_rank,
            backend_fingerprint,
            _validate_phase0,
        )

        def _validate_config():
            if not isinstance(compute_config, MoEConfig):
                raise ValueError(
                    "sm90_push requires compute_config to be a "
                    "flashinfer.fused_moe.MoEConfig, got "
                    f"{type(compute_config).__name__}"
                )
            if not isinstance(weight_pack, MoEWeightPack):
                raise ValueError(
                    "sm90_push requires weights to be a "
                    "flashinfer.fused_moe.MoEWeightPack, got "
                    f"{type(weight_pack).__name__}"
                )
            if compute_config.quant.variant is not QuantVariant.DeepSeekFp8:
                raise ValueError(
                    "sm90_push computes natively in FP8 block-scale: "
                    "compute_config.quant.variant must be QuantVariant.DeepSeekFp8 "
                    f"(got {compute_config.quant.variant!r}). Dense BF16 "
                    "checkpoints are converted to FP8 block-scale at LOAD time "
                    "via Sm90PushEpConfig.prepare_weights; MXFP8/NVFP4 have no "
                    "native SM90 compute and no packed loader here."
                )
            if compute_config.activation.activation_type is not ActivationType.Swiglu:
                raise ValueError(
                    "sm90_push implements the SwiGLU expert FFN only; got "
                    f"activation {compute_config.activation.activation_type!r}"
                )
            if world > 32:
                raise ValueError(
                    f"sm90_push supports a single-node NVLink group of at most "
                    f"32 ranks, got world_size={world} -- select a different "
                    "backend explicitly (no silent fallback)"
                )
            num_experts = compute_config.routing.num_experts
            if fleet_params.num_experts != num_experts:
                raise ValueError(
                    f"fleet_params.num_experts ({fleet_params.num_experts}) != "
                    f"compute_config.routing.num_experts ({num_experts})"
                )
            if num_experts % world != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by "
                    f"world_size ({world}) for contiguous expert partitioning"
                )
            e_loc = num_experts // world
            declared_local = compute_config.experts.local_num_experts
            if declared_local is not None and declared_local != e_loc:
                raise ValueError(
                    f"compute_config.experts.local_num_experts "
                    f"({declared_local}) != num_experts // world_size ({e_loc})"
                )
            expected_offset = rank * e_loc
            if compute_config.experts.local_expert_offset not in (0, expected_offset):
                raise ValueError(
                    "sm90_push uses contiguous expert partitioning: "
                    f"local_expert_offset must be 0 (default) or rank*e_local "
                    f"({expected_offset}), got "
                    f"{compute_config.experts.local_expert_offset}"
                )
            if config.payload_dtype not in (
                "fp8",
                "bf16",
            ) or config.combine_dtype not in ("fp8", "bf16"):
                raise ValueError(
                    f"payload_dtype/combine_dtype must be 'fp8' or 'bf16', got "
                    f"{config.payload_dtype!r}/{config.combine_dtype!r}"
                )
            if bootstrap.stream != 0:
                raise ValueError(
                    "sm90_push does not (yet) honor an explicit "
                    "bootstrap.stream; it launches every kernel on the CURRENT "
                    "torch stream. Run the layer inside `with "
                    "torch.cuda.stream(...)` instead of passing a raw stream "
                    "handle (no silent fallback)."
                )
            return None

        _run_guarded_phase(comm, rank, "config", _validate_config)
        e_local = compute_config.routing.num_experts // world
        hidden_size = fleet_params.token_hidden_size
        token_capacity = fleet_params.max_tokens_per_rank
        top_k = compute_config.routing.top_k
        device_index = (
            config.device_index
            if config.device_index is not None
            else (torch.cuda.current_device() if torch.cuda.is_available() else 0)
        )
        i_size = compute_config.experts.intermediate_size

        def _validate_view():
            view = weight_pack.get_view(Sm90PushEpConfig.WEIGHT_VIEW_KEY)
            for key in ("w13_fp8", "w13_sf", "w2_fp8", "w2_sf", "w13_interleaved"):
                if key not in view:
                    raise ValueError(
                        f"weight view {Sm90PushEpConfig.WEIGHT_VIEW_KEY!r} is "
                        f"missing {key!r}; build it with "
                        "Sm90PushEpConfig.prepare_weights"
                    )
            tag = bool(view["w13_interleaved"].item())
            if tag != config.fuse_fc1_epilogue:
                raise ValueError(
                    f"weight view was prepared with interleave_gate_up={tag} "
                    f"but the config runs fuse_fc1_epilogue="
                    f"{config.fuse_fc1_epilogue}; re-prepare the weights (the "
                    "layouts are not interchangeable -- no silent fallback)"
                )
            if view["w13_fp8"].shape != (e_local, 2 * i_size, hidden_size):
                raise ValueError(
                    f"w13_fp8 shape {tuple(view['w13_fp8'].shape)} != expected "
                    f"(e_local={e_local}, 2I={2 * i_size}, H={hidden_size}) -- "
                    "the view must hold THIS rank's expert slice"
                )
            return None

        _run_guarded_phase(comm, rank, "weight-view", _validate_view)
        view = weight_pack.get_view(Sm90PushEpConfig.WEIGHT_VIEW_KEY)
        w13_interleaved = bool(view["w13_interleaved"].item())

        pcfg = Sm90PushConfig(
            payload_dtype=Sm90PushPayload(config.payload_dtype),
            combine_dtype=Sm90PushCombine(config.combine_dtype),
            fuse_act=True,
            capacity_factor=config.capacity_factor,
            dedup_dispatch=config.dedup_dispatch,
            grouped_combine=config.grouped_combine,
            fuse_fc1_epilogue=config.fuse_fc1_epilogue,
        )
        self._pipe = _Sm90PushPipe(
            ep_size=world,
            rank=rank,
            num_local_experts=e_local,
            hidden_size=hidden_size,
            top_k=top_k,
            token_capacity=token_capacity,
            device_index=device_index,
            config=pcfg,
            comm_backend=comm,
            out_dtype=torch.bfloat16,
            allow_unverified_p2p=config.allow_unverified_p2p,
        )
        self._mm = _Sm90PushMoERunner(
            self._pipe,
            w13_fp8=view["w13_fp8"],
            w13_sf=view["w13_sf"],
            w2_fp8=view["w2_fp8"],
            w2_sf=view["w2_sf"],
            w13_interleaved=w13_interleaved,
        )
        self._token_capacity = token_capacity
        self._hidden_size = hidden_size
        self._top_k = top_k
        self._zero_copy = config.zero_copy_output
        self._in_flight = False
        self._round_event: "torch.cuda.Event | None" = None
        self._round_stream: "torch.cuda.Stream | None" = None
        # plain bool: a CPU tag buffer would break a no-op layer.to("cuda")
        self.w13_interleaved = w13_interleaved
        self._weight_view = {
            "w13_fp8": view["w13_fp8"],
            "w13_sf": view["w13_sf"],
            "w2_fp8": view["w2_fp8"],
            "w2_sf": view["w2_sf"],
        }

    def _stage_mega_moe_inputs(
        self,
        hidden_states: "torch.Tensor",
        topk_ids: "torch.Tensor",
        topk_weights: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Strictly validate the BF16 public inputs; inputs must be FINITE
        (checked only under FLASHINFER_VALIDATE_INPUTS=1 -- host sync)."""
        import os

        import torch

        dev = self._pipe.device
        if os.environ.get("FLASHINFER_VALIDATE_INPUTS", "0") not in ("", "0"):
            if not bool(torch.isfinite(hidden_states.float()).all()) or not bool(
                torch.isfinite(topk_weights).all()
            ):
                raise ValueError(
                    "sm90_push inputs must be finite (FLASHINFER_VALIDATE_INPUTS)"
                )
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "sm90_push takes native BF16 hidden_states (quantized 1x128 "
                f"to FP8 inside dispatch); got {hidden_states.dtype}"
            )
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self._hidden_size:
            raise ValueError(
                f"hidden_states must be (T, {self._hidden_size}), got "
                f"{tuple(hidden_states.shape)}"
            )
        T = hidden_states.shape[0]
        if self._token_capacity < T:
            raise ValueError(
                f"T={T} exceeds fleet_params.max_tokens_per_rank={self._token_capacity}"
            )
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise ValueError(
                "sm90_push takes topk_ids as int32 and topk_weights as "
                f"float32 (got {topk_ids.dtype} / {topk_weights.dtype}); "
                "convert once at the call site -- the layer performs no "
                "implicit casts (CUDA-graph safety)"
            )
        if topk_ids.shape != (T, self._top_k) or topk_weights.shape != (
            T,
            self._top_k,
        ):
            raise ValueError(
                f"routing tensors must be (T={T}, top_k={self._top_k}); got "
                f"{tuple(topk_ids.shape)} / {tuple(topk_weights.shape)}"
            )
        for name, t in (
            ("hidden_states", hidden_states),
            ("topk_ids", topk_ids),
            ("topk_weights", topk_weights),
        ):
            if t.device != dev:
                raise ValueError(f"{name} must be on {dev}, got {t.device}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        return hidden_states, topk_ids, topk_weights

    def weight_buffers(self) -> Dict[str, "torch.Tensor"]:
        """The prepared weight-view tensors (for nn.Module buffer registration)."""
        return dict(self._weight_view)

    def __call__(self, t: "MoEEpTensors") -> "torch.Tensor":
        """Whole-layer forward: BF16 in, BF16 out (shape of hidden_states)."""
        import torch

        if self._mm is None:
            raise RuntimeError("sm90_push layer was destroyed; rebuild it")
        if self._in_flight:
            raise RuntimeError(
                "sm90_push forward re-entered while a round is being "
                "submitted; the layer supports ONE in-flight forward "
                "(single-stream contract) -- serialize callers"
            )
        self._in_flight = True
        try:
            cur = torch.cuda.current_stream(self._pipe.device)
            # event query is illegal during stream capture and a record
            # would become a graph node, so the cross-stream guard applies
            # to eager calls only
            capturing = torch.cuda.is_current_stream_capturing()
            if (
                not capturing
                and self._round_event is not None
                and self._round_stream != cur
                and not self._round_event.query()
            ):
                raise RuntimeError(
                    "sm90_push forward submitted on a different stream while "
                    "the previous round is still executing; the layer runs "
                    "ONE round at a time on ONE stream -- wait for the "
                    "previous round's completion (or stay on its stream)"
                )
            x, ids, w = self._stage_mega_moe_inputs(
                t.hidden_states, t.topk_ids, t.topk_weights
            )
            out = self._mm(x, ids, w)
            # clone BEFORE recording: the guard event must be the round's
            # last work (zero-copy callers consume the view on this stream)
            result = out if self._zero_copy else out.clone()
            if not capturing:
                if self._round_event is None:
                    self._round_event = torch.cuda.Event()
                self._round_event.record(cur)
                self._round_stream = cur
            return result
        finally:
            self._in_flight = False

    def destroy(self) -> None:
        """Quiesce, agree with peers, then release references. Idempotent."""
        if self._pipe is None:
            return
        import torch

        torch.cuda.synchronize(self._pipe.device)
        self._pipe._comm.barrier()
        self._mm = None
        self._pipe = None
