"""MoEEpLayer — public nn.Module for MoE Expert-Parallel. Split backends
run transport-only dispatch/combine (identity inner compute); the sm90_push
whole-layer backend runs the full expert FFN in one fused pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
import torch.nn as nn

from .algo_knobs import (
    AlgoKnob,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
)
from .config import (
    BootstrapConfig,
    CombineInputParams,
    DispatchInputParams,
    FleetParams,
    HandleParams,
)

from .fleet import Fleet, create_fleet

if TYPE_CHECKING:
    from ..fused_moe.api import MoEConfig, MoEWeightPack
    from .split_backends.sm90_push import _Sm90PushEpBackend
    from .tensors import MoEEpTensors


class MoEEpLayer(nn.Module):
    """Backend-agnostic Expert-Parallel layer.

    Split backends (``NcclEpConfig`` / ``NvepConfig``) run transport-only
    dispatch/combine over a Fleet; ``Sm90PushEpConfig`` selects the
    whole-layer SM90 push backend (requires ``compute_config`` + ``weights``,
    constructs eagerly and collectively on every EP rank). Whole-layer
    lifecycle contract: docs/sm90_push_megamoe.md.
    """

    def __init__(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
        fleet_knobs: Sequence[AlgoKnob] = (),
        backend: Union[str, object] = "nccl_ep",
        compute_config: Optional["MoEConfig"] = None,
        weights: Optional["MoEWeightPack"] = None,
    ) -> None:
        super().__init__()
        self._bootstrap = bootstrap
        self._fleet_params = fleet_params
        self._fleet_knobs = list(fleet_knobs)
        self._backend = backend
        self._fleet: Fleet | None = None
        self._compute_config = compute_config
        self._weights = weights
        self._whole_layer: "_Sm90PushEpBackend | None" = None

        is_whole_layer = getattr(backend, "backend_name", None) == "sm90_push"
        if is_whole_layer:
            if compute_config is None or weights is None:
                raise ValueError(
                    "backend=Sm90PushEpConfig() is a whole-layer backend and "
                    "requires BOTH compute_config (MoEConfig) and weights "
                    "(MoEWeightPack with the 'sm90_push_fp8_block' view)."
                )
        elif compute_config is not None or weights is not None:
            # split backends would silently ignore them -- reject instead
            raise ValueError(
                "compute_config/weights are only consumed by the sm90_push "
                f"whole-layer backend; backend={backend!r} would ignore them"
            )
        self._is_whole_layer = is_whole_layer

        if is_whole_layer:
            # eager collective construction: no lazy compile may remain
            wl = self._ensure_whole_layer()
            for name, tensor in wl.weight_buffers().items():
                self.register_buffer(f"sm90_push_{name}", tensor, persistent=True)
            self.register_buffer(
                "sm90_push_w13_interleaved",
                torch.tensor(
                    wl.w13_interleaved,
                    dtype=torch.bool,
                    device=wl._pipe.device,
                ),
                persistent=True,
            )

    def _ensure_fleet(self) -> Fleet:
        if self._fleet is None:
            self._fleet = create_fleet(
                self._bootstrap,
                self._fleet_params,
                self._fleet_knobs,
                backend=self._backend,
            )
        return self._fleet

    def _ensure_whole_layer(self) -> "_Sm90PushEpBackend":
        if self._whole_layer is None:
            from .split_backends.sm90_push import _Sm90PushEpBackend, Sm90PushEpConfig

            backend = self._backend
            if not isinstance(backend, Sm90PushEpConfig):
                raise TypeError(
                    "whole-layer sm90_push backends must be selected with an "
                    f"Sm90PushEpConfig instance, got {type(backend).__name__}"
                )
            self._whole_layer = _Sm90PushEpBackend(
                backend,
                self._bootstrap,
                self._fleet_params,
                self._compute_config,
                self._weights,
            )
        return self._whole_layer

    def _assert_sm90_buffers_bound(self, what: str) -> None:
        """Hard-fail if a registered sm90 buffer stopped aliasing the tensor
        the running backend reads (conversion or assign-style load)."""
        if self._whole_layer is None:
            return
        for name, held in self._whole_layer.weight_buffers().items():
            buf = getattr(self, f"sm90_push_{name}", None)
            if buf is None or buf.data_ptr() != held.data_ptr():
                raise RuntimeError(
                    f"MoEEpLayer(sm90_push): buffer sm90_push_{name} was "
                    f"rebound by {what} and no longer aliases the tensor the "
                    "running pipeline reads (device/dtype conversion or "
                    "assign-style load). The layer state is torn and must be "
                    "destroyed and reconstructed; weight updates must copy_ "
                    "into the existing buffers."
                )

    def _apply(self, fn, *args, **kwargs):
        """Reject device/dtype conversions: the symm window is device-pinned
        and the FP8/scale buffers must keep their dtypes."""
        pointers = {}
        if self._whole_layer is not None:
            pointers = {
                name: tensor.data_ptr()
                for name, tensor in self._buffers.items()
                if name.startswith("sm90_push_") and tensor is not None
            }
            signatures = {
                (tensor.dtype, tensor.device)
                for name, tensor in self._buffers.items()
                if name.startswith("sm90_push_") and tensor is not None
            }
            for dtype, device in signatures:
                probe = torch.empty(1, dtype=dtype, device=device)
                moved = fn(probe)
                if (
                    moved.device != probe.device
                    or moved.dtype != probe.dtype
                    or moved.data_ptr() != probe.data_ptr()
                ):
                    raise RuntimeError(
                        "MoEEpLayer(sm90_push) cannot be moved, cast, or "
                        "rebound after construction. Construct the layer on "
                        "the target device instead."
                    )
        ret = super()._apply(fn, *args, **kwargs)
        for name, pointer in pointers.items():
            if self._buffers[name].data_ptr() != pointer:
                raise RuntimeError(
                    "MoEEpLayer(sm90_push) buffer storage changed during "
                    "_apply; the layer state is torn and must be rebuilt"
                )
        self._assert_sm90_buffers_bound("_apply")
        return ret

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self._whole_layer is not None:
            sm90_prefix = prefix + "sm90_push_"
            has_sm90_state = any(k.startswith(sm90_prefix) for k in state_dict)
            if local_metadata.get("assign_to_params_buffers", False) and has_sm90_state:
                raise RuntimeError(
                    "MoEEpLayer(sm90_push) does not support "
                    "load_state_dict(assign=True) for its sm90_push_* buffers; "
                    "use the default copy semantics"
                )
            if has_sm90_state:
                tag_key = prefix + "sm90_push_w13_interleaved"
                if tag_key not in state_dict:
                    raise RuntimeError(
                        "MoEEpLayer(sm90_push): checkpoint carries sm90_push "
                        "weights without sm90_push_w13_interleaved"
                    )
                incoming_tag = state_dict[tag_key]
                if (
                    not isinstance(incoming_tag, torch.Tensor)
                    or incoming_tag.dtype != torch.bool
                    or incoming_tag.numel() != 1
                ):
                    raise RuntimeError(
                        "MoEEpLayer(sm90_push): "
                        "sm90_push_w13_interleaved must be a bool scalar tensor"
                    )
                expected_tag = bool(self.sm90_push_w13_interleaved.item())
                loaded_tag = bool(incoming_tag.item())
                if loaded_tag != expected_tag:
                    raise RuntimeError(
                        "MoEEpLayer(sm90_push): checkpoint weights use "
                        f"interleave_gate_up={loaded_tag} but this layer was "
                        f"constructed with interleave_gate_up={expected_tag}"
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._assert_sm90_buffers_bound("load_state_dict")

    @staticmethod
    def _inner_compute_identity(
        expert_tensors: "torch.Tensor", num_tokens: int
    ) -> "torch.Tensor":
        """Stub inner compute — passes dispatched tokens through unchanged.

        Real quant-aware routing (cutlass_fused_moe / trtllm_* / cute_dsl)
        lands in a follow-up that takes (expert_tensors, num_tokens,
        EpConfig.quant) and dispatches to the right kernel.
        """
        return expert_tensors

    def forward(self, t: "MoEEpTensors") -> "torch.Tensor":
        if self._is_whole_layer:
            # one fused pipeline owns the forward; no Fleet/Handle exists
            if self._whole_layer is None:
                raise RuntimeError(
                    "MoEEpLayer(sm90_push) was destroyed; construct a new layer"
                )
            return self._whole_layer(t)
        fleet = self._ensure_fleet()
        handle_knobs: list[AlgoKnob] = [
            HandleAlgoKnobUserStream(stream=torch.cuda.current_stream().cuda_stream),
            HandleAlgoKnobTopKWeights(weights=t.topk_weights),
        ]
        handle = fleet.create_handle(
            HandleParams(topk_ids=t.topk_ids),
            algo_knobs=handle_knobs,
        )
        d = handle.dispatch(DispatchInputParams(x=[t.hidden_states]))
        expert_out = self._inner_compute_identity(d.expert_tensors, d.num_tokens)
        c = handle.combine(
            CombineInputParams(
                x=[expert_out],
                out=torch.empty_like(t.hidden_states),
            )
        )
        handle.complete()
        return c.x

    def destroy(self) -> None:
        """Tear down backend resources; idempotent, collective for the
        whole-layer backend (call on every rank)."""
        if self._fleet is not None:
            self._fleet.destroy()
            self._fleet = None
        if self._whole_layer is not None:
            self._whole_layer.destroy()
            self._whole_layer = None
