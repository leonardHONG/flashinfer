# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for the SM90 push-based MegaMoE (Hopper NVLink EP)."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _transform_weights_for_sm90_push_reference(
    w13, w2, weight_format="bf16", interleave_gate_up=False
):
    """Reference for ``transform_weights_for_sm90_push`` (bf16 path)."""
    if weight_format != "bf16":
        raise NotImplementedError(
            f"reference models weight_format='bf16' only, got {weight_format!r}"
        )

    def cast_128x128(w):
        N, K = w.shape
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

    def cast_all(ws):
        qs, ss = zip(*(cast_128x128(w) for w in ws), strict=True)
        return torch.stack(qs), torch.stack(ss)

    w13_q, w13_s = cast_all(w13)
    w2_q, w2_s = cast_all(w2)
    if interleave_gate_up:
        E, two_i, H = w13_q.shape
        nb = two_i // 256
        w13_q = (
            w13_q.reshape(E, 2, nb, 128, H)
            .transpose(1, 2)
            .reshape(E, two_i, H)
            .contiguous()
        )
        w13_s = (
            w13_s.reshape(E, 2, nb, H // 128)
            .transpose(1, 2)
            .reshape(E, two_i // 128, H // 128)
            .contiguous()
        )
    return w13_q, w13_s, w2_q, w2_s


def _transform_weights_for_sm90_push_init(
    *,
    num_local_experts: int = 2,
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    two_intermediate_size: int = 0,  # derived
    two_i_div_128: int = 0,  # derived
    h_div_128: int = 0,  # derived
    i_div_128: int = 0,  # derived
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``transform_weights_for_sm90_push``."""
    del two_intermediate_size, two_i_div_128, h_div_128, i_div_128  # derived
    torch.manual_seed(seed)
    w13 = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    w2 = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    return {"w13": w13, "w2": w2, "weight_format": "bf16"}


transform_weights_for_sm90_push_trace = TraceTemplate(
    op_type="quantization",
    name_prefix="transform_weights_for_sm90_push",
    description=(
        "Offline weight prep for the SM90 push-based MegaMoE: fold this "
        "rank's gate/up (w13) and down (w2) expert weights to 128x128 "
        "fp8-blockscale (e4m3 payload + f32 per-block scales) for "
        "DeepGEMM's grouped GEMM. 'bf16' is the only public format; "
        "weight_format='mxfp8'/'nvfp4' are experimental test-only "
        "SIMULATIONS (dense-BF16 round-trip through the format's quant "
        "grid -- no packed-checkpoint loader, no native MX/FP4 compute "
        "on Hopper)."
    ),
    axes={
        "num_local_experts": Const(abbrev="e"),
        "hidden_size": Const(abbrev="h"),
        "intermediate_size": Const(abbrev="i"),
        "two_intermediate_size": Var(description="2 * intermediate_size (gate+up)."),
        "two_i_div_128": Var(description="2 * intermediate_size // 128."),
        "h_div_128": Var(description="hidden_size // 128."),
        "i_div_128": Var(description="intermediate_size // 128."),
    },
    inputs={
        "w13": Tensor(
            ["num_local_experts", "two_intermediate_size", "hidden_size"],
            description="Gate/up weights (gate = w13[:, :I, :], up = w13[:, I:, :]).",
        ),
        "w2": Tensor(
            ["num_local_experts", "hidden_size", "intermediate_size"],
            description="Down weights.",
        ),
        "weight_format": Scalar(
            "string",
            optional=True,
            description=(
                "'bf16' (default; the only public format) or the "
                "experimental test-only simulations 'mxfp8'/'nvfp4'."
            ),
        ),
        "interleave_gate_up": Scalar(
            "bool",
            optional=True,
            description=(
                "Interleave w13's N dimension in 128-row blocks "
                "([g0, u0, g1, u1, ...]) for the FC1-fused-epilogue kernel; "
                "scales are re-ordered identically. Default False."
            ),
        ),
    },
    outputs={
        "w13_fp8": Tensor(
            ["num_local_experts", "two_intermediate_size", "hidden_size"],
            dtype="float8_e4m3fn",
        ),
        "w13_sf": Tensor(
            ["num_local_experts", "two_i_div_128", "h_div_128"],
            dtype="float32",
            description="Per-128x128-block scales for w13_fp8.",
        ),
        "w2_fp8": Tensor(
            ["num_local_experts", "hidden_size", "intermediate_size"],
            dtype="float8_e4m3fn",
        ),
        "w2_sf": Tensor(
            ["num_local_experts", "h_div_128", "i_div_128"],
            dtype="float32",
            description="Per-128x128-block scales for w2_fp8.",
        ),
    },
    tags=["status:verified", "quantization:fp8"],
    reference=_transform_weights_for_sm90_push_reference,
    init=_transform_weights_for_sm90_push_init,
)
