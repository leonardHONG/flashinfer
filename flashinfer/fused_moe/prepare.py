"""
Weight-preparation helpers for the unified MoE API.

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

Backends consume different native weight layouts (quantization + swizzle +
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


def prepare_sm90_push_fp8_block_weights(
    w13_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    interleave_gate_up: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build the ``sm90_push_fp8_block`` weight view (SM90 push MegaMoE EP)."""
    from .sm90_push_a2a import transform_weights_for_sm90_push

    if device is None:
        device = w13_bf16.device
    w13_bf16 = w13_bf16.to(device)
    w2_bf16 = w2_bf16.to(device)
    w13_fp8, w13_sf, w2_fp8, w2_sf = transform_weights_for_sm90_push(
        w13_bf16,
        w2_bf16,
        weight_format="bf16",
        interleave_gate_up=interleave_gate_up,
    )
    return {
        "w13_fp8": w13_fp8,
        "w13_sf": w13_sf,
        "w2_fp8": w2_fp8,
        "w2_sf": w2_sf,
        "w13_interleaved": torch.tensor(interleave_gate_up, dtype=torch.bool),
    }
