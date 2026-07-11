"""
Unified MoE API — configuration dataclasses and tensor groupings.

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

Config objects are frozen (immutable). Use ``dataclasses.replace`` to derive
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional

from torch import Tensor

from ..tllm_enums import ActivationType, RoutingMethodType


class QuantVariant(Enum):
    """Quantization scheme name; backend support is each backend's contract."""

    Bf16 = 0
    Fp8PerTensor = 1
    DeepSeekFp8 = 2
    MxFp8 = 3
    Nvfp4 = 4
    MxFp4 = 5
    MxInt4 = 6

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


@dataclass(frozen=True)
class RoutingConfig:
    """Expert routing parameters."""

    num_experts: int
    top_k: int
    method: RoutingMethodType = RoutingMethodType.Default
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: Optional[float] = None

    def __repr__(self) -> str:
        parts = [f"num_experts={self.num_experts!r}", f"top_k={self.top_k!r}"]
        if self.method != RoutingMethodType.Default:
            parts.append(f"method={self.method!r}")
        if self.n_group is not None:
            parts.append(f"n_group={self.n_group!r}")
        if self.topk_group is not None:
            parts.append(f"topk_group={self.topk_group!r}")
        if self.routed_scaling_factor is not None:
            parts.append(f"routed_scaling_factor={self.routed_scaling_factor!r}")
        return f"RoutingConfig({', '.join(parts)})"


@dataclass(frozen=True)
class QuantConfig:
    """Quantization scheme."""

    variant: QuantVariant = QuantVariant.Bf16
    swizzled_scale_factors: Optional[bool] = None
    per_token_scale: Optional[bool] = None


@dataclass(frozen=True)
class ActivationConfig:
    """Fused activation between GEMM1 and GEMM2."""

    # Convenience singletons — populated after class definition
    swiglu: ClassVar[ActivationConfig]
    geglu: ClassVar[ActivationConfig]
    relu2: ClassVar[ActivationConfig]
    identity: ClassVar[ActivationConfig]

    activation_type: ActivationType = ActivationType.Swiglu

    def __repr__(self) -> str:
        return f"ActivationConfig(activation_type={self.activation_type!r})"

    @property
    def is_gated(self) -> bool:
        return self.activation_type.is_gated


ActivationConfig.swiglu = ActivationConfig(ActivationType.Swiglu)
ActivationConfig.geglu = ActivationConfig(ActivationType.Geglu)
ActivationConfig.relu2 = ActivationConfig(ActivationType.Relu2)
ActivationConfig.identity = ActivationConfig(ActivationType.Identity)


@dataclass(frozen=True)
class ExpertConfig:
    """Expert geometry."""

    intermediate_size: int
    local_expert_offset: int = 0
    local_num_experts: Optional[int] = None

    def __repr__(self) -> str:
        parts = [f"intermediate_size={self.intermediate_size!r}"]
        if self.local_expert_offset != 0:
            parts.append(f"local_expert_offset={self.local_expert_offset!r}")
        if self.local_num_experts is not None:
            parts.append(f"local_num_experts={self.local_num_experts!r}")
        return f"ExpertConfig({', '.join(parts)})"


@dataclass(frozen=True)
class MoEConfig:
    """Top-level MoE *compute* configuration (what to compute, not where)."""

    routing: RoutingConfig
    quant: QuantConfig
    experts: ExpertConfig
    activation: ActivationConfig = field(
        default_factory=lambda: ActivationConfig(ActivationType.Swiglu)
    )

    # --- Dict-unpacking protocol: enables ``**config`` at call sites ---

    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class MoEWeightPack:
    """Long-lived weight container with per-backend native materializations."""

    native_views: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)

    def prepare_for(self, backend_key: str, view: Dict[str, Tensor]) -> None:
        """Register a backend-native weight view (caller owns the layout prep)."""
        self.native_views[backend_key] = view

    def get_view(self, backend_key: str) -> Dict[str, Tensor]:
        if backend_key not in self.native_views:
            raise KeyError(
                f"Weights not prepared for backend {backend_key!r}. "
                f"Available: {list(self.native_views)}"
            )
        return self.native_views[backend_key]
