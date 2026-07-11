#!/usr/bin/env python3
"""Synthetic SM90 push routing, wire-volume, and window-size estimator.

This is intentionally CPU-only.  It complements the GPU benchmark by making
the routing imbalance and the byte reductions from dedup dispatch / grouped
combine explicit without requiring a model server.
"""

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable


CONFIGS = {
    "TINY": dict(tokens=256, hidden_size=1024, intermediate=1024, experts=8, top_k=2),
    "SMALL": dict(
        tokens=2048, hidden_size=4096, intermediate=2048, experts=32, top_k=6
    ),
    "DSV3": dict(
        tokens=2048, hidden_size=7168, intermediate=2048, experts=256, top_k=8
    ),
}


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _distribution(values: list[int]) -> dict[str, float]:
    mean = statistics.fmean(values) if values else 0.0
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "max": float(max(values, default=0)),
        "max_over_mean": float(max(values, default=0) / mean) if mean else 0.0,
        "cv": float(std / mean) if mean else 0.0,
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p99": _percentile(values, 99),
    }


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _window_bytes(
    *,
    ep_size: int,
    experts: int,
    tokens: int,
    top_k: int,
    hidden_size: int,
    payload_dtype: str,
    combine_dtype: str,
    dedup: bool,
    grouped: bool,
    capacity_factor: float,
) -> int:
    """Mirror the PushPipe protocol regions; excludes local GEMM workspace."""
    local_experts = experts // ep_size
    max_routes = ep_size * tokens * top_k
    meta_rows = max(int(capacity_factor * max_routes), 1)
    pool_rows = max(int(capacity_factor * ep_size * tokens), 1) if dedup else meta_rows
    row_bytes = hidden_size if payload_dtype == "fp8" else 2 * hidden_size
    scale_bytes = hidden_size // 128 * 4 if payload_dtype == "fp8" else 0
    combine_slots = ep_size if grouped else top_k

    offset = 0
    offset = _align(offset + pool_rows * row_bytes)
    offset = _align(offset + pool_rows * scale_bytes)
    offset = _align(offset + meta_rows * 16)
    offset = _align(offset + 8)  # packed pool head
    offset = _align(offset + ep_size * 8)  # base cells
    offset = _align(offset + local_experts * ep_size * 8)  # count cells
    offset = _align(offset + ep_size * 8)  # combine-done cells
    offset = _align(offset + ep_size * 8)  # ack cells
    if combine_dtype == "bf16":
        offset = _align(offset + tokens * top_k * hidden_size * 2)
        offset = _align(offset)
        offset = _align(offset)
    else:
        offset = _align(offset)
        offset = _align(offset + tokens * combine_slots * hidden_size)
        offset = _align(offset + tokens * combine_slots * (hidden_size // 128) * 4)
    return offset


def _routes_for_token(
    rng: random.Random,
    *,
    routing: str,
    source_rank: int,
    experts: int,
    local_experts: int,
    top_k: int,
) -> list[int]:
    if routing == "hot":
        return [0] * top_k
    if routing == "hot1":
        tail_pool = list(range(1, experts))
        return [0, *rng.sample(tail_pool, top_k - 1)]
    if routing == "all_remote":
        remote = [
            expert
            for expert in range(experts)
            if expert // local_experts != source_rank
        ]
        return rng.sample(remote, top_k)
    return rng.sample(range(experts), top_k)


def analyze(args: argparse.Namespace) -> dict:
    cfg = dict(CONFIGS[args.config])
    if args.tokens is not None:
        cfg["tokens"] = args.tokens
    tokens = cfg["tokens"]
    hidden = cfg["hidden_size"]
    experts = cfg["experts"]
    top_k = cfg["top_k"]
    ep_size = args.ep_size
    if experts % ep_size:
        raise ValueError(f"experts={experts} must be divisible by ep_size={ep_size}")
    if top_k > experts:
        raise ValueError(f"top_k={top_k} exceeds experts={experts}")
    if args.routing == "all_remote" and ep_size == 1:
        raise ValueError("all_remote routing requires ep_size > 1")
    local_experts = experts // ep_size

    rng = random.Random(args.seed)
    expert_counts = [0] * experts
    owner_counts = [0] * ep_size
    remote_routes = 0
    remote_owner_pairs = 0
    distinct_owners = []

    for source_rank in range(ep_size):
        for _ in range(tokens):
            routes = _routes_for_token(
                rng,
                routing=args.routing,
                source_rank=source_rank,
                experts=experts,
                local_experts=local_experts,
                top_k=top_k,
            )
            owners = {expert // local_experts for expert in routes}
            distinct_owners.append(len(owners))
            remote_owner_pairs += sum(owner != source_rank for owner in owners)
            for expert in routes:
                owner = expert // local_experts
                expert_counts[expert] += 1
                owner_counts[owner] += 1
                remote_routes += owner != source_rank

    payload_row = hidden if args.payload_dtype == "fp8" else 2 * hidden
    payload_scale = hidden // 128 * 4 if args.payload_dtype == "fp8" else 0
    combine_row = hidden if args.combine_dtype == "fp8" else 2 * hidden
    combine_scale = hidden // 128 * 4 if args.combine_dtype == "fp8" else 0
    meta_row = 16

    baseline_dispatch = remote_routes * (payload_row + payload_scale + meta_row)
    optimized_dispatch = (
        remote_owner_pairs * (payload_row + payload_scale) + remote_routes * meta_row
    )
    baseline_combine = remote_routes * (combine_row + combine_scale)
    optimized_combine = remote_owner_pairs * (combine_row + combine_scale)
    baseline_wire = baseline_dispatch + baseline_combine
    optimized_wire = optimized_dispatch + optimized_combine

    common_window = dict(
        ep_size=ep_size,
        experts=experts,
        tokens=tokens,
        top_k=top_k,
        hidden_size=hidden,
        payload_dtype=args.payload_dtype,
        combine_dtype=args.combine_dtype,
        capacity_factor=args.capacity_factor,
    )
    baseline_window = _window_bytes(**common_window, dedup=False, grouped=False)
    optimized_window = _window_bytes(
        **common_window,
        dedup=True,
        grouped=args.combine_dtype == "fp8",
    )

    return {
        "kind": "sm90_push_route_balance",
        "config": args.config,
        "routing": args.routing,
        "seed": args.seed,
        "ep_size": ep_size,
        "tokens_per_rank": tokens,
        "hidden_size": hidden,
        "intermediate_size": cfg["intermediate"],
        "experts": experts,
        "top_k": top_k,
        "payload_dtype": args.payload_dtype,
        "combine_dtype": args.combine_dtype,
        "total_routes": ep_size * tokens * top_k,
        "remote_routes": remote_routes,
        "remote_owner_token_pairs": remote_owner_pairs,
        "distinct_owners_per_token": _distribution(distinct_owners),
        "tokens_per_expert": _distribution(expert_counts),
        "routes_per_owner_rank": _distribution(owner_counts),
        "wire_bytes": {
            "baseline": baseline_wire,
            "dedup_grouped": optimized_wire,
            "baseline_over_optimized": (
                baseline_wire / optimized_wire if optimized_wire else 1.0
            ),
            "note": "Remote payload/meta estimates; protocol control cells are excluded.",
        },
        "symmetric_window_bytes_per_rank": {
            "baseline": baseline_window,
            "dedup_grouped": optimized_window,
            "baseline_over_optimized": (
                baseline_window / optimized_window if optimized_window else 1.0
            ),
            "note": "Protocol window only; local GEMM workspace/weights are excluded.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, default="SMALL")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument(
        "--routing", choices=("random", "hot", "hot1", "all_remote"), default="random"
    )
    parser.add_argument("--payload-dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--combine-dtype", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tokens is not None and args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.ep_size < 1:
        parser.error("--ep-size must be positive")
    if not (0.0 < args.capacity_factor <= 1.0):
        parser.error("--capacity-factor must be in (0, 1]")

    result = analyze(args)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
