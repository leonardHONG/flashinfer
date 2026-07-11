#!/usr/bin/env python3
"""Summarize SM90 operator, routing, vLLM, and SGLang quick results."""

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _median(values: Iterable[Any]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return statistics.median(finite) if finite else None


def _percentile(values: Iterable[Any], percentile: float) -> float | None:
    ordered = sorted(value for item in values if (value := _finite(item)) is not None)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _load_objects(path: Path) -> list[tuple[dict[str, Any], Path]]:
    files: list[Path]
    if path.is_dir():
        files = sorted(
            [
                *path.rglob("*.json"),
                *path.rglob("*.jsonl"),
            ]
        )
    else:
        files = [path]

    objects: list[tuple[dict[str, Any], Path]] = []
    for file in files:
        if file.name in {"summary.json"}:
            continue
        try:
            if file.suffix == ".jsonl":
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        value = json.loads(line)
                        if isinstance(value, dict):
                            objects.append((value, file))
            else:
                value = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    objects.append((value, file))
                elif isinstance(value, list):
                    objects.extend(
                        (item, file) for item in value if isinstance(item, dict)
                    )
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {file}: {exc}")
    return objects


def _operator_section(
    records: list[dict[str, Any]], route_records: list[dict[str, Any]]
) -> list[str]:
    if not records:
        return []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("mode") not in {
            "e2e",
            "e2e_nccl",
            "transport",
            "transport_nccl",
        }:
            continue
        config = json.dumps(record.get("config", {}), sort_keys=True)
        key = (
            record.get("case_id", ""),
            record.get("ep"),
            record.get("routing"),
            config,
            bool(record.get("graph")),
        )
        grouped[key].append(record)

    lines = [
        "## Operator quick A/B",
        "",
        "| EP | Routing | Tokens | Graph | Baseline p50 ms | Optimized p50 ms | "
        "Baseline/optimized | Optimized p99 ms | NCCL/optimized | "
        "Optimized cos min / growth max |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    transport_lines = [
        "## Transport quick A/B",
        "",
        "| EP | Routing | Tokens | Graph | Baseline p50 ms | Optimized p50 ms | "
        "Baseline/optimized | Optimized p99 ms | NCCL/optimized | "
        "Estimated optimized payload GB/s |",
        "|---:|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    route_index = {
        (
            record.get("ep_size"),
            record.get("routing"),
            record.get("tokens_per_rank"),
        ): record
        for record in route_records
    }
    mismatches = []
    for (_, ep, routing, config_json, graph), rows in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        config = json.loads(config_json)
        provenance = {
            (
                row.get("git_commit"),
                row.get("source_hash"),
                row.get("cuda_version"),
                row.get("gpu_name"),
            )
            for row in rows
        }
        mismatch = len(provenance) > 1
        if mismatch:
            mismatches.append(
                f"EP{ep}/{routing}/T={config.get('tokens', 'n/a')}/graph={graph}"
            )
        baseline = [
            row
            for row in rows
            if row.get("mode") == "e2e"
            and not row.get("dedup")
            and not row.get("grouped_combine")
            and not row.get("fuse_fc1")
        ]
        optimized = [
            row
            for row in rows
            if row.get("mode") == "e2e"
            and row.get("dedup")
            and row.get("grouped_combine")
            and row.get("fuse_fc1")
        ]
        nccl = [
            row
            for row in rows
            if row.get("mode") == "e2e_nccl"
            and row.get("dedup")
            and row.get("grouped_combine")
            and row.get("fuse_fc1")
        ]
        base_p50 = None if mismatch else _median(row.get("p50_ms") for row in baseline)
        opt_p50 = None if mismatch else _median(row.get("p50_ms") for row in optimized)
        opt_p99 = None if mismatch else _median(row.get("p99_ms") for row in optimized)
        nccl_p50 = None if mismatch else _median(row.get("p50_ms") for row in nccl)
        ab_speedup = base_p50 / opt_p50 if base_p50 and opt_p50 else None
        nccl_speedup = nccl_p50 / opt_p50 if nccl_p50 and opt_p50 else None
        cos_values = [
            value for row in optimized if (value := _finite(row.get("cos"))) is not None
        ]
        growth_values = [
            value
            for row in optimized
            if (value := _finite(row.get("growth"))) is not None
        ]
        correctness = (
            f"{_fmt(min(cos_values) if cos_values else None, 5)} / "
            f"{_fmt(max(growth_values) if growth_values else None, 3)}"
        )
        lines.append(
            f"| {ep} | {routing} | {config.get('tokens', 'n/a')} | "
            f"{'yes' if graph else 'no'} | {_fmt(base_p50)} | {_fmt(opt_p50)} | "
            f"{_fmt(ab_speedup)}x | {_fmt(opt_p99)} | {_fmt(nccl_speedup)}x | "
            f"{correctness} |"
        )
        transport_baseline = [
            row
            for row in rows
            if row.get("mode") == "transport"
            and not row.get("dedup")
            and not row.get("grouped_combine")
            and not row.get("fuse_fc1")
        ]
        transport_optimized = [
            row
            for row in rows
            if row.get("mode") == "transport"
            and row.get("dedup")
            and row.get("grouped_combine")
            and row.get("fuse_fc1")
        ]
        transport_nccl_rows = [
            row
            for row in rows
            if row.get("mode") == "transport_nccl"
            and row.get("dedup")
            and row.get("grouped_combine")
            and row.get("fuse_fc1")
        ]
        trans_base = (
            None
            if mismatch
            else _median(row.get("p50_ms") for row in transport_baseline)
        )
        trans_opt = (
            None
            if mismatch
            else _median(row.get("p50_ms") for row in transport_optimized)
        )
        trans_p99 = (
            None
            if mismatch
            else _median(row.get("p99_ms") for row in transport_optimized)
        )
        trans_nccl = (
            None
            if mismatch
            else _median(row.get("p50_ms") for row in transport_nccl_rows)
        )
        trans_speedup = trans_base / trans_opt if trans_base and trans_opt else None
        trans_nccl_speedup = (
            trans_nccl / trans_opt if trans_nccl and trans_opt else None
        )
        route = route_index.get((ep, routing, config.get("tokens")))
        wire_bytes = (
            _finite(route["wire_bytes"]["dedup_grouped"]) if route is not None else None
        )
        payload_gbps = (
            wire_bytes / (trans_opt * 1e6) if wire_bytes and trans_opt else None
        )
        transport_lines.append(
            f"| {ep} | {routing} | {config.get('tokens', 'n/a')} | "
            f"{'yes' if graph else 'no'} | {_fmt(trans_base)} | {_fmt(trans_opt)} | "
            f"{_fmt(trans_speedup)}x | {_fmt(trans_p99)} | "
            f"{_fmt(trans_nccl_speedup)}x | {_fmt(payload_gbps, 2)} |"
        )
    lines.append("")
    transport_lines.append("")
    if mismatches:
        lines.extend(
            [
                "> Refused to compare records with mismatched provenance: "
                + ", ".join(mismatches),
                "",
            ]
        )
    return [*lines, *transport_lines]


def _route_section(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    lines = [
        "## Synthetic routing and protocol bytes",
        "",
        "| EP | Routing | Tokens/rank | Expert max/mean | Owner-rank max/mean | "
        "Wire reduction | Window reduction |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for record in sorted(
        records,
        key=lambda item: (
            item.get("ep_size", 0),
            item.get("routing", ""),
            item.get("tokens_per_rank", 0),
        ),
    ):
        expert = record["tokens_per_expert"]
        owner = record["routes_per_owner_rank"]
        wire = record["wire_bytes"]["baseline_over_optimized"]
        window = record["symmetric_window_bytes_per_rank"]["baseline_over_optimized"]
        lines.append(
            f"| {record['ep_size']} | {record['routing']} | "
            f"{record['tokens_per_rank']} | {_fmt(expert['max_over_mean'])}x | "
            f"{_fmt(owner['max_over_mean'])}x | {_fmt(wire)}x | "
            f"{_fmt(window)}x |"
        )
    lines.append("")
    return lines


def _memory_section(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    lines = [
        "## Whole-GPU memory samples",
        "",
        "| Record | GPU | Baseline MiB | Peak MiB | Peak delta MiB | Total MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record_index, record in enumerate(records, start=1):
        for gpu_index, values in sorted(record.get("gpus", {}).items()):
            lines.append(
                f"| {record_index} | {gpu_index} | "
                f"{values.get('baseline_used_mib', 'n/a')} | "
                f"{values.get('used_mib', 'n/a')} | "
                f"{values.get('peak_delta_mib', 'n/a')} | "
                f"{values.get('total_mib', 'n/a')} |"
            )
    lines.extend(
        [
            "",
            "> These are whole-GPU nvidia-smi samples, not PyTorch allocator "
            "or DeepGEMM-workspace-only measurements.",
            "",
        ]
    )
    return lines


def _filename_labels(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    match = re.match(r"(vllm|sglang)_(.+)_(prefill|decode)$", stem)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "service", stem, "unknown"


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _percentile_field(
    record: dict[str, Any], metric: str, percentile: int
) -> float | None:
    aliases = (metric, "e2e_latency") if metric == "e2el" else (metric,)
    direct_keys = [
        key
        for alias in aliases
        for key in (
            f"p{percentile}_{alias}_ms",
            f"{alias}_p{percentile}_ms",
            f"p{percentile}_{alias}",
            f"{alias}_p{percentile}",
        )
    ]
    if percentile == 50:
        direct_keys = [
            *(
                key
                for alias in aliases
                for key in (f"median_{alias}_ms", f"median_{alias}")
            ),
            *direct_keys,
        ]
    for key in direct_keys:
        value = _finite(record.get(key))
        if value is not None:
            return value

    pairs = record.get(f"percentiles_{metric}_ms")
    if isinstance(pairs, list):
        for pair in pairs:
            if (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and _finite(pair[0]) == percentile
            ):
                return _finite(pair[1])

    raw = record.get(f"{metric}s")
    if isinstance(raw, list):
        if metric == "itl":
            raw = [
                value
                for request_values in raw
                for value in (
                    request_values
                    if isinstance(request_values, list)
                    else [request_values]
                )
            ]
        value = _percentile(raw, percentile)
        # Detailed vLLM/SGLang request arrays are stored in seconds.
        return value * 1000.0 if value is not None else None
    return None


def _derived_goodput(
    record: dict[str, Any], ttft_slo_ms: float, tpot_slo_ms: float
) -> float | None:
    existing = _finite(record.get("request_goodput"))
    if existing is not None:
        return existing
    ttfts = record.get("ttfts")
    if not isinstance(ttfts, list):
        return None
    tpots = record.get("tpots")
    if not isinstance(tpots, list):
        itls = record.get("itls")
        if isinstance(itls, list):
            tpots = [
                statistics.fmean(values) if isinstance(values, list) and values else 0.0
                for values in itls
            ]
    if not isinstance(tpots, list) or len(tpots) != len(ttfts):
        return None
    duration = _finite(
        record.get("duration")
        or record.get("duration_s")
        or record.get("benchmark_duration")
    )
    if not duration:
        return None
    good = sum(
        1
        for ttft, tpot in zip(ttfts, tpots, strict=True)
        if float(ttft) * 1000.0 < ttft_slo_ms and float(tpot) * 1000.0 < tpot_slo_ms
    )
    return good / duration


def _service_row(
    record: dict[str, Any],
    path: Path,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, Any] | None:
    if record.get("mode") or record.get("kind"):
        return None
    framework, file_variant, file_scenario = _filename_labels(path)
    metadata = _metadata(record)
    variant = str(record.get("variant") or metadata.get("variant") or file_variant)
    scenario = str(record.get("scenario") or metadata.get("scenario") or file_scenario)
    request_throughput = _finite(
        record.get("request_throughput") or record.get("request_throughput_req_s")
    )
    token_throughput = _finite(
        record.get("total_token_throughput")
        or record.get("total_throughput")
        or record.get("output_throughput")
        or record.get("output_token_throughput")
    )
    if request_throughput is None and _percentile_field(record, "ttft", 50) is None:
        return None
    return {
        "framework": framework,
        "variant": variant,
        "scenario": scenario,
        "request_throughput": request_throughput,
        "token_throughput": token_throughput,
        "goodput": _derived_goodput(record, ttft_slo_ms, tpot_slo_ms),
        **{
            f"{metric}_p{percentile}": _percentile_field(record, metric, percentile)
            for metric in ("ttft", "tpot", "itl", "e2el")
            for percentile in (50, 90, 99)
        },
    }


def _service_section(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        "## Serving quick results",
        "",
        "| Framework | Variant | Scenario | Req/s | Tok/s | Goodput req/s | "
        "TTFT p50/p90/p99 ms | TPOT p50/p90/p99 ms | E2E p50/p90/p99 ms |",
        "|---|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in sorted(
        rows, key=lambda item: (item["framework"], item["scenario"], item["variant"])
    ):
        triple = lambda metric: "/".join(  # noqa: E731
            _fmt(row[f"{metric}_p{p}"], 2) for p in (50, 90, 99)
        )
        lines.append(
            f"| {row['framework']} | {row['variant']} | {row['scenario']} | "
            f"{_fmt(row['request_throughput'], 2)} | "
            f"{_fmt(row['token_throughput'], 2)} | {_fmt(row['goodput'], 2)} | "
            f"{triple('ttft')} | {triple('tpot')} | {triple('e2el')} |"
        )
    lines.append("")

    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[(row["framework"], row["scenario"])].append(row)
    comparisons = []
    for (framework, scenario), case_rows in sorted(by_case.items()):
        baseline = next(
            (
                row
                for row in case_rows
                if "baseline" in row["variant"].lower()
                or "nccl" in row["variant"].lower()
            ),
            None,
        )
        optimized = next(
            (row for row in case_rows if "sm90" in row["variant"].lower()), None
        )
        if baseline is None or optimized is None:
            continue
        comparisons.append((framework, scenario, baseline, optimized))
    if comparisons:
        lines.extend(
            [
                "## Serving baseline versus SM90 push",
                "",
                "| Framework | Scenario | TTFT p50 speedup | TPOT p50 speedup | "
                "E2E p50 speedup | Req/s uplift | Goodput uplift |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for framework, scenario, baseline, optimized in comparisons:
            latency_speedup = lambda metric: (  # noqa: E731
                baseline[metric] / optimized[metric]
                if baseline[metric] and optimized[metric]
                else None
            )
            throughput_uplift = (
                optimized["request_throughput"] / baseline["request_throughput"]
                if baseline["request_throughput"] and optimized["request_throughput"]
                else None
            )
            goodput_uplift = (
                optimized["goodput"] / baseline["goodput"]
                if baseline["goodput"] and optimized["goodput"]
                else None
            )
            lines.append(
                f"| {framework} | {scenario} | "
                f"{_fmt(latency_speedup('ttft_p50'))}x | "
                f"{_fmt(latency_speedup('tpot_p50'))}x | "
                f"{_fmt(latency_speedup('e2el_p50'))}x | "
                f"{_fmt(throughput_uplift)}x | {_fmt(goodput_uplift)}x |"
            )
        lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="result file or directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=15.0)
    args = parser.parse_args()

    objects = _load_objects(args.path)
    operator = [obj for obj, _ in objects if obj.get("mode")]
    routes = [obj for obj, _ in objects if obj.get("kind") == "sm90_push_route_balance"]
    memory = [obj for obj, _ in objects if obj.get("kind") == "sm90_push_gpu_memory"]
    service = [
        row
        for obj, path in objects
        if (row := _service_row(obj, path, args.ttft_slo_ms, args.tpot_slo_ms))
        is not None
    ]

    lines = [
        "# SM90 Push MegaMoE Quick Benchmark Summary",
        "",
        f"Goodput SLO: TTFT < {args.ttft_slo_ms:g} ms and "
        f"TPOT < {args.tpot_slo_ms:g} ms.",
        "",
        *_operator_section(operator, routes),
        *_route_section(routes),
        *_memory_section(memory),
        *_service_section(service),
    ]
    if not operator and not routes and not memory and not service:
        lines.extend(["No recognized result records were found.", ""])
    text = "\n".join(lines)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
