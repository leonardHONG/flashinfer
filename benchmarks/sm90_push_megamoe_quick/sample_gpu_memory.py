#!/usr/bin/env python3
"""Sample whole-GPU memory while a command runs or for a fixed duration."""

# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sample() -> dict[str, dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values = {}
    for line in result.stdout.splitlines():
        index, uuid, used, total = (part.strip() for part in line.split(","))
        values[index] = {
            "uuid": uuid,
            "used_mib": int(used),
            "total_mib": int(total),
        }
    if not values:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return values


def _update_peak(
    peak: dict[str, dict[str, int]], sample: dict[str, dict[str, int]]
) -> None:
    for index, current in sample.items():
        previous = peak.setdefault(index, dict(current))
        if current["used_mib"] > previous["used_mib"]:
            previous["used_mib"] = current["used_mib"]


def monitor(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command and args.duration_s is None:
        raise ValueError("provide --duration-s or a command after --")
    if command and args.duration_s is not None:
        raise ValueError("--duration-s and a command are mutually exclusive")

    baseline = _sample()
    peak = {index: dict(values) for index, values in baseline.items()}
    process = subprocess.Popen(command) if command else None
    started = time.perf_counter()
    samples = 0
    deadline = started + args.duration_s if args.duration_s is not None else None

    while True:
        now = time.perf_counter()
        process_done = process is not None and process.poll() is not None
        duration_done = deadline is not None and now >= deadline
        if process_done or duration_done:
            break
        try:
            _update_peak(peak, _sample())
            samples += 1
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            print(f"warning: nvidia-smi sample failed: {exc}", file=sys.stderr)
        time.sleep(args.interval_ms / 1000.0)

    _update_peak(peak, _sample())
    samples += 1
    duration = time.perf_counter() - started
    return_code = process.wait() if process is not None else 0
    result = {
        "kind": "sm90_push_gpu_memory",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration,
        "interval_ms": args.interval_ms,
        "sample_count": samples,
        "command": command,
        "command_return_code": return_code,
        "gpus": {
            index: {
                **peak_values,
                "baseline_used_mib": baseline[index]["used_mib"],
                "peak_delta_mib": max(
                    peak_values["used_mib"] - baseline[index]["used_mib"], 0
                ),
            }
            for index, peak_values in peak.items()
        },
        "note": (
            "Whole-GPU nvidia-smi samples, not allocator-only memory. Run "
            "separately from publication timing because polling can perturb "
            "the host."
        ),
    }
    return result, return_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.duration_s is not None and args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.interval_ms < 20:
        parser.error("--interval-ms must be at least 20")

    try:
        result, return_code = monitor(args)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
