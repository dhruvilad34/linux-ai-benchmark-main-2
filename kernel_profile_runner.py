#!/usr/bin/env python3
"""
Kernel Profiling & Scaling Analysis runner.

This utility wraps the project benchmark runner with perf, strace, bpftrace,
and continuous resource sampling so every execution captures the artefacts
needed for the Week 1 deliverables:

- perf: context switches, CPU migrations, task clock, cycles, instructions
- strace: per-syscall latency / counts table
- bpftrace: sched_switch trace counts (per pid/command)
- system sampler: CPU/GPU/memory utilisation over time

Usage example:
    python /home/aruldha/project/kernel_profile_runner.py \
        --benchmarks humaneval mint gaia swebench \
        --agent-counts 500 1000 1500 2000

All outputs are placed under logs/kernel_profile/<timestamp>/<benchmark>/agents_<N>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import psutil
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _which(cmd: str) -> Optional[str]:
    """Return the absolute path to an executable or None."""
    return shutil.which(cmd)


def _ensure_parent(path: Path) -> None:
    """Create parent directory for the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_config(path: Path) -> Dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _save_config(data: Dict, path: Path) -> None:
    _ensure_parent(path)
    with open(path, "w") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _format_command(cmd: Sequence[str]) -> str:
    """Return a shell-safe representation of a command list."""
    return shlex.join(map(str, cmd))


def _utcnow() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ---------------------------------------------------------------------------
# Metrics sampling
# ---------------------------------------------------------------------------


class MetricSampler(threading.Thread):
    """Background sampler for CPU/memory/GPU metrics."""

    def __init__(self, path: Path, interval: float = 1.0):
        super().__init__(daemon=True)
        self.path = path
        self.interval = interval
        self._stop_event = threading.Event()
        self._gpu_available = _which("nvidia-smi") is not None
        _ensure_parent(self.path)

        # Prime psutil CPU percent to avoid first-call zero.
        psutil.cpu_percent(interval=None)
        psutil.cpu_times_percent(interval=None)

    def stop(self) -> None:
        self._stop_event.set()

    def _sample_gpu(self) -> List[Dict]:
        if not self._gpu_available:
            return []
        try:
            query = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(query, text=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            return []

        gpus: List[Dict] = []
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpu_idx = int(parts[0])
                util = float(parts[1])
                mem_util = float(parts[2])
                mem_used = float(parts[3])
                mem_total = float(parts[4])
            except ValueError:
                continue
            gpus.append(
                {
                    "gpu_index": gpu_idx,
                    "util_percent": util,
                    "mem_util_percent": mem_util,
                    "mem_used_mb": mem_used,
                    "mem_total_mb": mem_total,
                }
            )
        return gpus

    def run(self) -> None:
        with open(self.path, "w") as fh:
            while not self._stop_event.is_set():
                cpu_times = psutil.cpu_times_percent(interval=None)
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_stats = psutil.cpu_stats()
                mem = psutil.virtual_memory()
                sample = {
                    "timestamp": _utcnow(),
                    "cpu_percent": cpu_percent,
                    "cpu_user_percent": getattr(cpu_times, "user", 0.0),
                    "cpu_system_percent": getattr(cpu_times, "system", 0.0),
                    "cpu_iowait_percent": getattr(cpu_times, "iowait", 0.0),
                    "ctx_switches_total": getattr(cpu_stats, "ctx_switches", None),
                    "ctx_switches_voluntary": getattr(cpu_stats, "voluntary_ctx_switches", None)
                    if hasattr(cpu_stats, "voluntary_ctx_switches")
                    else None,
                    "ctx_switches_involuntary": getattr(cpu_stats, "involuntary_ctx_switches", None)
                    if hasattr(cpu_stats, "involuntary_ctx_switches")
                    else None,
                    "mem_used_mb": mem.used / (1024 * 1024),
                    "mem_available_mb": mem.available / (1024 * 1024),
                    "mem_percent": mem.percent,
                    "swap_used_mb": psutil.swap_memory().used / (1024 * 1024),
                    "gpu": self._sample_gpu(),
                }
                fh.write(json.dumps(sample) + "\n")
                fh.flush()
                # Sleep in smaller increments to react faster on stop.
                for _ in range(int(self.interval * 10) or 1):
                    if self._stop_event.wait(self.interval / (int(self.interval * 10) or 1)):
                        return


def summarize_metrics(path: Path) -> Dict:
    if not path.exists():
        return {}

    count = 0
    cpu_sum = 0.0
    cpu_max = 0.0
    mem_sum = 0.0
    mem_max = 0.0
    gpu_sum = 0.0
    gpu_max = 0.0
    gpu_count = 0

    with open(path, "r") as fh:
        for line in fh:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            cpu = float(sample.get("cpu_percent") or 0.0)
            mem = float(sample.get("mem_percent") or 0.0)

            cpu_sum += cpu
            mem_sum += mem
            cpu_max = max(cpu_max, cpu)
            mem_max = max(mem_max, mem)

            gpus = sample.get("gpu") or []
            if gpus:
                avg_gpu = sum(g.get("util_percent", 0.0) for g in gpus) / len(gpus)
                gpu_sum += avg_gpu
                gpu_max = max(gpu_max, max(g.get("util_percent", 0.0) for g in gpus))
                gpu_count += 1

    if count == 0:
        return {}

    summary = {
        "samples": count,
        "cpu_percent_avg": cpu_sum / count,
        "cpu_percent_max": cpu_max,
        "mem_percent_avg": mem_sum / count,
        "mem_percent_max": mem_max,
    }
    if gpu_count:
        summary.update(
            {
                "gpu_util_percent_avg": gpu_sum / gpu_count,
                "gpu_util_percent_max": gpu_max,
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Profiling orchestration
# ---------------------------------------------------------------------------


def parse_perf_output(path: Path) -> Dict[str, Dict[str, str]]:
    """Parse perf stat output into a dictionary."""
    if not path.exists():
        return {}

    metrics: Dict[str, Dict[str, str]] = {}
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("Performance counter"):
                continue
            # Skip warnings
            if stripped.startswith("Error") or stripped.startswith("Failed"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            value_token = parts[0]
            event = parts[1]
            # Normalise value by removing commas
            value = value_token.replace(",", "")
            annotation = " ".join(parts[2:]) if len(parts) > 2 else ""
            metrics[event] = {"value": value, "annotation": annotation}
    return metrics


def parse_strace_summary(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with open(path, "r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("% time") or stripped.startswith("------"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            # strace -c output can have optional errors column.
            if len(parts) == 5:
                percent, seconds, usecs_per_call, calls, syscall = parts
                errors = "0"
            else:
                percent, seconds, usecs_per_call, calls, errors, syscall = parts[:6]
            rows.append(
                {
                    "syscall": syscall,
                    "percent_time": percent.rstrip("%"),
                    "seconds": seconds,
                    "usecs_per_call": usecs_per_call,
                    "calls": calls,
                    "errors": errors,
                }
            )
    return rows


def build_run_command(
    python_bin: str,
    main_path: Path,
    config_path: Path,
    benchmark: str,
    extra_args: Sequence[str],
) -> List[str]:
    cmd: List[str] = [
        python_bin,
        str(main_path),
        "--config",
        str(config_path),
        "--benchmark",
        benchmark,
    ]
    cmd.extend(extra_args)
    return cmd


def maybe_wrap_with_strace(cmd: List[str], output_path: Path, enabled: bool) -> Tuple[List[str], bool]:
    if not enabled:
        return cmd, False
    strace_bin = _which("strace")
    if not strace_bin:
        print("⚠️  strace not found; skipping syscall tracing.", file=sys.stderr)
        return cmd, False
    wrapped = [strace_bin, "-f", "-c", "-o", str(output_path)] + cmd
    return wrapped, True


def maybe_wrap_with_bpftrace(cmd: List[str], output_path: Path, enabled: bool) -> Tuple[List[str], bool]:
    if not enabled:
        return cmd, False
    if os.geteuid() != 0:
        print("⚠️  bpftrace requires root privileges; skipping.", file=sys.stderr)
        return cmd, False
    bpftrace_bin = _which("bpftrace")
    if not bpftrace_bin:
        print("⚠️  bpftrace not found; skipping sched_switch tracing.", file=sys.stderr)
        return cmd, False
    program = textwrap.dedent(
        """
        tracepoint:sched:sched_switch
        {
            @[tid, curtask->comm] = count();
        }

        END
        {
            printf("Context Switch Counts (tid,comm) -> count\\n");
            print(@);
        }
        """
    ).strip()
    _ensure_parent(output_path)
    command_string = _format_command(cmd)
    wrapped = [
        bpftrace_bin,
        "-o",
        str(output_path),
        "-e",
        program,
        "-c",
        command_string,
    ]
    return wrapped, True


def maybe_wrap_with_perf(cmd: List[str], output_path: Path, enabled: bool) -> Tuple[List[str], bool]:
    if not enabled:
        return cmd, False
    perf_bin = _which("perf")
    if not perf_bin:
        print("⚠️  perf not found; skipping perf stat.", file=sys.stderr)
        return cmd, False
    events = "context-switches,cpu-migrations,task-clock,cycles,instructions"
    wrapped = [
        perf_bin,
        "stat",
        "-o",
        str(output_path),
        "-e",
        events,
        "--",
    ] + cmd
    return wrapped, True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_profile(
    python_bin: str,
    main_path: Path,
    base_config: Path,
    benchmark: str,
    agent_count: int,
    output_dir: Path,
    extra_args: Sequence[str],
    enable_perf: bool,
    enable_strace: bool,
    enable_bpftrace: bool,
    sampler_interval: float,
) -> Dict:
    run_dir = output_dir / benchmark / f"agents_{agent_count}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Prepare run-specific config
    cfg = _load_config(base_config)
    cfg["num_agents"] = agent_count
    cfg["benchmark"] = benchmark
    cfg["output_dir"] = str(run_dir)
    config_path = run_dir / "config.yaml"
    _save_config(cfg, config_path)

    metrics_log_path = run_dir / "system_metrics.jsonl"
    sampler = MetricSampler(metrics_log_path, interval=sampler_interval)

    base_cmd = build_run_command(python_bin, main_path, config_path, benchmark, extra_args)
    stdout_path = run_dir / "project_stdout.log"
    stderr_path = run_dir / "project_stderr.log"
    strace_path = run_dir / "strace_summary.txt"
    bpftrace_path = run_dir / "bpftrace.txt"
    perf_path = run_dir / "perf_stat.txt"

    wrapped_cmd, strace_enabled = maybe_wrap_with_strace(base_cmd, strace_path, enable_strace)
    wrapped_cmd, bpf_enabled = maybe_wrap_with_bpftrace(wrapped_cmd, bpftrace_path, enable_bpftrace)
    wrapped_cmd, perf_enabled = maybe_wrap_with_perf(wrapped_cmd, perf_path, enable_perf)

    command_repr = _format_command(wrapped_cmd)

    start_time = _utcnow()
    sampler.start()

    with open(stdout_path, "w") as stdout_fh, open(stderr_path, "w") as stderr_fh:
        proc = subprocess.run(
            wrapped_cmd,
            cwd=str(main_path.parent),
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
        )

    sampler.stop()
    sampler.join(timeout=5)
    end_time = _utcnow()

    perf_metrics = parse_perf_output(perf_path) if perf_enabled else {}
    strace_metrics = parse_strace_summary(strace_path) if strace_enabled else []
    metric_summary = summarize_metrics(metrics_log_path)

    run_summary = {
        "benchmark": benchmark,
        "num_agents": agent_count,
        "config_path": str(config_path),
        "command": command_repr,
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "return_code": proc.returncode,
        "tools": {
            "perf": perf_enabled,
            "strace": strace_enabled,
            "bpftrace": bpf_enabled,
        },
        "outputs": {
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "perf_stat": str(perf_path) if perf_enabled else None,
            "strace_summary": str(strace_path) if strace_enabled else None,
            "bpftrace_output": str(bpftrace_path) if bpf_enabled else None,
            "system_metrics": str(metrics_log_path),
        },
        "perf_metrics": perf_metrics,
        "strace_metrics": strace_metrics,
        "system_metrics_summary": metric_summary,
    }

    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(run_summary, fh, indent=2)

    return run_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run project benchmarks under perf/strace/bpftrace with system monitoring."
    )
    parser.add_argument(
        "--config",
        default="/home/aruldha/project/config/config.yaml",
        help="Base config file to clone per run.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mint", "gaia", "swebench"],
        help="Benchmarks to execute (passed via --benchmark).",
    )
    parser.add_argument(
        "--agent-counts",
        nargs="+",
        type=int,
        default=[500, 1000, 1500, 2000],
        help="Agent counts to iterate over.",
    )
    parser.add_argument(
        "--output-root",
        default="/home/aruldha/project/logs/kernel_profile",
        help="Root directory for profiler outputs.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to run main.py with.",
    )
    parser.add_argument(
        "--sampler-interval",
        type=float,
        default=1.0,
        help="Seconds between system metric samples.",
    )
    parser.add_argument(
        "--disable-perf",
        action="store_true",
        help="Disable perf stat wrapping.",
    )
    parser.add_argument(
        "--disable-strace",
        action="store_true",
        help="Disable strace syscall summary.",
    )
    parser.add_argument(
        "--disable-bpftrace",
        action="store_true",
        help="Disable bpftrace sched_switch tracing.",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to main.py (use --extra-args -- <args>).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    base_config = Path(args.config).resolve()
    if not base_config.exists():
        print(f"❌ Base config not found: {base_config}", file=sys.stderr)
        return 1

    main_path = Path("/home/aruldha/project/main.py").resolve()
    if not main_path.exists():
        print(f"❌ main.py not found at {main_path}", file=sys.stderr)
        return 1

    output_root = Path(args.output_root).resolve()
    timestamp_root = output_root / dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    timestamp_root.mkdir(parents=True, exist_ok=True)

    extra_args: List[str] = []
    if args.extra_args:
        if args.extra_args[0] == "--":
            extra_args = [str(x) for x in args.extra_args[1:]]
        else:
            extra_args = [str(x) for x in args.extra_args]

    print("🚀 Kernel Profiling & Scaling Analysis")
    print(f"   Base config: {base_config}")
    print(f"   Output root: {timestamp_root}")
    print(f"   Benchmarks: {', '.join(args.benchmarks)}")
    print(f"   Agent counts: {', '.join(str(a) for a in args.agent_counts)}")
    if extra_args:
        print(f"   Extra args -> main.py: {extra_args}")

    runs: List[Dict] = []
    for benchmark in args.benchmarks:
        for agent_count in args.agent_counts:
            print(f"\n▶️  Running benchmark={benchmark} agents={agent_count}")
            run_summary = run_profile(
                python_bin=args.python_bin,
                main_path=main_path,
                base_config=base_config,
                benchmark=benchmark,
                agent_count=agent_count,
                output_dir=timestamp_root,
                extra_args=extra_args,
                enable_perf=not args.disable_perf,
                enable_strace=not args.disable_strace,
                enable_bpftrace=not args.disable_bpftrace,
                sampler_interval=args.sampler_interval,
            )
            runs.append(run_summary)
            status = "✅" if run_summary["return_code"] == 0 else "⚠️"
            print(f"{status} Completed benchmark={benchmark} agents={agent_count}")
            if run_summary["system_metrics_summary"]:
                sm = run_summary["system_metrics_summary"]
                print(
                    f"   CPU avg {sm.get('cpu_percent_avg', 0):.1f}% "
                    f"(max {sm.get('cpu_percent_max', 0):.1f}%), "
                    f"Mem avg {sm.get('mem_percent_avg', 0):.1f}% "
                    f"(max {sm.get('mem_percent_max', 0):.1f}%)"
                )

    index_path = timestamp_root / "summary_index.json"
    with open(index_path, "w") as fh:
        json.dump({"runs": runs, "created_at": _utcnow()}, fh, indent=2)

    print(f"\n📦 Summary index saved to {index_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


