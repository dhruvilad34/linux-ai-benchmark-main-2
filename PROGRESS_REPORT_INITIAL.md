# AI Concurrency Kernel Optimization – Initial Progress Snapshot

_Covers day‑1 deliverables prior to the runtime optimizations._

---

## 1. Objectives
- Automate kernel-tuning for low-latency AI concurrency workloads.
- Provide a before/after benchmark harness.
- Integrate the tuning flow into the existing `agent_concurrency.py` entry point.

## 2. Deliverables (Day 1)

| File | Description |
|------|-------------|
| `optimize_kernel.sh` | Applies safe kernel scheduler and memory settings (`sched_min_granularity`, `sched_wakeup_granularity`, `sched_migration_cost`, disables autogroup, lowers `vm.swappiness`, allocates huge pages, sets NVMe scheduler to mq-deadline). Script is idempotent, heavily commented, and warns when sudo/root privileges are required. |
| `kernel_benchmark.py` | Runs the workload twice—before and after optimization—capturing latency, throughput, CPU/GPU utilisation, context-switch metrics via `pidstat`, and optional scheduler latency via `runqlat` (if `bcc-tools` installed). Outputs JSON files plus a comparison table. |
| `agent_concurrency.py` | Wrapper around `main.py` that adds a `--optimize` flag to invoke the tuning script automatically before launching the workload. Falls back gracefully when sudo is not available. |

## 3. Documentation
- `KERNEL_OPTIMIZATION_USAGE.md` – shows how to run the tuning scripts, benchmark, and interpret the output. Includes troubleshooting tips for missing tools (`pidstat`, `runqlat`) and notes on root vs non-root environments.
- `WHEN_TO_RUN_WHAT.md` – quick decision tree for running the benchmark vs. normal workloads or manual tuning. Provides concrete commands for each scenario.

## 4. Benchmark Output (Baseline)
The initial benchmark (before non-root runtime tweaks) established a consistent reporting format, even though no significant improvements were yet observed:
```
Metric             Before     After     Δ Improvement
----------------------------------------------------
avg_latency_s      ~6 s       ~6 s      (± noise)
throughput_rps     ~0.16      ~0.15     (± noise)
context_switches   ≈ 241K/s   ≈ 244K/s  (system-wide, noisy)
```
- Context switches were dominated by other processes because the host is a shared 256-core server.
- GPU idle time remained high; the workload used only one HumanEval task at a time.
- These findings motivated the later runtime-level optimizations captured in `PROGRESS_REPORT.md`.

## 5. Next Steps Identified
- Add user-level optimizations for non-root environments (implemented later via `optimize_kernel_no_sudo.sh`).
- Improve runtime behaviour without kernel tuning (CPU affinity, batching, perf logging).
- Capture cleaner context-switch metrics (per-process `pidstat -w -p`).
- Scale workload and adjust batching to push GPU utilisation.

This file serves as the “before improvements” snapshot. For the full evolution—non-root fallbacks, runtime tweaks, latest benchmarks—see `PROGRESS_REPORT.md`.





