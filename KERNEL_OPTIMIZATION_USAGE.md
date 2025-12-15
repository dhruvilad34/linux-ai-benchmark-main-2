# Kernel Optimization Usage Guide

This guide explains how to use the kernel-level performance optimization tools.

## Quick Start

### Option 1: Run with Kernel Optimization (Recommended)

Apply kernel tuning and run your AI concurrency workload in one command:

```bash
cd /home/aruldha/project
python agent_concurrency.py --optimize
```

This will:
1. Apply kernel optimization (requires sudo password)
2. Run the main AI concurrency workload
3. Use optimized kernel parameters throughout execution

### Option 2: Automated Benchmark (Before/After Comparison)

Run a complete benchmark that compares performance before and after kernel tuning:

```bash
cd /home/aruldha/project
python kernel_benchmark.py
```

This will:
1. Run `agent_concurrency.py` WITHOUT optimization (baseline)
2. Apply kernel optimization
3. Run `agent_concurrency.py` WITH optimization
4. Generate a comparison table showing improvements
5. Save results to `kernel_benchmark_results/`

**Output files:**
- `kernel_benchmark_results/comparison.txt` - Formatted table (copy to report)
- `kernel_benchmark_results/combined_results.json` - Full metrics in JSON
- `kernel_benchmark_results/results_before.json` - Before metrics
- `kernel_benchmark_results/results_after.json` - After metrics

### Option 3: Manual Kernel Optimization

Apply kernel tuning manually (without running the workload):

```bash
cd /home/aruldha/project
sudo ./optimize_kernel.sh
```

**Note:** These changes are temporary and reset on reboot. To make permanent, add to `/etc/sysctl.conf`.

## Detailed Usage

### Running with Custom Arguments

You can pass any arguments that `main.py` accepts:

```bash
# Run with optimization and custom config
python agent_concurrency.py --optimize --config config/config.yaml --num_tasks 10

# Run benchmark with custom arguments
python kernel_benchmark.py --num_tasks 5
```

### Understanding the Output

The benchmark comparison table shows:

```
Metric             Before     After     Δ Improvement
-----------------------------------------------------
avg_latency_s      8.03 s     5.67 s     +29.4%
throughput_rps     0.124      0.198      +59.6%
context_switches   12100 /s   4800 /s    -60.3%
cpu_util           45.2 %     52.1 %     +15.3%
gpu_idle_time      28 %       9 %        -67.8%
sched_latency      12.5 μs    8.2 μs     -34.4%
```

**Improvement calculation:**
- For metrics where **lower is better** (latency, context switches, GPU idle): Negative % = improvement
- For metrics where **higher is better** (throughput, CPU util): Positive % = improvement

### Requirements

**System tools:**
- `pidstat` - For context switch monitoring (install: `sudo apt-get install sysstat`)
- `sudo` access - Required for kernel parameter changes
- Optional: `runqlat` (BCC tools) - For kernel scheduling latency (install: `sudo apt-get install bpfcc-tools`)

**Python dependencies:**
- All dependencies from `requirements.txt` (already installed in your venv)

### Troubleshooting

**"pidstat not found"**
```bash
sudo apt-get install sysstat
```

**"runqlat not found" (optional)**
```bash
sudo apt-get install bpfcc-tools
```

**"Permission denied" for optimize_kernel.sh**
```bash
chmod +x optimize_kernel.sh
```

**Kernel parameters not applying:**
- Ensure you're running with `sudo`
- Some parameters may not be available on all systems (script handles this gracefully)
- Check `/proc/sys/` to verify current values

**Benchmark fails:**
- Ensure `agent_concurrency.py` runs successfully without optimization first
- Check that logs directory is writable
- Verify all Python dependencies are installed

## Example Workflow

1. **First, test without optimization:**
   ```bash
   python agent_concurrency.py --config config/config.yaml --num_tasks 1
   ```

2. **Then run the full benchmark:**
   ```bash
   python kernel_benchmark.py
   ```

3. **Review the results:**
   ```bash
   cat kernel_benchmark_results/comparison.txt
   ```

4. **Copy the table to your report:**
   ```bash
   cat kernel_benchmark_results/comparison.txt
   # Copy the output and paste into your report
   ```

## Making Kernel Changes Permanent

To make kernel optimizations persist across reboots:

1. Add to `/etc/sysctl.conf`:
   ```bash
   sudo nano /etc/sysctl.conf
   ```
   
   Add these lines:
   ```
   kernel.sched_min_granularity_ns=1000000
   kernel.sched_wakeup_granularity_ns=2000000
   kernel.sched_migration_cost_ns=5000000
   kernel.sched_autogroup_enabled=0
   vm.swappiness=10
   vm.nr_hugepages=4096
   ```

2. Apply immediately:
   ```bash
   sudo sysctl -p
   ```

3. For I/O scheduler and THP, create a systemd service or add to `/etc/rc.local`.

## Notes

- Kernel optimizations are **temporary** by default (reset on reboot)
- The benchmark script runs the workload **twice** (before + after), so it takes longer
- Results are saved in `kernel_benchmark_results/` directory
- The comparison table is formatted for easy copying into reports








