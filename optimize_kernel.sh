#!/bin/bash

# optimize_kernel.sh
# Safe Linux kernel tuning for AI concurrency performance optimization
# This script applies kernel parameters to reduce context switching and scheduling latency
# Idempotent: safe to re-run multiple times

set -e  # Exit on error (but we'll handle errors gracefully)

echo "[INFO] Applying kernel tuning for AI acceleration..."

# Check if running as root (required for sysctl and /sys modifications)
if [ "$EUID" -ne 0 ]; then 
    echo "[ERROR] This script must be run as root (use sudo)"
    exit 1
fi

# ============================================================================
# SCHEDULER TUNING - Reduce context switching overhead
# ============================================================================

# kernel.sched_min_granularity_ns: Minimum time slice for a task before preemption
# Default: ~3-4ms. Setting to 1ms reduces latency for short tasks but may increase
# context switching. For AI workloads, 1ms is a good balance.
# Value: nanoseconds (1000000 ns = 1 ms)
sysctl -w kernel.sched_min_granularity_ns=1000000 2>/dev/null || echo "[WARN] Could not set sched_min_granularity_ns"

# kernel.sched_wakeup_granularity_ns: Time a newly woken task must run before
# preempting the current task. Higher values reduce preemption frequency.
# Default: ~4ms. Setting to 2ms allows faster response to new tasks.
# Value: nanoseconds (2000000 ns = 2 ms)
sysctl -w kernel.sched_wakeup_granularity_ns=2000000 2>/dev/null || echo "[WARN] Could not set sched_wakeup_granularity_ns"

# kernel.sched_migration_cost_ns: Cost threshold for task migration between CPUs
# Higher values reduce unnecessary CPU migrations, improving cache locality.
# Default: ~500000 (0.5ms). Setting to 5ms reduces migrations for AI workloads.
# Value: nanoseconds (5000000 ns = 5 ms)
sysctl -w kernel.sched_migration_cost_ns=5000000 2>/dev/null || echo "[WARN] Could not set sched_migration_cost_ns"

# kernel.sched_autogroup_enabled: Disable automatic process grouping
# Autogrouping can cause scheduling delays. Disabling improves latency for
# single-process AI workloads.
# Value: 0 = disabled, 1 = enabled
sysctl -w kernel.sched_autogroup_enabled=0 2>/dev/null || echo "[WARN] Could not set sched_autogroup_enabled"

# ============================================================================
# MEMORY MANAGEMENT TUNING
# ============================================================================

# vm.swappiness: Controls how aggressively the kernel swaps memory to disk
# Lower values keep more data in RAM, reducing I/O overhead for AI workloads.
# Default: 60. Setting to 10 reduces swapping (good for systems with enough RAM).
# Value: 0-100 (0 = never swap, 100 = always swap)
sysctl -w vm.swappiness=10 2>/dev/null || echo "[WARN] Could not set swappiness"

# vm.nr_hugepages: Allocate huge pages for better memory performance
# Huge pages reduce TLB misses and improve memory access performance.
# 4096 pages × 2MB = 8GB of huge pages (adjust based on your system RAM).
# Note: This reserves memory, so ensure you have enough free RAM.
# Value: number of 2MB pages (on x86_64)
if [ -f /proc/sys/vm/nr_hugepages ]; then
    sysctl -w vm.nr_hugepages=4096 2>/dev/null || echo "[WARN] Could not set nr_hugepages (may need more free RAM)"
else
    echo "[WARN] Huge pages not supported on this system"
fi

# ============================================================================
# TRANSPARENT HUGE PAGES (THP) TUNING
# ============================================================================

# Disable transparent huge pages for more predictable memory behavior
# THP can cause latency spikes during defragmentation. For AI workloads with
# predictable memory patterns, disabling THP can improve consistency.
# Options: always, madvise, never
if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo never > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "[WARN] Could not disable transparent huge pages"
else
    echo "[WARN] Transparent huge pages not available on this system"
fi

# ============================================================================
# I/O SCHEDULER TUNING
# ============================================================================

# Set I/O scheduler to mq-deadline for NVMe SSDs
# mq-deadline is optimized for modern SSDs and provides better latency
# than the default scheduler (often cfq or none).
# Note: This assumes your main disk is nvme0n1. Adjust if different.
# Common alternatives: nvme0n1, sda, sdb, etc.
if [ -d /sys/block/nvme0n1/queue ]; then
    if [ -f /sys/block/nvme0n1/queue/scheduler ]; then
        echo mq-deadline > /sys/block/nvme0n1/queue/scheduler 2>/dev/null || echo "[WARN] Could not set I/O scheduler for nvme0n1"
    else
        echo "[WARN] I/O scheduler not available for nvme0n1"
    fi
else
    # Try to find the main disk (usually sda or the first NVMe device)
    MAIN_DISK=$(lsblk -dno NAME,TYPE | grep -E 'disk|nvme' | head -1 | awk '{print $1}')
    if [ -n "$MAIN_DISK" ] && [ -f /sys/block/$MAIN_DISK/queue/scheduler ]; then
        echo "[INFO] Setting I/O scheduler for $MAIN_DISK"
        echo mq-deadline > /sys/block/$MAIN_DISK/queue/scheduler 2>/dev/null || echo "[WARN] Could not set I/O scheduler for $MAIN_DISK"
    else
        echo "[WARN] Could not find main disk for I/O scheduler tuning"
    fi
fi

# ============================================================================
# VERIFICATION (Optional - show current values)
# ============================================================================

echo ""
echo "[INFO] Kernel tuning complete! Current values:"
echo "  kernel.sched_min_granularity_ns: $(sysctl -n kernel.sched_min_granularity_ns 2>/dev/null || echo 'N/A')"
echo "  kernel.sched_wakeup_granularity_ns: $(sysctl -n kernel.sched_wakeup_granularity_ns 2>/dev/null || echo 'N/A')"
echo "  kernel.sched_migration_cost_ns: $(sysctl -n kernel.sched_migration_cost_ns 2>/dev/null || echo 'N/A')"
echo "  kernel.sched_autogroup_enabled: $(sysctl -n kernel.sched_autogroup_enabled 2>/dev/null || echo 'N/A')"
echo "  vm.swappiness: $(sysctl -n vm.swappiness 2>/dev/null || echo 'N/A')"
echo "  vm.nr_hugepages: $(sysctl -n vm.nr_hugepages 2>/dev/null || echo 'N/A')"

if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo "  transparent_hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo 'N/A')"
fi

if [ -f /sys/block/nvme0n1/queue/scheduler ]; then
    echo "  nvme0n1 scheduler: $(cat /sys/block/nvme0n1/queue/scheduler 2>/dev/null || echo 'N/A')"
fi

echo ""
echo "[INFO] ✅ Kernel tuning applied successfully!"
echo "[NOTE] These changes are temporary and will reset on reboot."
echo "[NOTE] To make them permanent, add them to /etc/sysctl.conf or /etc/sysctl.d/"





