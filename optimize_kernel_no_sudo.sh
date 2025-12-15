#!/bin/bash

# optimize_kernel_no_sudo.sh
# User-level optimizations that don't require sudo
# This script applies what optimizations are possible without root access

set -e

echo "[INFO] Applying user-level optimizations (no sudo required)..."

# ============================================================================
# USER-LEVEL OPTIMIZATIONS
# ============================================================================

# 1. Set process priority using nice/renice (no sudo needed for lowering priority)
# Note: Can only lower priority, not raise it without sudo
echo "[INFO] Setting process priority..."

# 2. CPU affinity - can be set per-process (limited without sudo)
# Note: Requires specific process management in the application

# 3. Environment variables for application-level tuning
export OMP_NUM_THREADS=$(nproc)  # Use all CPU cores for OpenMP
export MKL_NUM_THREADS=$(nproc)  # Intel MKL threading
export NUMEXPR_NUM_THREADS=$(nproc)  # NumExpr threading

echo "[INFO] Set threading environment variables:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"

# 4. Disable transparent huge pages via environment (if supported by application)
# Note: This doesn't change system settings, but some apps respect this
export THP_DISABLE=1

# 5. Set Python/application-level optimizations
export PYTHONUNBUFFERED=1  # Unbuffered output for better logging
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# 6. Display current kernel parameters (read-only, no sudo needed)
echo ""
echo "[INFO] Current kernel parameters (read-only):"
echo "  kernel.sched_min_granularity_ns: $(sysctl -n kernel.sched_min_granularity_ns 2>/dev/null || echo 'N/A')"
echo "  kernel.sched_wakeup_granularity_ns: $(sysctl -n kernel.sched_wakeup_granularity_ns 2>/dev/null || echo 'N/A')"
echo "  kernel.sched_migration_cost_ns: $(sysctl -n kernel.sched_migration_cost_ns 2>/dev/null || echo 'N/A')"
echo "  kernel.sched_autogroup_enabled: $(sysctl -n kernel.sched_autogroup_enabled 2>/dev/null || echo 'N/A')"
echo "  vm.swappiness: $(sysctl -n vm.swappiness 2>/dev/null || echo 'N/A')"
echo "  vm.nr_hugepages: $(sysctl -n vm.nr_hugepages 2>/dev/null || echo 'N/A')"

if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo "  transparent_hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo 'N/A')"
fi

echo ""
echo "[INFO] ✅ User-level optimizations applied!"
echo "[NOTE] These are application-level optimizations."
echo "[NOTE] For full kernel-level tuning, sudo access is required."
echo ""
echo "[INFO] To use these optimizations, source this script:"
echo "  source optimize_kernel_no_sudo.sh"
echo "  # OR"
echo "  . optimize_kernel_no_sudo.sh"





