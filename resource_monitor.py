# resource_monitor.py
"""
Resource monitoring module for CPU, GPU, RAM, disk, and network metrics.
Logs metrics to both Weave and Weights & Biases (W&B) dashboards.
"""
import os
import time
import psutil
import threading
from typing import Dict, List, Any, Optional

# Optional imports with graceful fallback
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    # Create a dummy decorator if weave is not available
    def weave_op():
        def decorator(func):
            return func
        return decorator
    weave = type('obj', (object,), {'op': lambda: weave_op()})()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available() if torch.cuda.is_available() else False
except ImportError:
    TORCH_AVAILABLE = False


def _log_resources_impl() -> Dict[str, Any]:
    """
    Collect and log system resource metrics to Weave and W&B.
    
    Returns:
        Dictionary containing all collected metrics
    """
    metrics = {}
    
    # CPU & RAM metrics
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    net = psutil.net_io_counters()
    
    # Enhanced CPU metrics (per-core and overall)
    cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=None)
    
    # Enhanced memory metrics
    swap = psutil.swap_memory()
    
    # Enhanced disk I/O metrics
    disk_io = psutil.disk_io_counters()
    
    # Enhanced network metrics with rates
    net_io = psutil.net_io_counters()
    
    # Process count and system load
    process_count = len(psutil.pids())
    
    metrics.update({
        # Overall CPU
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        
        # Per-core CPU (sample a few cores for visualization)
        "cpu_core_0": cpu_percent_per_core[0] if len(cpu_percent_per_core) > 0 else 0,
        "cpu_core_1": cpu_percent_per_core[1] if len(cpu_percent_per_core) > 1 else 0,
        "cpu_core_2": cpu_percent_per_core[2] if len(cpu_percent_per_core) > 2 else 0,
        "cpu_core_3": cpu_percent_per_core[3] if len(cpu_percent_per_core) > 3 else 0,
        
        # RAM metrics
        "ram_used_mb": round(mem.used / (1024**2), 2),
        "ram_total_mb": round(mem.total / (1024**2), 2),
        "ram_percent": mem.percent,
        "ram_available_mb": round(mem.available / (1024**2), 2),
        "ram_free_mb": round(mem.free / (1024**2), 2),
        "ram_cached_mb": round(getattr(mem, 'cached', 0) / (1024**2), 2),
        
        # Swap metrics
        "swap_used_mb": round(swap.used / (1024**2), 2),
        "swap_total_mb": round(swap.total / (1024**2), 2),
        "swap_percent": swap.percent,
        
        # Disk usage
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "disk_percent": round(disk.used / disk.total * 100, 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        
        # Disk I/O
        "disk_read_mb": round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
        "disk_write_mb": round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0,
        "disk_read_count": disk_io.read_count if disk_io else 0,
        "disk_write_count": disk_io.write_count if disk_io else 0,
        
        # Network I/O
        "net_sent_mb": round(net_io.bytes_sent / (1024**2), 2),
        "net_recv_mb": round(net_io.bytes_recv / (1024**2), 2),
        "net_packets_sent": net_io.packets_sent,
        "net_packets_recv": net_io.packets_recv,
        "net_errors_in": net_io.errin,
        "net_errors_out": net_io.errout,
        
        # System metrics
        "process_count": process_count,
    })
    
    # GPU metrics
    gpu_stats = []
    
    # Try GPUtil first (more detailed info)
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_stats.append({
                    "id": gpu.id,
                    "name": gpu.name,
                    "load_%": round(gpu.load * 100, 2),
                    "mem_used_mb": round(gpu.memoryUsed, 2),
                    "mem_free_mb": round(gpu.memoryFree, 2),
                    "mem_total_mb": round(gpu.memoryTotal, 2),
                    "mem_percent": round(gpu.memoryUsed / gpu.memoryTotal * 100, 2) if gpu.memoryTotal > 0 else 0,
                    "temp_c": getattr(gpu, "temperature", None)
                })
        except Exception as e:
            print(f"⚠️  GPUtil error: {e}")
            gpu_stats = []
    
    # Fallback to torch.cuda if available
    if TORCH_AVAILABLE and (not gpu_stats or len(gpu_stats) == 0):
        try:
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
                mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**2)  # MB
                
                gpu_stats.append({
                    "id": i,
                    "name": device_name,
                    "load_%": None,  # Not available via torch
                    "mem_used_mb": round(mem_allocated, 2),
                    "mem_total_mb": round(mem_total, 2),
                    "mem_percent": round(mem_allocated / mem_total * 100, 2) if mem_total > 0 else 0,
                    "temp_c": None  # Not available via torch
                })
        except Exception as e:
            print(f"⚠️  torch.cuda error: {e}")
    
    # Add per-GPU metrics to main metrics dict (for graph visualization)
    if gpu_stats:
        metrics["gpu_count"] = len(gpu_stats)
        for i, gpu in enumerate(gpu_stats):
            # Use clean metric names that W&B will automatically graph
            prefix = f"gpu_{i}_"
            metrics[f"{prefix}load_percent"] = gpu["load_%"] if gpu["load_%"] is not None else 0.0
            metrics[f"{prefix}memory_used_mb"] = gpu["mem_used_mb"]
            metrics[f"{prefix}memory_free_mb"] = gpu.get("mem_free_mb", gpu["mem_total_mb"] - gpu["mem_used_mb"])
            metrics[f"{prefix}memory_total_mb"] = gpu["mem_total_mb"]
            metrics[f"{prefix}memory_percent"] = gpu["mem_percent"]
            if gpu["temp_c"] is not None:
                metrics[f"{prefix}temperature_c"] = gpu["temp_c"]
            # Keep name as metadata (not graphed)
            metrics[f"{prefix}name"] = gpu["name"]
    else:
        metrics["gpu_count"] = 0
    
    # Store full GPU info array for reference
    metrics["gpu_info"] = gpu_stats
    
    # Note: W&B logging is now handled in ResourceMonitor._monitor_loop()
    # to ensure proper step tracking for time-series graphs
    # This function just collects metrics and returns them
    
    return metrics

# Apply weave decorator conditionally
# Disable Weave if it's causing HTTP 500 errors (server-side issues)
if WEAVE_AVAILABLE:
    try:
        # Check if Weave should be disabled
        if os.environ.get("DISABLE_WEAVE", "false").lower() == "true":
            log_resources = _log_resources_impl
        else:
            try:
                log_resources = weave.op()(_log_resources_impl)
            except Exception as e:
                # If weave.op() fails, use the function directly
                print(f"⚠️  Weave decorator failed, using function directly: {e}")
                log_resources = _log_resources_impl
    except Exception as e:
        # If any Weave operation fails, disable it
        print(f"⚠️  Weave operation failed, disabling Weave: {e}")
        log_resources = _log_resources_impl
else:
    log_resources = _log_resources_impl


class ResourceMonitor:
    """
    Resource monitor that runs in a background thread and logs metrics periodically.
    """
    def __init__(self, interval: int = 10):
        """
        Initialize resource monitor.
        
        Args:
            interval: Time in seconds between metric collection (default: 10)
        """
        self.interval = interval
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the resource monitor in a background thread."""
        if self.running:
            print("⚠️  Resource monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"✅ Resource monitor started (interval: {self.interval}s)")
    
    def stop(self):
        """Stop the resource monitor."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Resource monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread."""
        step = 0
        while self.running:
            try:
                metrics = log_resources()
                
                # Add step counter for time-series visualization
                # This ensures W&B creates proper graphs over time
                if WANDB_AVAILABLE and wandb and hasattr(wandb, 'log'):
                    try:
                        if hasattr(wandb, 'run') and wandb.run is not None:
                            # Log with step for time-series graphs
                            # Don't include '_step' in the metrics dict, just pass step= parameter
                            wandb.log(metrics, step=step)
                            step += 1
                        else:
                            # W&B not initialized yet, skip logging
                            pass
                    except Exception as e:
                        # Silently skip if W&B not ready
                        pass
                
                # Print summary every 10th iteration to avoid spam
                if hasattr(self, '_counter'):
                    self._counter += 1
                else:
                    self._counter = 0
                
                if self._counter % 6 == 0:  # Print every 6 iterations (30 seconds with 5s interval)
                    # Get run info for display
                    run_info = ""
                    if WANDB_AVAILABLE and wandb and hasattr(wandb, 'run') and wandb.run:
                        config = wandb.run.config
                        if hasattr(config, 'num_agents') and hasattr(config, 'num_tasks'):
                            run_info = f" [{config.num_agents} agents, {config.num_tasks} tasks]"
                    
                    gpu_info = ""
                    if metrics.get('gpu_count', 0) > 0:
                        gpu_load = metrics.get('gpu_0_load_percent', 0)
                        gpu_mem = metrics.get('gpu_0_memory_percent', 0)
                        gpu_info = f", GPU0={gpu_load:.1f}% load, {gpu_mem:.1f}% mem"
                    print(f"📊 Resources (step {step}){run_info}: CPU={metrics.get('cpu_percent', 0):.1f}%, "
                          f"RAM={metrics.get('ram_percent', 0):.1f}%{gpu_info}")
            except Exception as e:
                print(f"⚠️  Resource monitoring error: {e}")
            
            time.sleep(self.interval)


def start_monitor(interval: int = 10) -> ResourceMonitor:
    """
    Convenience function to start resource monitoring.
    
    Args:
        interval: Time in seconds between metric collection
        
    Returns:
        ResourceMonitor instance
    """
    monitor = ResourceMonitor(interval=interval)
    monitor.start()
    return monitor

