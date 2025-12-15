#!/usr/bin/env python3
"""
Real-time Metrics Dashboard for ML/AI Workload Monitoring
Displays CPU, GPU, Memory, I/O, and Application metrics in a formatted dashboard
"""
import os
import sys
import time
import json
import subprocess
import psutil
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None


class MetricsDashboard:
    """Real-time metrics dashboard for system and application monitoring."""
    
    def __init__(self, logs_dir: str = "logs", update_interval: float = 2.0):
        self.logs_dir = logs_dir
        self.update_interval = update_interval
        self.running = False
        
    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def get_progress_bar(self, value: float, max_value: float = 100.0, width: int = 20) -> str:
        """Generate a progress bar string."""
        if max_value == 0:
            return "[" + "░" * width + "]"
        percentage = min(value / max_value, 1.0)
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {value:.1f}%"
    
    def get_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get GPU metrics using nvidia-smi."""
        gpu_metrics = []
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                for gpu_id in range(n_gpus):
                    try:
                        # Get GPU info using nvidia-smi
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit', 
                             '--format=csv,noheader,nounits'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        lines = result.stdout.strip().split('\n')
                        if gpu_id < len(lines):
                            parts = lines[gpu_id].split(', ')
                            if len(parts) >= 8:
                                gpu_metrics.append({
                                    'id': int(parts[0]),
                                    'util_gpu': float(parts[1]),
                                    'util_mem': float(parts[2]),
                                    'mem_used_mb': float(parts[3]),
                                    'mem_total_mb': float(parts[4]),
                                    'temp': float(parts[5]),
                                    'power_draw': float(parts[6]),
                                    'power_limit': float(parts[7])
                                })
                    except Exception as e:
                        # Fallback to torch if nvidia-smi fails
                        try:
                            mem_info = torch.cuda.mem_get_info(gpu_id)
                            gpu_metrics.append({
                                'id': gpu_id,
                                'util_gpu': 0.0,
                                'util_mem': (mem_info[1] - mem_info[0]) / mem_info[1] * 100,
                                'mem_used_mb': (mem_info[1] - mem_info[0]) / (1024**2),
                                'mem_total_mb': mem_info[1] / (1024**2),
                                'temp': 0.0,
                                'power_draw': 0.0,
                                'power_limit': 0.0
                            })
                        except:
                            pass
        except Exception:
            pass
        return gpu_metrics
    
    def get_cpu_metrics(self) -> Dict[str, Any]:
        """Get CPU metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(percpu=True, interval=0.1)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # Get CPU frequency
            try:
                freq = psutil.cpu_freq()
                freq_current = freq.current if freq else 0
            except:
                freq_current = 0
            
            return {
                'percent': cpu_percent,
                'per_core': cpu_per_core[:8],  # First 8 cores
                'load_1min': load_avg[0],
                'load_5min': load_avg[1],
                'load_15min': load_avg[2],
                'freq_mhz': freq_current,
                'cores': psutil.cpu_count(logical=True)
            }
        except Exception:
            return {'percent': 0, 'per_core': [], 'load_1min': 0, 'load_5min': 0, 'load_15min': 0, 'freq_mhz': 0, 'cores': 0}
    
    def get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory metrics."""
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            return {
                'total_gb': mem.total / (1024**3),
                'used_gb': mem.used / (1024**3),
                'free_gb': mem.free / (1024**3),
                'available_gb': mem.available / (1024**3),
                'percent': mem.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent
            }
        except Exception:
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'available_gb': 0, 'percent': 0, 'swap_total_gb': 0, 'swap_used_gb': 0, 'swap_percent': 0}
    
    def get_disk_metrics(self) -> Dict[str, Any]:
        """Get disk I/O metrics."""
        try:
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            return {
                'total_gb': disk.total / (1024**3),
                'used_gb': disk.used / (1024**3),
                'free_gb': disk.free / (1024**3),
                'percent': disk.percent,
                'read_mb_s': disk_io.read_bytes / (1024**2) if disk_io else 0,
                'write_mb_s': disk_io.write_bytes / (1024**2) if disk_io else 0
            }
        except Exception:
            return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'percent': 0, 'read_mb_s': 0, 'write_mb_s': 0}
    
    def get_process_metrics(self, process_name: str = "python") -> Dict[str, Any]:
        """Get metrics for specific process."""
        try:
            process_metrics = {
                'count': 0,
                'total_cpu': 0.0,
                'total_mem_gb': 0.0,
                'main_pid': None,
                'main_cpu': 0.0,
                'main_mem_gb': 0.0,
                'context_switches': {'voluntary': 0, 'nonvoluntary': 0}
            }
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status']):
                try:
                    if process_name.lower() in proc.info['name'].lower():
                        process_metrics['count'] += 1
                        process_metrics['total_cpu'] += proc.info['cpu_percent'] or 0
                        process_metrics['total_mem_gb'] += (proc.info['memory_info'].rss / (1024**3)) if proc.info['memory_info'] else 0
                        
                        # Get main process (python main.py)
                        cmdline = proc.cmdline() if hasattr(proc, 'cmdline') else []
                        if 'main.py' in ' '.join(cmdline):
                            process_metrics['main_pid'] = proc.info['pid']
                            process_metrics['main_cpu'] = proc.info['cpu_percent'] or 0
                            process_metrics['main_mem_gb'] = (proc.info['memory_info'].rss / (1024**3)) if proc.info['memory_info'] else 0
                            
                            # Get context switches
                            try:
                                status_file = f"/proc/{proc.info['pid']}/status"
                                if os.path.exists(status_file):
                                    with open(status_file, 'r') as f:
                                        for line in f:
                                            if 'voluntary_ctxt_switches:' in line:
                                                process_metrics['context_switches']['voluntary'] = int(line.split()[1])
                                            elif 'nonvoluntary_ctxt_switches:' in line:
                                                process_metrics['context_switches']['nonvoluntary'] = int(line.split()[1])
                            except:
                                pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return process_metrics
        except Exception:
            return {'count': 0, 'total_cpu': 0.0, 'total_mem_gb': 0.0, 'main_pid': None, 'main_cpu': 0.0, 'main_mem_gb': 0.0, 'context_switches': {'voluntary': 0, 'nonvoluntary': 0}}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics from logs."""
        metrics = {
            'throughput_rps': 0.0,
            'avg_latency_s': 0.0,
            'tasks_done': 0,
            'tasks_total': 0,
            'progress_percent': 0.0
        }
        
        try:
            # Try to read from score.json
            score_file = os.path.join(self.logs_dir, 'score.json')
            if os.path.exists(score_file):
                with open(score_file, 'r') as f:
                    score_data = json.load(f)
                    metrics['throughput_rps'] = score_data.get('avg_throughput_rps', 0.0)
                    metrics['avg_latency_s'] = score_data.get('avg_latency_s', 0.0)
                    metrics['tasks_total'] = score_data.get('total', 0)
            
            # Try to read from agent_summary.json
            summary_file = os.path.join(self.logs_dir, 'agent_summary.json')
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    num_agents = summary_data.get('num_agents', 0)
                    num_tasks = summary_data.get('num_tasks', 0)
                    metrics['tasks_total'] = num_agents * num_tasks
                    
                    # Count completed tasks from agents
                    tasks_done = 0
                    for agent in summary_data.get('agents', []):
                        tasks_done += agent.get('total_tasks', 0)
                    metrics['tasks_done'] = tasks_done
            
            if metrics['tasks_total'] > 0:
                metrics['progress_percent'] = (metrics['tasks_done'] / metrics['tasks_total']) * 100
        except Exception:
            pass
        
        return metrics
    
    def format_size(self, size_gb: float) -> str:
        """Format size in GB."""
        if size_gb >= 1024:
            return f"{size_gb/1024:.2f}TB"
        return f"{size_gb:.2f}GB"
    
    def display_dashboard(self):
        """Display the metrics dashboard."""
        self.clear_screen()
        
        # Get all metrics
        cpu_metrics = self.get_cpu_metrics()
        memory_metrics = self.get_memory_metrics()
        disk_metrics = self.get_disk_metrics()
        gpu_metrics = self.get_gpu_metrics()
        process_metrics = self.get_process_metrics()
        app_metrics = self.get_application_metrics()
        
        # Header
        print("=" * 80)
        print(f"📊 REAL-TIME METRICS DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()
        
        # System Overview
        print("┌" + "─" * 78 + "┐")
        print("│ SYSTEM OVERVIEW" + " " * 63 + "│")
        print("├" + "─" * 78 + "┤")
        cpu_bar = self.get_progress_bar(cpu_metrics['percent'], 100, 20)
        mem_bar = self.get_progress_bar(memory_metrics['percent'], 100, 20)
        disk_bar = self.get_progress_bar(disk_metrics['percent'], 100, 20)
        
        print(f"│ CPU:  {cpu_bar}  |  Load: {cpu_metrics['load_1min']:.2f}/{cpu_metrics['load_5min']:.2f}/{cpu_metrics['load_15min']:.2f}  |  Freq: {cpu_metrics['freq_mhz']:.0f}MHz")
        print(f"│ RAM:  {mem_bar}  |  {self.format_size(memory_metrics['used_gb'])}/{self.format_size(memory_metrics['total_gb'])}  |  Swap: {memory_metrics['swap_percent']:.1f}%")
        print(f"│ Disk: {disk_bar}  |  {self.format_size(disk_metrics['used_gb'])}/{self.format_size(disk_metrics['total_gb'])}  |  I/O: {disk_metrics['read_mb_s']:.1f}MB/s read, {disk_metrics['write_mb_s']:.1f}MB/s write")
        print("└" + "─" * 78 + "┘")
        print()
        
        # GPU Metrics
        if gpu_metrics:
            print("┌" + "─" * 78 + "┐")
            print("│ GPU METRICS (Per GPU)" + " " * 55 + "│")
            print("├" + "─" * 78 + "┤")
            for gpu in gpu_metrics:
                util_bar = self.get_progress_bar(gpu['util_gpu'], 100, 15)
                mem_percent = (gpu['mem_used_mb'] / gpu['mem_total_mb']) * 100 if gpu['mem_total_mb'] > 0 else 0
                mem_bar = self.get_progress_bar(mem_percent, 100, 15)
                mem_used_gb = gpu['mem_used_mb'] / 1024
                mem_total_gb = gpu['mem_total_mb'] / 1024
                
                temp_status = "🟢" if gpu['temp'] < 70 else "🟡" if gpu['temp'] < 80 else "🔴"
                print(f"│ GPU {gpu['id']}: {util_bar} {gpu['util_gpu']:.0f}% util | {mem_bar} {mem_used_gb:.1f}GB/{mem_total_gb:.1f}GB | {temp_status} {gpu['temp']:.0f}°C | {gpu['power_draw']:.0f}W/{gpu['power_limit']:.0f}W")
            print("└" + "─" * 78 + "┘")
            print()
        
        # Application Metrics
        print("┌" + "─" * 78 + "┐")
        print("│ APPLICATION METRICS" + " " * 59 + "│")
        print("├" + "─" * 78 + "┤")
        progress_bar = self.get_progress_bar(app_metrics['progress_percent'], 100, 20)
        print(f"│ Throughput: {app_metrics['throughput_rps']:.3f} RPS  |  Latency: {app_metrics['avg_latency_s']:.2f}s avg")
        print(f"│ Tasks: {app_metrics['tasks_done']:,}/{app_metrics['tasks_total']:,}  |  Progress: {progress_bar} {app_metrics['progress_percent']:.1f}%")
        print("└" + "─" * 78 + "┘")
        print()
        
        # Process Metrics
        if process_metrics['main_pid']:
            print("┌" + "─" * 78 + "┐")
            print("│ PROCESS METRICS (Main Process)" + " " * 48 + "│")
            print("├" + "─" * 78 + "┤")
            total_switches = process_metrics['context_switches']['voluntary'] + process_metrics['context_switches']['nonvoluntary']
            print(f"│ PID: {process_metrics['main_pid']}  |  CPU: {process_metrics['main_cpu']:.1f}%  |  Memory: {process_metrics['main_mem_gb']:.2f}GB")
            print(f"│ Context Switches: {total_switches:,} total ({process_metrics['context_switches']['voluntary']:,} voluntary, {process_metrics['context_switches']['nonvoluntary']:,} non-voluntary)")
            print(f"│ Python Processes: {process_metrics['count']} total  |  Total CPU: {process_metrics['total_cpu']:.1f}%  |  Total Memory: {process_metrics['total_mem_gb']:.2f}GB")
            print("└" + "─" * 78 + "┘")
            print()
        
        # CPU Cores (if available)
        if cpu_metrics['per_core']:
            print("┌" + "─" * 78 + "┐")
            print("│ CPU CORES (First 8)" + " " * 60 + "│")
            print("├" + "─" * 78 + "┤")
            core_str = "│ "
            for i, core_percent in enumerate(cpu_metrics['per_core'][:8]):
                core_bar = self.get_progress_bar(core_percent, 100, 5)
                core_str += f"C{i}: {core_bar}  "
            print(core_str)
            print("└" + "─" * 78 + "┘")
            print()
        
        # Footer
        print("=" * 80)
        print(f"Update interval: {self.update_interval}s  |  Press Ctrl+C to exit")
        print("=" * 80)
    
    def run(self):
        """Run the dashboard in a loop."""
        self.running = True
        try:
            while self.running:
                self.display_dashboard()
                time.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user.")
            self.running = False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Metrics Dashboard')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory containing log files')
    parser.add_argument('--interval', type=float, default=2.0, help='Update interval in seconds')
    args = parser.parse_args()
    
    dashboard = MetricsDashboard(logs_dir=args.logs_dir, update_interval=args.interval)
    dashboard.run()


if __name__ == '__main__':
    main()







