#!/usr/bin/env python3
"""
kernel_benchmark.py
Automated benchmark script that runs agent_concurrency.py before and after kernel tuning,
captures performance metrics, and generates a comparison report.
"""

import os
import sys
import json
import subprocess
import time
import signal
import threading
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import re

class BenchmarkRunner:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.results_dir = self.project_dir / "kernel_benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def find_main_process_pid(self, process_name: str = "python") -> Optional[int]:
        """Find the PID of the main Python process running agent_concurrency.py"""
        try:
            # Use pgrep to find the process
            result = subprocess.run(
                ["pgrep", "-f", "agent_concurrency.py"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = [int(pid) for pid in result.stdout.strip().split('\n') if pid]
                # Return the first PID (usually the main process)
                return pids[0] if pids else None
        except Exception as e:
            print(f"[WARN] Could not find process PID: {e}")
        return None
    
    def run_pidstat(self, pid: Optional[int], duration: int, output_file: str) -> subprocess.Popen:
        """Run pidstat to monitor context switches and CPU usage"""
        cmd = ["pidstat", "-w", "1", "-p", str(pid)] if pid else ["pidstat", "-w", "1", "-u", "1"]
        
        # If pid is None, monitor all processes (less accurate but works)
        if pid is None:
            cmd = ["pidstat", "-w", "1", "-u", "1"]
        
        try:
            with open(output_file, "w") as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True
                )
            return proc
        except FileNotFoundError:
            print("[WARN] pidstat not found. Install with: sudo apt-get install sysstat")
            return None
        except Exception as e:
            print(f"[WARN] Could not start pidstat: {e}")
            return None
    
    def run_runqlat(self, duration: int, output_file: str) -> Optional[subprocess.Popen]:
        """Run runqlat (BCC tool) to measure kernel scheduling latency"""
        # Check if BCC tools are available
        runqlat_paths = [
            "/usr/share/bcc/tools/runqlat",
            "/usr/local/share/bcc/tools/runqlat",
            "runqlat"  # Try in PATH
        ]
        
        runqlat_cmd = None
        for path in runqlat_paths:
            if shutil.which(path) or os.path.exists(path):
                runqlat_cmd = path
                break
        
        if not runqlat_cmd:
            print("[INFO] runqlat (BCC tool) not found. Skipping kernel scheduling latency measurement.")
            print("[INFO] Install BCC tools: sudo apt-get install bpfcc-tools")
            return None
        
        try:
            # runqlat runs for specified duration and outputs histogram
            # We'll run it in the background and capture output
            with open(output_file, "w") as f:
                proc = subprocess.Popen(
                    ["sudo", runqlat_cmd, str(duration)],
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True
                )
            return proc
        except Exception as e:
            print(f"[WARN] Could not start runqlat: {e}")
            return None
    
    def parse_runqlat_output(self, runqlat_file: str) -> Dict[str, float]:
        """Parse runqlat output to extract average scheduling latency"""
        if not os.path.exists(runqlat_file):
            return {"avg_sched_latency_us": 0.0}
        
        try:
            with open(runqlat_file, "r") as f:
                content = f.read()
            
            # runqlat output is a histogram with latency buckets
            # Format: usecs : count distribution
            # We need to calculate weighted average
            
            latency_values = []
            counts = []
            
            # Parse histogram lines
            for line in content.split('\n'):
                if ':' in line and 'distribution' in line:
                    # Extract latency range and count
                    # Example: "0-1          : 12345    |@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@|"
                    parts = line.split(':')
                    if len(parts) >= 2:
                        range_str = parts[0].strip()
                        count_str = parts[1].split('|')[0].strip()
                        
                        try:
                            # Parse range (e.g., "0-1" or "2-3")
                            if '-' in range_str:
                                low, high = map(int, range_str.split('-'))
                                mid = (low + high) / 2
                            else:
                                mid = int(range_str)
                            
                            count = int(count_str)
                            
                            latency_values.append(mid)
                            counts.append(count)
                        except (ValueError, IndexError):
                            continue
            
            # Calculate weighted average
            if latency_values and counts:
                total_count = sum(counts)
                if total_count > 0:
                    weighted_sum = sum(lat * count for lat, count in zip(latency_values, counts))
                    avg_latency = weighted_sum / total_count
                    return {"avg_sched_latency_us": avg_latency}
            
            # Fallback: try to find average in summary lines
            for line in content.split('\n'):
                if 'avg' in line.lower() or 'average' in line.lower():
                    # Try to extract number
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        return {"avg_sched_latency_us": float(numbers[0])}
            
            return {"avg_sched_latency_us": 0.0}
        except Exception as e:
            print(f"[WARN] Could not parse runqlat output: {e}")
            return {"avg_sched_latency_us": 0.0}
    
    def parse_pidstat_output(self, pidstat_file: str) -> Dict[str, float]:
        """Parse pidstat output to extract context switch rate and CPU utilization"""
        if not os.path.exists(pidstat_file):
            return {"context_switches_per_sec": 0.0, "cpu_util": 0.0}
        
        try:
            with open(pidstat_file, "r") as f:
                lines = f.readlines()
            
            # pidstat output format:
            # Time      UID       PID   cswch/s nvcswch/s  Command
            # 10:00:01     0      1234   100.00    50.00    python
            
            cswch_values = []
            nvcswch_values = []
            cpu_values = []
            
            for line in lines:
                # Skip header lines
                if "PID" in line or "Linux" in line or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        # Try to parse context switch values
                        if parts[3].replace('.', '').isdigit():
                            cswch = float(parts[3])
                            nvcswch = float(parts[4]) if len(parts) > 4 else 0.0
                            cswch_values.append(cswch)
                            nvcswch_values.append(nvcswch)
                    except (ValueError, IndexError):
                        pass
                
                # Try to parse CPU if available (pidstat -u format)
                if len(parts) >= 4:
                    try:
                        cpu_str = parts[3] if len(parts) > 3 else parts[2]
                        if cpu_str.replace('.', '').replace('%', '').isdigit():
                            cpu = float(cpu_str.replace('%', ''))
                            cpu_values.append(cpu)
                    except (ValueError, IndexError):
                        pass
            
            # Calculate averages
            avg_cswch = sum(cswch_values) / len(cswch_values) if cswch_values else 0.0
            avg_nvcswch = sum(nvcswch_values) / len(nvcswch_values) if nvcswch_values else 0.0
            avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
            
            total_switches = avg_cswch + avg_nvcswch
            
            return {
                "context_switches_per_sec": total_switches,
                "cpu_util": avg_cpu,
                "voluntary_cswch": avg_cswch,
                "nonvoluntary_cswch": avg_nvcswch
            }
        except Exception as e:
            print(f"[WARN] Could not parse pidstat output: {e}")
            return {"context_switches_per_sec": 0.0, "cpu_util": 0.0}
    
    def extract_metrics_from_logs(self, logs_dir: str) -> Dict[str, float]:
        """Extract metrics from the logs directory (score.json, agent_summary.json)"""
        metrics = {
            "avg_latency_s": 0.0,
            "throughput_rps": 0.0,
            "successes": 0,
            "gpu_idle": 0.0
        }
        
        logs_path = Path(logs_dir)
        
        # Try to read score.json
        score_file = logs_path / "score.json"
        if score_file.exists():
            try:
                with open(score_file, "r") as f:
                    score_data = json.load(f)
                    metrics["avg_latency_s"] = score_data.get("avg_latency_s", 0.0)
                    metrics["throughput_rps"] = score_data.get("avg_throughput_rps", 0.0)
                    metrics["successes"] = score_data.get("pass@1", 0.0) * score_data.get("total", 0)
            except Exception as e:
                print(f"[WARN] Could not read score.json: {e}")
        
        # Try to read agent_summary.json for more detailed metrics
        summary_file = logs_path / "agent_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    summary_data = json.load(f)
                    agents = summary_data.get("agents", [])
                    if agents:
                        # Calculate averages across all agents
                        avg_latencies = [a.get("avg_latency_s", 0.0) for a in agents if a.get("avg_latency_s")]
                        avg_throughputs = [a.get("throughput_rps", 0.0) for a in agents if a.get("throughput_rps")]
                        
                        if avg_latencies:
                            metrics["avg_latency_s"] = sum(avg_latencies) / len(avg_latencies)
                        if avg_throughputs:
                            metrics["throughput_rps"] = sum(avg_throughputs) / len(avg_throughputs)
                        if agents:
                            metrics["successes"] = sum(a.get("successes", 0) for a in agents)
            except Exception as e:
                print(f"[WARN] Could not read agent_summary.json: {e}")
        
        # Try to estimate GPU idle time from gpu_usage.csv if available
        gpu_csv = logs_path / "gpu_usage.csv"
        if gpu_csv.exists():
            try:
                import csv
                gpu_utils = []
                with open(gpu_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Look for GPU utilization columns
                        for key, value in row.items():
                            if "util" in key.lower() or "gpu" in key.lower():
                                try:
                                    util = float(value)
                                    gpu_utils.append(util)
                                except (ValueError, TypeError):
                                    pass
                
                if gpu_utils:
                    avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
                    metrics["gpu_idle"] = max(0.0, 100.0 - avg_gpu_util)  # Idle = 100 - utilization
            except Exception as e:
                print(f"[WARN] Could not read GPU usage: {e}")
        
        return metrics
    
    def run_benchmark_phase(self, phase: str, apply_optimization: bool = False) -> Dict[str, any]:
        """Run a single benchmark phase (before or after optimization)"""
        print(f"\n{'='*70}")
        print(f"🧪 BENCHMARK PHASE: {phase.upper()}")
        print(f"{'='*70}\n")
        
        # Create a temporary logs directory for this phase
        phase_logs_dir = self.project_dir / "logs" / f"benchmark_{phase}"
        phase_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup original logs if they exist
        original_logs = self.project_dir / "logs"
        if (original_logs / "score.json").exists():
            # Move existing logs to backup
            backup_dir = original_logs / f"backup_{int(time.time())}"
            backup_dir.mkdir(exist_ok=True)
            for f in original_logs.glob("*.json"):
                if f.name != "score.json" or phase == "before":
                    shutil.copy2(f, backup_dir / f.name)
        
        # Apply kernel optimization if requested
        if apply_optimization:
            print("[INFO] Applying kernel optimization...")
            optimize_script = self.project_dir / "optimize_kernel.sh"
            if optimize_script.exists():
                try:
                    result = subprocess.run(
                        ["sudo", str(optimize_script)],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode == 0:
                        print("[SUCCESS] Kernel optimization applied")
                    else:
                        print(f"[WARN] Kernel optimization returned code {result.returncode}")
                except Exception as e:
                    print(f"[WARN] Could not apply kernel optimization: {e}")
            else:
                print(f"[WARN] optimize_kernel.sh not found at {optimize_script}")
        
        # Start pidstat monitoring in background
        pidstat_file = self.results_dir / f"pidstat_{phase}.txt"
        pidstat_proc = None
        
        # Start runqlat for kernel scheduling latency (optional, BCC tool)
        runqlat_file = self.results_dir / f"runqlat_{phase}.txt"
        runqlat_proc = None
        
        # Try to find the process PID (will be set after we start the process)
        main_pid = None
        
        # Start the agent_concurrency.py script
        script_path = self.project_dir / "agent_concurrency.py"
        if not script_path.exists():
            print(f"[ERROR] agent_concurrency.py not found at {script_path}")
            return {}
        
        print(f"[INFO] Starting agent_concurrency.py...")
        print(f"[INFO] Logs will be saved to: {phase_logs_dir}")
        
        # Change to project directory
        original_cwd = os.getcwd()
        os.chdir(self.project_dir)
        
        try:
            # Start the process
            start_time = time.time()
            
            # Run the script (without --optimize flag to avoid double optimization)
            cmd = [sys.executable, str(script_path)]
            if apply_optimization:
                # If we already applied optimization manually, don't pass --optimize
                pass
            else:
                # For "before" phase, don't optimize
                pass
            
            # Add any additional arguments from command line
            if len(sys.argv) > 1:
                # Skip script name and phase arguments
                additional_args = [arg for arg in sys.argv[1:] if arg not in ["--phase", phase]]
                cmd.extend(additional_args)
            
            print(f"[INFO] Running: {' '.join(cmd)}")
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.project_dir)
            )
            
            # Wait a moment for process to start, then find PID
            time.sleep(2)
            main_pid = process.pid
            
            # Start pidstat monitoring
            if main_pid:
                print(f"[INFO] Monitoring process PID {main_pid} with pidstat...")
                pidstat_proc = self.run_pidstat(main_pid, duration=3600, output_file=str(pidstat_file))
            
            # Start runqlat for kernel scheduling latency (optional)
            print(f"[INFO] Starting runqlat for kernel scheduling latency measurement...")
            runqlat_proc = self.run_runqlat(duration=3600, output_file=str(runqlat_file))
            
            # Wait for process to complete
            stdout, stderr = process.communicate()
            end_time = time.time()
            runtime = end_time - start_time
            
            # Stop pidstat
            if pidstat_proc:
                try:
                    pidstat_proc.terminate()
                    pidstat_proc.wait(timeout=5)
                except:
                    pidstat_proc.kill()
            
            # Stop runqlat
            if runqlat_proc:
                try:
                    runqlat_proc.terminate()
                    runqlat_proc.wait(timeout=5)
                except:
                    runqlat_proc.kill()
            
            # Check return code
            if process.returncode != 0:
                print(f"[WARN] Process exited with code {process.returncode}")
                if stderr:
                    print(f"[ERROR] {stderr[:500]}")  # Print first 500 chars of stderr
            
            print(f"[INFO] Process completed in {runtime:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n[INFO] Benchmark interrupted by user")
            if pidstat_proc:
                pidstat_proc.terminate()
            if runqlat_proc:
                runqlat_proc.terminate()
            if process:
                process.terminate()
            return {}
        except Exception as e:
            print(f"[ERROR] Failed to run benchmark: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            os.chdir(original_cwd)
        
        # Extract metrics from logs
        logs_dir = self.project_dir / "logs"
        metrics = self.extract_metrics_from_logs(str(logs_dir))
        
        # Parse pidstat output
        pidstat_metrics = self.parse_pidstat_output(str(pidstat_file))
        metrics.update(pidstat_metrics)
        
        # Parse runqlat output (if available)
        runqlat_metrics = self.parse_runqlat_output(str(runqlat_file))
        metrics.update(runqlat_metrics)
        
        # Add phase and runtime
        metrics["phase"] = phase
        metrics["runtime_s"] = runtime
        
        # Save results
        results_file = self.results_dir / f"results_{phase}.json"
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n[SUCCESS] Phase {phase} completed!")
        print(f"  Metrics saved to: {results_file}")
        
        return metrics
    
    def generate_comparison_table(self, before: Dict, after: Dict) -> str:
        """Generate a formatted comparison table"""
        def format_value(value, unit=""):
            if isinstance(value, float):
                if value >= 1000:
                    return f"{value/1000:.1f}K{unit}"
                elif value >= 1:
                    return f"{value:.2f}{unit}"
                else:
                    return f"{value:.3f}{unit}"
            return str(value)
        
        def calculate_improvement(before_val, after_val, higher_is_better=True):
            if before_val == 0:
                return "N/A"
            change = ((after_val - before_val) / before_val) * 100
            if higher_is_better:
                sign = "+" if change > 0 else ""
            else:
                sign = "-" if change > 0 else "+"
                change = abs(change)
            return f"{sign}{change:.1f}%"
        
        # Build table
        lines = []
        lines.append("=" * 80)
        lines.append("KERNEL OPTIMIZATION BENCHMARK RESULTS")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"{'Metric':<25} {'Before':<20} {'After':<20} {'Δ Improvement':<15}")
        lines.append("-" * 80)
        
        # Latency (lower is better)
        before_lat = before.get("avg_latency_s", 0.0)
        after_lat = after.get("avg_latency_s", 0.0)
        lat_improvement = calculate_improvement(before_lat, after_lat, higher_is_better=False)
        lines.append(f"{'avg_latency_s':<25} {format_value(before_lat, ' s'):<20} {format_value(after_lat, ' s'):<20} {lat_improvement:<15}")
        
        # Throughput (higher is better)
        before_thru = before.get("throughput_rps", 0.0)
        after_thru = after.get("throughput_rps", 0.0)
        thru_improvement = calculate_improvement(before_thru, after_thru, higher_is_better=True)
        lines.append(f"{'throughput_rps':<25} {format_value(before_thru, ' req/s'):<20} {format_value(after_thru, ' req/s'):<20} {thru_improvement:<15}")
        
        # Context switches (lower is better)
        before_csw = before.get("context_switches_per_sec", 0.0)
        after_csw = after.get("context_switches_per_sec", 0.0)
        csw_improvement = calculate_improvement(before_csw, after_csw, higher_is_better=False)
        lines.append(f"{'context_switches':<25} {format_value(before_csw, ' /s'):<20} {format_value(after_csw, ' /s'):<20} {csw_improvement:<15}")
        
        # CPU utilization
        before_cpu = before.get("cpu_util", 0.0)
        after_cpu = after.get("cpu_util", 0.0)
        cpu_improvement = calculate_improvement(before_cpu, after_cpu, higher_is_better=True)
        lines.append(f"{'cpu_util':<25} {format_value(before_cpu, ' %'):<20} {format_value(after_cpu, ' %'):<20} {cpu_improvement:<15}")
        
        # GPU idle time (lower is better - means more GPU usage)
        before_gpu_idle = before.get("gpu_idle", 0.0)
        after_gpu_idle = after.get("gpu_idle", 0.0)
        gpu_improvement = calculate_improvement(before_gpu_idle, after_gpu_idle, higher_is_better=False)
        lines.append(f"{'gpu_idle_time':<25} {format_value(before_gpu_idle, ' %'):<20} {format_value(after_gpu_idle, ' %'):<20} {gpu_improvement:<15}")
        
        # Kernel scheduling latency (if available from runqlat)
        before_sched_lat = before.get("avg_sched_latency_us", 0.0)
        after_sched_lat = after.get("avg_sched_latency_us", 0.0)
        if before_sched_lat > 0 or after_sched_lat > 0:
            sched_improvement = calculate_improvement(before_sched_lat, after_sched_lat, higher_is_better=False)
            lines.append(f"{'sched_latency':<25} {format_value(before_sched_lat, ' μs'):<20} {format_value(after_sched_lat, ' μs'):<20} {sched_improvement:<15}")
        
        # Successes
        before_succ = before.get("successes", 0)
        after_succ = after.get("successes", 0)
        succ_improvement = calculate_improvement(before_succ, after_succ, higher_is_better=True)
        lines.append(f"{'successes':<25} {format_value(before_succ):<20} {format_value(after_succ):<20} {succ_improvement:<15}")
        
        lines.append("-" * 80)
        lines.append("")
        
        return "\n".join(lines)
    
    def run_full_benchmark(self):
        """Run the complete before/after benchmark"""
        print("\n" + "="*70)
        print("🚀 KERNEL-LEVEL PERFORMANCE BENCHMARK")
        print("="*70)
        print("\nThis will:")
        print("  1. Run agent_concurrency.py WITHOUT kernel optimization")
        print("  2. Apply kernel optimization")
        print("  3. Run agent_concurrency.py WITH kernel optimization")
        print("  4. Compare results and generate a report")
        print("\n" + "="*70 + "\n")
        
        # Phase 1: Before optimization
        before_results = self.run_benchmark_phase("before", apply_optimization=False)
        
        if not before_results:
            print("[ERROR] Before phase failed. Aborting benchmark.")
            return
        
        # Wait a moment between phases
        print("\n[INFO] Waiting 5 seconds before next phase...")
        time.sleep(5)
        
        # Phase 2: After optimization
        after_results = self.run_benchmark_phase("after", apply_optimization=True)
        
        if not after_results:
            print("[ERROR] After phase failed. Cannot generate comparison.")
            return
        
        # Generate comparison
        print("\n" + "="*70)
        print("📊 BENCHMARK COMPARISON")
        print("="*70)
        
        comparison_table = self.generate_comparison_table(before_results, after_results)
        print(comparison_table)
        
        # Save comparison to file
        comparison_file = self.results_dir / "comparison.txt"
        with open(comparison_file, "w") as f:
            f.write(comparison_table)
        
        # Save combined results as JSON
        combined_results = {
            "before": before_results,
            "after": after_results,
            "timestamp": time.time()
        }
        combined_file = self.results_dir / "combined_results.json"
        with open(combined_file, "w") as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\n[SUCCESS] Benchmark complete!")
        print(f"  Comparison table: {comparison_file}")
        print(f"  Combined results: {combined_file}")
        print(f"  All results: {self.results_dir}")

def main():
    """Main entry point"""
    # Get project directory (assume script is in project root)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    runner = BenchmarkRunner(project_dir)
    runner.run_full_benchmark()

if __name__ == "__main__":
    main()

