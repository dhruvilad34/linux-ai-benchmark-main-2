"""
perf_logger.py

Background performance logger that records lightweight CPU/GPU counters every
two seconds.  This runs without root privileges and gives a poor-man's kernel
trace by emitting fine-grained stats to perf_log.jsonl.
"""
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

import psutil


class PerfLogger:
    """
    Background logger for latency-sensitive metrics.  Keeping this information
    in-process helps correlate kernel scheduling behaviour with model activity
    without expensive external tooling.
    """

    def __init__(self, log_path: str, interval: float = 2.0):
        self.log_path = log_path
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Prepare process reference once so we avoid repeated /proc scans.
        self._proc = psutil.Process(os.getpid())

        # Truncate existing file so each run starts fresh.
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            # Fall back to writing alongside CWD if permissions are tight.
            fallback = os.path.join(os.getcwd(), "perf_log.jsonl")
            self.log_path = fallback
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("")

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            timestamp = time.time()
            entry = {
                "ts": timestamp,
            }
            try:
                cpu_times = self._proc.cpu_times()
                ctx = self._proc.num_ctx_switches()
                mem_info = self._proc.memory_info()

                entry.update(
                    {
                        # Lower cpu_user/cpu_sys numbers mean fewer kernel context switches.
                        "cpu_user": cpu_times.user,
                        "cpu_sys": cpu_times.system,
                        "ctx_switches_vol": ctx.voluntary,
                        "ctx_switches_invol": ctx.involuntary,
                        "mem_used_mb": round(mem_info.rss / (1024**2), 2),
                    }
                )
            except psutil.Error:
                pass

            gpu_stats = self._query_gpu_util()
            if gpu_stats is not None:
                entry["gpu"] = gpu_stats

            try:
                with open(self.log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(entry) + "\n")
            except OSError:
                # If we cannot write, silently drop the sample.
                pass

            # Sleep using Event.wait so we can exit promptly.
            self._stop_event.wait(self.interval)

    @staticmethod
    def _query_gpu_util() -> Optional[List[Dict[str, Any]]]:
        """
        Capture GPU utilisation with nvidia-smi if present.  This mimics
        `watch -n 1 nvidia-smi` without spawning a heavy external watcher.
        """
        if shutil.which("nvidia-smi") is None:
            return None

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,utilization.memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None

            stats: List[Dict[str, Any]] = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    stats.append(
                        {
                            "gpu_id": int(parts[0]),
                            "util_percent": float(parts[1]),
                            "mem_util_percent": float(parts[2]),
                        }
                    )
                except ValueError:
                    continue
            return stats if stats else None
        except (subprocess.SubprocessError, OSError):
            return None


