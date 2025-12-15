# gpu_monitor.py
import subprocess, threading, time, os

class GPUMonitor:
    def __init__(self, out_csv: str, interval: float = 1.0):
        self.out_csv = out_csv
        self.interval = interval
        self.proc = None
        self.thread = None
        self.stop_flag = False

    def start(self):
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        # dmon includes util, pwr, mem, etc.
        self.proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "pu", "-d", str(int(self.interval)), "-o", "DT"],
            stdout=open(self.out_csv, "w"),
            stderr=subprocess.STDOUT
        )

    def stop(self):
        try:
            if self.proc:
                self.proc.terminate()
        except Exception:
            pass
