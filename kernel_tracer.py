# kernel_tracer.py
import subprocess, os

class KernelTracer:
    def __init__(self, out_txt: str):
        self.out_txt = out_txt

    def run_perf(self, cmd: list):
        os.makedirs(os.path.dirname(self.out_txt), exist_ok=True)
        with open(self.out_txt, "w") as f:
            # context-switches & task-clock are key for this project
            perf = ["perf", "stat", "-e", "context-switches,cpu-migrations,task-clock,cycles,instructions"]
            subprocess.run(perf + cmd, stdout=f, stderr=f, check=False)
