# metrics_logger.py
import csv, os, statistics as stats
from typing import Dict, Any, List

class MetricsLogger:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.metrics_csv = os.path.join(self.out_dir, "metrics.csv")

    def write_latencies(self, latencies: List[float]):
        with open(self.metrics_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["latency_s"])
            for l in latencies:
                w.writerow([f"{l:.6f}"])

    def summary(self, latencies: List[float], throughput: float, successes: int, total: int) -> Dict[str, Any]:
        return {
            "n": len(latencies),
            "avg_latency_s": stats.mean(latencies) if latencies else None,
            "p50_s": stats.median(latencies) if latencies else None,
            "p95_s": (sorted(latencies)[int(0.95*len(latencies))-1] if latencies else None),
            "throughput_rps": throughput,
            "successes": successes,
            "total": total
        }
