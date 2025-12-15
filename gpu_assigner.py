# gpu_assigner.py
"""
Dynamic GPU assignment module for balanced, efficient GPU utilization.
Assigns agents to the least-loaded GPU based on real-time memory usage.
"""
import torch
import threading
from typing import Optional, Dict, Any
from functools import lru_cache

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None


class GPUAssigner:
    """
    Manages dynamic GPU assignment for agents based on real-time load.
    Thread-safe implementation for concurrent agent assignment.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.assignment_counts = {}  # Track assignments per GPU for debugging
    
    def get_least_loaded_gpu(self) -> str:
        """
        Get the GPU with the most free memory (least loaded).
        
        Returns:
            Device string like "cuda:0" or "cpu" if no GPUs available
        """
        if not torch.cuda.is_available():
            return "cpu"
        
        # Try GPUtil first (more detailed)
        if GPU_AVAILABLE and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Find GPU with most free memory
                    best_gpu = min(gpus, key=lambda x: x.memoryUsed)
                    device = f"cuda:{best_gpu.id}"
                    with self.lock:
                        self.assignment_counts[device] = self.assignment_counts.get(device, 0) + 1
                    return device
            except Exception as e:
                print(f"⚠️  GPUtil error in get_least_loaded_gpu: {e}")
        
        # Fallback to torch.cuda
        try:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                return "cpu"
            
            # Get free memory for each GPU
            gpu_memory = []
            for i in range(n_gpus):
                try:
                    mem_info = torch.cuda.mem_get_info(i)
                    free_mem = mem_info[0]
                    gpu_memory.append((i, free_mem))
                except Exception:
                    continue
            
            if gpu_memory:
                # Select GPU with most free memory
                best_gpu_idx, _ = max(gpu_memory, key=lambda x: x[1])
                device = f"cuda:{best_gpu_idx}"
                with self.lock:
                    self.assignment_counts[device] = self.assignment_counts.get(device, 0) + 1
                return device
        except Exception as e:
            print(f"⚠️  torch.cuda error in get_least_loaded_gpu: {e}")
        
        return "cpu"
    
    def assign_gpu(self, agent_id: Optional[int] = None) -> str:
        """
        Assign a GPU to an agent (thread-safe).
        
        Args:
            agent_id: Optional agent identifier for logging
            
        Returns:
            Device string like "cuda:0" or "cpu"
        """
        device = self.get_least_loaded_gpu()
        if agent_id is not None:
            print(f"🧠 Agent {agent_id} assigned to {device}")
        return device
    
    def get_assignment_stats(self) -> Dict[str, int]:
        """Get statistics about GPU assignments."""
        with self.lock:
            return self.assignment_counts.copy()
    
    def check_gpu_availability(self, threshold: float = 0.9) -> bool:
        """
        Check if any GPU has available memory below threshold.
        
        Args:
            threshold: Memory utilization threshold (0.9 = 90%)
            
        Returns:
            True if at least one GPU has memory below threshold, False otherwise
        """
        if not torch.cuda.is_available():
            return False
        
        # Try GPUtil first
        if GPU_AVAILABLE and GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    # Check if any GPU has memory utilization below threshold
                    for gpu in gpus:
                        mem_util = gpu.memoryUsed / gpu.memoryTotal if gpu.memoryTotal > 0 else 1.0
                        if mem_util < threshold:
                            return True
                    return False  # All GPUs are above threshold
            except Exception:
                pass
        
        # Fallback to torch.cuda
        try:
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                try:
                    mem_info = torch.cuda.mem_get_info(i)
                    free_mem = mem_info[0]
                    total_mem = mem_info[1]
                    if total_mem > 0:
                        mem_util = 1.0 - (free_mem / total_mem)
                        if mem_util < threshold:
                            return True
                except Exception:
                    continue
        except Exception:
            pass
        
        return False  # Assume unavailable if we can't check


# Global instance
_gpu_assigner = GPUAssigner()


def get_least_loaded_gpu() -> str:
    """Convenience function to get least-loaded GPU."""
    return _gpu_assigner.get_least_loaded_gpu()


def assign_gpu(agent_id: Optional[int] = None) -> str:
    """Convenience function to assign GPU to agent."""
    return _gpu_assigner.assign_gpu(agent_id)


def check_gpu_availability(threshold: float = 0.9) -> bool:
    """Convenience function to check GPU availability."""
    return _gpu_assigner.check_gpu_availability(threshold)


def get_assignment_stats() -> Dict[str, int]:
    """Get GPU assignment statistics."""
    return _gpu_assigner.get_assignment_stats()












