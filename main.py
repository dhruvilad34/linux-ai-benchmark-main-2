# main.py
import argparse, yaml, asyncio, os, sys, json
import threading
import time
from pathlib import Path
import typing
from typing import Any, Dict, List, Optional

import torch
import psutil

from model_loader import ModelLoader
from agent_manager import AgentManager
from human_eval_runner import HumanEvalRunner
from gpu_monitor import GPUMonitor
from kernel_tracer import KernelTracer
from metrics_logger import MetricsLogger
from utils import set_verbose, log_info, log_warning, log_error, log_success
from perf_logger import PerfLogger
from react_agent_framework import ReActAgent, run_react_pipeline

# Limit Torch to a single CPU thread so the scheduler keeps our workers on pinned cores.
try:
    torch.set_num_threads(1)
except Exception:
    pass

# Weave and W&B imports
try:
    import weave
    import wandb
    WEAVE_AVAILABLE = True
    WANDB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Weave/W&B not available: {e}")
    WEAVE_AVAILABLE = False
    WANDB_AVAILABLE = False

# Resource monitoring
try:
    from resource_monitor import start_monitor
    RESOURCE_MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Resource monitor not available: {e}")
    RESOURCE_MONITOR_AVAILABLE = False

# Set environment variables to avoid warnings/errors
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def log_mem_snapshot(
    tag: str,
    gpu_id: Optional[int] = None,
    out_dir: str = "logs",
    log_to_wandb: bool = True,
) -> Dict[str, Any]:
    """
    Snapshot CPU RAM + GPU VRAM (Torch mem_get_info; NVML fallback if CUDA is unhealthy).
    Writes to: <out_dir>/memory_snapshots.jsonl

    Recommended tags:
      before_model_load_gpu{gid}
      after_model_load_gpu{gid}
      before_inference_gpu{gid}
      after_inference_gpu{gid}
    """
    snap: Dict[str, Any] = {"tag": tag, "ts": time.time()}
    snap_path = os.path.join(out_dir, "memory_snapshots.jsonl")

    # CPU RAM
    vm = psutil.virtual_memory()
    snap["cpu_ram_used_gb"] = round((vm.total - vm.available) / (1024**3), 3)
    snap["cpu_ram_total_gb"] = round(vm.total / (1024**3), 3)
    snap["cpu_ram_percent"] = float(vm.percent)

    # GPU VRAM
    if torch.cuda.is_available():
        if gpu_id is None:
            try:
                gpu_id = torch.cuda.current_device()
            except Exception:
                gpu_id = 0
        snap["gpu_id"] = int(gpu_id)

        # 1) Try torch mem_get_info (fast, best when CUDA is healthy)
        try:
            try:
                torch.cuda.synchronize(gpu_id)
            except Exception:
                pass

            free_b, total_b = torch.cuda.mem_get_info(gpu_id)
            snap["gpu_vram_free_gb"] = round(free_b / (1024**3), 3)
            snap["gpu_vram_total_gb"] = round(total_b / (1024**3), 3)
            snap["gpu_vram_used_gb"] = round((total_b - free_b) / (1024**3), 3)
            snap["gpu_mem_source"] = "torch"
        except Exception as e:
            snap["gpu_mem_get_info_error"] = str(e)

        # 2) allocated/reserved (may still work even if mem_get_info failed)
        try:
            snap["gpu_mem_allocated_gb"] = round(torch.cuda.memory_allocated(gpu_id) / (1024**3), 3)
            snap["gpu_mem_reserved_gb"] = round(torch.cuda.memory_reserved(gpu_id) / (1024**3), 3)
        except Exception:
            pass

        # 3) NVML fallback if CUDA crashed (device-side assert etc.)
        if "gpu_mem_get_info_error" in snap:
            try:
                import pynvml
                pynvml.nvmlInit()
                h = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_id))
                info = pynvml.nvmlDeviceGetMemoryInfo(h)
                snap["gpu_vram_total_gb"] = round(info.total / (1024**3), 3)
                snap["gpu_vram_used_gb"] = round(info.used / (1024**3), 3)
                snap["gpu_vram_free_gb"] = round(info.free / (1024**3), 3)
                snap["gpu_mem_source"] = "nvml"
            except Exception as e:
                snap["gpu_nvml_error"] = str(e)

    # Console output
    if "gpu_id" in snap and "gpu_vram_used_gb" in snap and "gpu_vram_total_gb" in snap:
        log_info(
            f"📌 MEM [{tag}] | RAM {snap['cpu_ram_used_gb']}/{snap['cpu_ram_total_gb']} GB ({snap['cpu_ram_percent']}%) | "
            f"GPU{snap['gpu_id']} VRAM used {snap['gpu_vram_used_gb']}/{snap['gpu_vram_total_gb']} GB | "
            f"alloc {snap.get('gpu_mem_allocated_gb','NA')} GB | reserved {snap.get('gpu_mem_reserved_gb','NA')} GB "
            f"(src={snap.get('gpu_mem_source','NA')})"
        )
    else:
        log_info(
            f"📌 MEM [{tag}] | RAM {snap['cpu_ram_used_gb']}/{snap['cpu_ram_total_gb']} GB ({snap['cpu_ram_percent']}%) | GPU mem N/A"
        )

    # Write JSONL
    _append_jsonl(snap_path, snap)

    # Optional W&B logging with clear, simple metric names
    if log_to_wandb and WANDB_AVAILABLE:
        try:
            if wandb.run:
                # Calculate elapsed time in seconds (for X-axis)
                elapsed_sec = round(time.time() - wandb.run.start_time, 1) if hasattr(wandb.run, "start_time") else 0
                
                # Log clean metrics with descriptive names
                # Format: "Category/Metric_Name (Unit)"
                wandb.log({
                    # Time (X-axis for charts)
                    "Time/Elapsed_Seconds": elapsed_sec,
                    
                    # GPU Memory metrics (Y-axis: Gigabytes)
                    "GPU_Memory/VRAM_Used_GB": snap.get("gpu_vram_used_gb", 0),
                    "GPU_Memory/VRAM_Free_GB": snap.get("gpu_vram_free_gb", 0),
                    "GPU_Memory/VRAM_Total_GB": snap.get("gpu_vram_total_gb", 0),
                    "GPU_Memory/PyTorch_Allocated_GB": snap.get("gpu_mem_allocated_gb", 0),
                    "GPU_Memory/PyTorch_Reserved_GB": snap.get("gpu_mem_reserved_gb", 0),
                    
                    # CPU Memory metrics
                    "CPU_Memory/RAM_Used_GB": snap.get("cpu_ram_used_gb", 0),
                    "CPU_Memory/RAM_Total_GB": snap.get("cpu_ram_total_gb", 0),
                    "CPU_Memory/RAM_Percent": snap.get("cpu_ram_percent", 0),
                    
                    # Metadata
                    "Metadata/GPU_ID": snap.get("gpu_id", -1),
                    "Metadata/Stage": tag,
                })
        except Exception:
            pass

    return snap


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_process_affinity(core_hint: Optional[int] = None) -> None:
    """
    Pin this process to a small CPU set. Keeping workers on the same cores improves cache locality.
    """
    try:
        proc = psutil.Process(os.getpid())
        available = proc.cpu_affinity()
    except AttributeError:
        return
    except psutil.AccessDenied:
        log_warning("⚠️  Unable to set cpu_affinity (permission denied).")
        return

    if not available:
        available = list(range(psutil.cpu_count(logical=True) or 1))

    if core_hint is not None:
        target_count = max(1, min(core_hint, len(available)))
    else:
        target_count = max(1, min(4, len(available)))

    target_cores = available[:target_count]

    try:
        current_affinity = proc.cpu_affinity()
        if sorted(current_affinity) == sorted(target_cores):
            return
        proc.cpu_affinity(target_cores)
        log_info(f"🎯 Pinned process to CPU cores {target_cores} to reduce scheduler migrations.")
    except psutil.Error as e:
        log_warning(f"⚠️  Could not update cpu_affinity: {e}")


def start_perf_logger(log_dir: str) -> Optional[PerfLogger]:
    log_path = os.path.join(log_dir, "perf_log.jsonl")
    try:
        perf_logger = PerfLogger(log_path, interval=2.0)
        perf_logger.start()
        log_info(f"📝 Perf logger capturing CPU/GPU stats every 2s at {log_path}")
        return perf_logger
    except Exception as e:
        log_warning(f"⚠️  Perf logger disabled: {e}")
        return None


def _sanitize_completion_for_humaneval(text: str) -> str:
    """
    If the model output contains runtime errors / CUDA errors, return a safe stub.
    HumanEval expects python code, not logs.
    """
    if not text:
        return "    pass"
    bad_markers = [
        "CUDA error",
        "device-side assert triggered",
        "gpu_mem_get_info_error",
        "Traceback (most recent call last)",
        "[ERROR]",
        "Assertion `probability tensor contains",
    ]
    for m in bad_markers:
        if m in text:
            return "    pass"
    # Ensure indentation for function body
    completion = text.strip()
    if completion and not completion.startswith("    "):
        lines = completion.split("\n")
        completion = "\n".join(("    " + ln.lstrip()) if ln.strip() else ln for ln in lines)
    if len(completion.strip()) < 5:
        return "    pass"
    return completion


async def main():
    log_info("\n" + "=" * 60)
    log_info("🖥️  SYSTEM RESOURCES SUMMARY")
    log_info("=" * 60)

    # GPU detection
    log_info("🟢 GPU DETECTION:")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        log_info(f"   Detected {n_gpus} GPU(s):")
        for i in range(n_gpus):
            try:
                name = torch.cuda.get_device_name(i)
                free_b, total_b = torch.cuda.mem_get_info(i)
                free_mb = free_b / (1024**2)
                total_mb = total_b / (1024**2)
                used_mb = total_mb - free_mb
                log_info(f"   GPU {i}: {name}")
                log_info(f"      VRAM: {free_mb:.1f}MB free / {total_mb:.1f}MB total ({used_mb:.1f}MB used)")
            except Exception as e:
                log_info(f"   GPU {i}: Could not get info ({e})")
    else:
        log_info("   CUDA not available")

    log_info(f"🧠 CPU cores: {psutil.cpu_count(logical=True)} (logical), {psutil.cpu_count(logical=False)} (physical)")
    mem = psutil.virtual_memory()
    log_info(f"💾 Total RAM: {round(mem.total / (1024**3), 2)} GB")
    log_info(f"💾 Available RAM: {round(mem.available / (1024**3), 2)} GB ({mem.percent}% used)")

    # Init Weave (optional)
    global WEAVE_AVAILABLE
    if WEAVE_AVAILABLE:
        try:
            if os.environ.get("DISABLE_WEAVE", "false").lower() == "true":
                log_warning("⚠️  Weave disabled via DISABLE_WEAVE")
                WEAVE_AVAILABLE = False
            else:
                weave.init("linux-ai-acceleration")
                log_success("✅ Weave initialized: linux-ai-acceleration")
        except Exception as e:
            log_warning(f"⚠️  Weave init failed: {e}")
            WEAVE_AVAILABLE = False

    log_info("=" * 60 + "\n")

    # Parse args
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--num_tasks", type=int, default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.num_tasks is not None:
        cfg["humaneval_limit"] = args.num_tasks
    if args.benchmark:
        cfg["benchmark"] = args.benchmark

    out_dir = cfg.get("output_dir", "logs")
    os.makedirs(out_dir, exist_ok=True)

    # Pin CPU
    core_hint_cfg = cfg.get("runtime_cpu_cores")
    apply_process_affinity(core_hint_cfg if isinstance(core_hint_cfg, int) else None)

    # Verbose
    verbose = cfg.get("verbose", False)
    set_verbose(verbose)

    # Make sure batching not too small
    if cfg.get("batch_size", 0) < 4:
        cfg["batch_size"] = 4
        log_info("🧺 batch_size bumped to 4 to keep GPU kernels batched efficiently.")

    # W&B init (fix local import bug + define metric step)
    wandb_initialized = False
    num_agents = int(cfg.get("num_agents", 0))
    num_tasks = int(cfg.get("humaneval_limit", 0) or 0)

    if WANDB_AVAILABLE:
        try:
            run_name = f"{num_agents}agents-{num_tasks}tasks"

            os.environ["WANDB_MODE"] = "online"
            if "WANDB_DIR" not in os.environ:
                os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")

            # tensorboard sync safe
            try:
                import tensorboard  # noqa: F401
                sync_tb = True
            except ImportError:
                sync_tb = False
                log_warning("⚠️  TensorBoard not available, disabling tensorboard sync")

            wandb.init(
                project="linux-ai-benchmark-main-2",
                entity=os.environ.get("WANDB_ENTITY", "lad71-university-of-windsor"),
                name=run_name,
                sync_tensorboard=sync_tb,
                mode="online",
                tags=[f"{num_agents}agents", f"{num_tasks}tasks", "resource-monitoring", "memory-snapshots"],
                config={
                    "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cpu_cores": psutil.cpu_count(logical=True),
                    "total_ram_gb": round(mem.total / (1024**3), 2),
                    "num_agents": num_agents,
                    "num_tasks": num_tasks,
                    "humaneval_limit": num_tasks,
                    "benchmark": cfg.get("benchmark", "humaneval"),
                    "model_id": cfg.get("model_id", ""),
                    "max_new_tokens": cfg.get("max_new_tokens", 512),
                    "temperature": cfg.get("temperature", 0.25),
                    "batch_size": cfg.get("batch_size", 4),
                    "use_data_parallelism": bool(cfg.get("use_data_parallelism", False)),
                },
                notes=f"Run with {num_agents} agents and {num_tasks} tasks. Memory snapshots enabled.",
            )
            wandb_initialized = True
            if wandb.run:
                log_success(f"✅ W&B initialized: {wandb.run.project}")
                log_info(f"   View run: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")

                # Define metrics with clear X-axis (Time) for W&B charts
                try:
                    # Set Time/Elapsed_Seconds as X-axis for all charts
                    wandb.define_metric("Time/Elapsed_Seconds")
                    
                    # GPU Memory charts: X-axis = Time, Y-axis = GB
                    wandb.define_metric("GPU_Memory/*", step_metric="Time/Elapsed_Seconds")
                    wandb.define_metric("GPU_Memory/VRAM_Used_GB", summary="max")
                    wandb.define_metric("GPU_Memory/PyTorch_Allocated_GB", summary="max")
                    
                    # CPU Memory charts: X-axis = Time, Y-axis = GB or %
                    wandb.define_metric("CPU_Memory/*", step_metric="Time/Elapsed_Seconds")
                    
                    # Resource monitor charts: X-axis = Time, Y-axis = %
                    wandb.define_metric("Resource/Elapsed_Seconds")
                    wandb.define_metric("Resource/*", step_metric="Resource/Elapsed_Seconds")
                    
                except Exception:
                    pass
        except Exception as e:
            log_warning(f"⚠️  W&B initialization failed: {e}")
            wandb_initialized = False

    # Resource monitor
    resource_monitor = None
    perf_logger: Optional[PerfLogger] = None
    if RESOURCE_MONITOR_AVAILABLE:
        try:
            resource_monitor = start_monitor(interval=5)
            log_success("✅ Resource monitor started (interval: 5s)")
        except Exception as e:
            log_warning(f"⚠️  Resource monitor failed to start: {e}")

    # Perf logger
    perf_logger = start_perf_logger(out_dir)

    # Model load mode
    use_data_parallelism = bool(cfg.get("use_data_parallelism", False))

    # IMPORTANT: log before-model-load for ALL GPUs
    if torch.cuda.is_available():
        for gid in range(torch.cuda.device_count()):
            log_mem_snapshot(f"before_model_load_gpu{gid}", gpu_id=gid, out_dir=out_dir)
    else:
        log_mem_snapshot("before_model_load_cpu_only", gpu_id=None, out_dir=out_dir)

    # Load model(s)
    models = None
    model = None

    if use_data_parallelism and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        log_success(f"🚀 DATA PARALLELISM MODE: Loading model on {n_gpus} GPUs separately")

        # tokenizer once
        model_id = cfg.get("model_id")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # Ensure pad token exists (helps attention_mask + generate stability)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        log_success("✅ Tokenizer loaded (shared across all GPUs)")

        loader = ModelLoader(cfg)
        models = {}

        for gpu_id in range(n_gpus):
            log_info(f"\n{'='*60}")
            log_info(f"🔄 Loading model on GPU {gpu_id}...")
            log_info(f"{'='*60}")

            torch.cuda.set_device(gpu_id)

            _, m = loader.load(gpu_id=gpu_id)

            # Verify model params on correct device
            target_device = torch.device(f"cuda:{gpu_id}")
            ok = True
            try:
                for p in m.parameters():
                    if p.device != target_device:
                        ok = False
                        break
                if not ok:
                    m = m.to(target_device)
                log_success(f"   ✅ Model params verified on {target_device}")
            except Exception as e:
                log_warning(f"   ⚠️  Could not verify parameters: {e}")

            models[gpu_id] = m
            log_mem_snapshot(f"after_model_load_gpu{gpu_id}", gpu_id=gpu_id, out_dir=out_dir)

        log_success(f"\n✅ Loaded {n_gpus} model instances (one per GPU)")
    else:
        tokenizer, model = ModelLoader(cfg).load()
        # Ensure pad token exists in sharded mode too
        try:
            if getattr(tokenizer, "pad_token_id", None) is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    # Prepare tasks
    bench = cfg.get("benchmark", "humaneval")
    if bench != "humaneval":
        log_warning(f"[WARN] Only humaneval stub wired; got {bench}. Using humaneval.")
    runner = HumanEvalRunner(cfg)
    tasks = runner.load_tasks()
    prompt_data_list = runner.extract_prompts(tasks)

    react_full = bool(cfg.get("react_full_evaluation", False))
    react_log_base = Path(cfg.get("react_log_path", "react_log.jsonl"))

    # Monitors
    gpu_csv = os.path.join(out_dir, "gpu_trace.csv")
    gpu_mon = GPUMonitor(gpu_csv, interval=cfg.get("gpu_monitor_interval", 1))
    logger = MetricsLogger(out_dir)

    # CROSS-EVALUATION
    print("\n" + "=" * 60)
    print("Running CROSS-EVALUATION BENCHMARK")
    print("=" * 60)

    num_agents = int(cfg.get("num_agents", 50))
    num_tasks = len(prompt_data_list)

    baseline_cfg = cfg.copy()
    baseline_cfg["num_samples"] = 1
    baseline_cfg["enable_reflection"] = False

    # Create MxN pairs
    cross_eval_pairs = []
    for agent_id in range(num_agents):
        for task_idx, prompt_data in enumerate(prompt_data_list):
            cross_eval_pairs.append({"agent_id": agent_id, "task_idx": task_idx, "prompt_data": prompt_data.copy()})

    gpu_mon.start()
    all_results = {}

    try:
        batch_size = int(baseline_cfg.get("batch_size", 8))

        if use_data_parallelism and models is not None:
            n_gpus = len(models)
            log_success(f"\n🚀 DATA PARALLELISM: Distributing tasks across {n_gpus} GPUs (round-robin)")

            gpu_task_batches = {i: [] for i in range(n_gpus)}
            for idx, pair in enumerate(cross_eval_pairs):
                gid = idx % n_gpus
                gpu_task_batches[gid].append((pair["agent_id"], pair["prompt_data"]))

            async def process_gpu(gpu_id: int, task_batch: list):
                torch.cuda.set_device(gpu_id)
                gpu_model = models[gpu_id]

                # Log before inference snapshot for this GPU
                log_mem_snapshot(f"before_inference_gpu{gpu_id}", gpu_id=gpu_id, out_dir=out_dir)

                gpu_results = {}
                agent_task_map: Dict[int, List[Dict[str, Any]]] = {}
                for agent_id, task_data in task_batch:
                    agent_task_map.setdefault(agent_id, []).append(task_data)

                for agent_id, agent_task_list in agent_task_map.items():
                    manager = AgentManager(tokenizer, gpu_model, baseline_cfg, agent_id=agent_id)

                    try:
                        result = await manager.run_tasks(agent_task_list, use_multi_sample=False, batch_size=batch_size)
                    except Exception as e:
                        log_warning(f"⚠️  Agent {agent_id} failed on GPU {gpu_id}: {e}")
                        result = {
                            "agent_id": agent_id,
                            "latencies": [],
                            "outputs": ["    pass"] * len(agent_task_list),
                            "task_ids": [t.get("task_id", "") for t in agent_task_list],
                            "successes": 0,
                            "total": len(agent_task_list),
                            "runtime_s": 0.0,
                        }

                    gpu_results[agent_id] = result

                # Log after inference snapshot for this GPU (may use NVML if CUDA is unhealthy)
                log_mem_snapshot(f"after_inference_gpu{gpu_id}", gpu_id=gpu_id, out_dir=out_dir)

                return gpu_id, gpu_results

            gpu_results_list = await asyncio.gather(*[
                process_gpu(gpu_id, gpu_task_batches[gpu_id]) for gpu_id in range(n_gpus)
            ])

            for gpu_id, gpu_results in gpu_results_list:
                for agent_id, result in gpu_results.items():
                    all_results[agent_id] = result

        else:
            async def process_agent(agent_id: int):
                agent_tasks = [pair["prompt_data"] for pair in cross_eval_pairs if pair["agent_id"] == agent_id]
                manager = AgentManager(tokenizer, model, baseline_cfg, agent_id=agent_id)
                return agent_id, await manager.run_tasks(agent_tasks, use_multi_sample=False, batch_size=batch_size)

            if num_agents > 1:
                agent_results = await asyncio.gather(*[process_agent(aid) for aid in range(num_agents)])
                for aid, res in agent_results:
                    all_results[aid] = res
            else:
                aid, res = await process_agent(0)
                all_results[aid] = res

    finally:
        gpu_mon.stop()

    # PROCESS RESULTS
    print("\n" + "=" * 60)
    print("PROCESSING CROSS-EVALUATION RESULTS")
    print("=" * 60)

    agent_summaries = {}

    for agent_id, result in all_results.items():
        outputs = result.get("outputs", [])
        task_ids = result.get("task_ids", [])
        latencies = result.get("latencies", [])

        agent_results = []
        for i, output in enumerate(outputs):
            if i < len(task_ids) and task_ids[i]:
                task_id = task_ids[i]
            elif i < len(prompt_data_list):
                task_id = prompt_data_list[i].get("task_id", f"HumanEval/{i}")
            else:
                task_id = f"HumanEval/{i}"

            completion = _sanitize_completion_for_humaneval(output if isinstance(output, str) else "")
            agent_results.append({"task_id": task_id, "completion": completion})

        # Save per-agent results
        agent_jsonl = os.path.join(out_dir, f"agent_{agent_id}_results.jsonl")
        from human_eval.data import write_jsonl
        write_jsonl(agent_jsonl, agent_results)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_time = float(result.get("runtime_s", 0.0) or 0.0)
        throughput = len(outputs) / max(total_time, 0.001)
        successes = int(result.get("successes", 0) or 0)

        agent_summaries[agent_id] = {
            "agent_id": agent_id,
            "total_tasks": len(outputs),
            "successes": successes,
            "avg_latency_s": avg_latency,
            "total_time_s": total_time,
            "throughput_rps": throughput,
            "results_file": agent_jsonl,
        }

        print(f"✅ Agent {agent_id}: {len(agent_results)} results saved to {agent_jsonl}")

    # EVALUATE
    print("\n" + "=" * 60)
    print("EVALUATING AGENT RESULTS")
    print("=" * 60)

    for agent_id, summary in agent_summaries.items():
        agent_jsonl = summary["results_file"]
        print(f"\n📊 Evaluating Agent {agent_id}...")
        try:
            eval_result = runner.evaluate_outputs(
                agent_jsonl,
                k=[1, 10, 100],
                n_workers=cfg.get("eval_workers", 4),
                timeout=cfg.get("eval_timeout", 10.0),
            )
            agent_summaries[agent_id]["pass@1"] = eval_result.get("pass@1", 0.0)
            agent_summaries[agent_id]["pass@10"] = eval_result.get("pass@10", 0.0)
            agent_summaries[agent_id]["pass@100"] = eval_result.get("pass@100", 0.0)

            agent_eval_file = os.path.join(out_dir, f"agent_{agent_id}_evaluation.json")
            with open(agent_eval_file, "w") as f:
                json.dump(eval_result, f, indent=2)

            print(f"   Agent {agent_id} Pass@1: {eval_result.get('pass@1', 0.0):.2%}")
        except Exception as e:
            print(f"   ⚠️  Agent {agent_id} evaluation failed: {e}")
            agent_summaries[agent_id]["pass@1"] = 0.0
            agent_summaries[agent_id]["pass@10"] = 0.0
            agent_summaries[agent_id]["pass@100"] = 0.0

    summary_file = os.path.join(out_dir, "agent_summary.json")
    with open(summary_file, "w") as f:
        json.dump(
            {
                "num_agents": num_agents,
                "num_tasks": num_tasks,
                "total_evaluations": num_agents * num_tasks,
                "agents": list(agent_summaries.values()),
            },
            f,
            indent=2,
        )

    overall_pass1 = sum(s.get("pass@1", 0.0) for s in agent_summaries.values()) / len(agent_summaries) if agent_summaries else 0.0
    overall_latency = sum(s.get("avg_latency_s", 0.0) for s in agent_summaries.values()) / len(agent_summaries) if agent_summaries else 0.0
    overall_throughput = sum(s.get("throughput_rps", 0.0) for s in agent_summaries.values()) / len(agent_summaries) if agent_summaries else 0.0

    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(
            {
                "pass@1": overall_pass1,
                "total": num_tasks,
                "num_agents": num_agents,
                "avg_latency_s": overall_latency,
                "avg_throughput_rps": overall_throughput,
            },
            f,
            indent=2,
        )

    if WANDB_AVAILABLE and wandb.run:
        try:
            wandb.log(
                {
                    "overall_pass@1": overall_pass1,
                    "overall_avg_latency_s": overall_latency,
                    "overall_avg_throughput_rps": overall_throughput,
                    "num_agents": num_agents,
                    "num_tasks": num_tasks,
                    "total_evaluations": num_agents * num_tasks,
                }
            )
        except Exception as e:
            log_warning(f"⚠️  W&B logging failed: {e}")

    if resource_monitor:
        try:
            resource_monitor.stop()
            log_info("🛑 Resource monitor stopped")
        except Exception:
            pass

    if perf_logger:
        try:
            perf_logger.stop()
        except Exception:
            pass

    print("\n=== Files Saved ===")
    print(f"✅ Agent summary: {summary_file}")
    print(f"✅ Score: {os.path.join(out_dir, 'score.json')}")
    print(f"✅ Memory snapshots: {os.path.join(out_dir, 'memory_snapshots.jsonl')}")
    print(f"✅ GPU trace: {os.path.join(out_dir, 'gpu_trace.csv')}")


if __name__ == "__main__":
    asyncio.run(main())
