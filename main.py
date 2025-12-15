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

# Limit Torch to a single CPU thread so the scheduler keeps our workers on
# the pinned cores.  This cuts down on CPU oversubscription when the kernel
# dispatches background workers.
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

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_process_affinity(core_hint: Optional[int] = None) -> None:
    """
    Pin this process to a small CPU set.  Keeping workers on the same cores
    improves cache locality and cuts kernel migration overhead.
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
    """
    Launch the JSONL perf logger so we can correlate workloads with kernel
    behaviour without root access.
    """
    log_path = os.path.join(log_dir, "perf_log.jsonl")
    try:
        perf_logger = PerfLogger(log_path, interval=2.0)
        perf_logger.start()
        log_info(f"📝 Perf logger capturing CPU/GPU stats every 2s at {log_path}")
        log_info("   (For live views you can run: watch -n 1 nvidia-smi | vmstat 1 | mpstat -P ALL 1)")
        return perf_logger
    except Exception as e:
        log_warning(f"⚠️  Perf logger disabled: {e}")
        return None

async def main():
    # Display system summary on startup (only if verbose)
    log_info("\n" + "="*60)
    log_info("🖥️  SYSTEM RESOURCES SUMMARY")
    log_info("="*60)
    
    # Enhanced GPU detection with detailed specs
    log_info("🟢 GPU DETECTION:")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        log_info(f"   Detected {n_gpus} GPU(s):")
        
        # Try GPUtil for detailed info
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    free_mb = gpu.memoryFree
                    total_mb = gpu.memoryTotal
                    used_mb = gpu.memoryUsed
                    log_info(f"   GPU {gpu.id}: {gpu.name}")
                    log_info(f"      VRAM: {free_mb:.1f}MB free / {total_mb:.1f}MB total ({used_mb:.1f}MB used, {gpu.load*100:.1f}% load)")
                    if hasattr(gpu, 'temperature') and gpu.temperature:
                        log_info(f"      Temperature: {gpu.temperature}°C")
        except ImportError:
            # Fallback to torch.cuda
            for i in range(n_gpus):
                try:
                    name = torch.cuda.get_device_name(i)
                    mem_info = torch.cuda.mem_get_info(i)
                    free_mb = mem_info[0] / (1024**2)
                    total_mb = mem_info[1] / (1024**2)
                    used_mb = total_mb - free_mb
                    log_info(f"   GPU {i}: {name}")
                    log_info(f"      VRAM: {free_mb:.1f}MB free / {total_mb:.1f}MB total ({used_mb:.1f}MB used)")
                except Exception as e:
                    log_info(f"   GPU {i}: Could not get info ({e})")
        except Exception as e:
            log_warning(f"   ⚠️  Error getting GPU details: {e}")
    else:
        log_info("   CUDA not available")
    
    # CPU info
    log_info(f"🧠 CPU cores: {psutil.cpu_count(logical=True)} (logical), {psutil.cpu_count(logical=False)} (physical)")
    
    # RAM info
    mem = psutil.virtual_memory()
    log_info(f"💾 Total RAM: {round(mem.total / (1024**3), 2)} GB")
    log_info(f"💾 Available RAM: {round(mem.available / (1024**3), 2)} GB ({mem.percent}% used)")
    
    # Initialize Weave (optional - disable if causing errors)
    global WEAVE_AVAILABLE
    if WEAVE_AVAILABLE:
        try:
            # Disable Weave if it's causing HTTP 500 errors
            # Set environment variable to disable Weave
            if os.environ.get("DISABLE_WEAVE", "false").lower() == "true":
                log_warning("⚠️  Weave disabled via DISABLE_WEAVE environment variable")
                WEAVE_AVAILABLE = False
            else:
                try:
                    weave.init("linux-ai-acceleration")
                    log_success("✅ Weave initialized: linux-ai-acceleration")
                except Exception as e:
                    log_warning(f"⚠️  Weave initialization failed: {e}")
                    # Disable Weave for this run if init fails
                    WEAVE_AVAILABLE = False
        except Exception as e:
            log_warning(f"⚠️  Weave initialization failed: {e}")
            WEAVE_AVAILABLE = False
    
    log_info("="*60 + "\n")
    
    # Parse arguments and load config first
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

    # Pin to a small CPU set *after* reading config so advanced users can
    # override the number of cores (runtime_cpu_cores).
    core_hint_cfg = cfg.get("runtime_cpu_cores")
    apply_process_affinity(core_hint_cfg if isinstance(core_hint_cfg, int) else None)

    # Encourage GPU batching so we amortise kernel launch overhead.  Users can
    # still override this via config, but we keep a sensible minimum.
    if cfg.get("batch_size", 0) < 4:
        cfg["batch_size"] = 4
        log_info("🧺 batch_size bumped to 4 to keep GPU kernels batched efficiently.")
    
    # Set verbosity from config
    verbose = cfg.get("verbose", False)
    set_verbose(verbose)
    
    # Now initialize W&B with agent and task information
    num_agents = cfg.get("num_agents", 0)
    num_tasks = cfg.get("humaneval_limit", 0)
    
    wandb_initialized = False
    if WANDB_AVAILABLE:
        try:
            # Create descriptive run name with agent and task info
            run_name = f"{num_agents}agents-{num_tasks}tasks"
            
            # Force online mode - check environment variable
            # Force online mode - set environment variable first
            os.environ["WANDB_MODE"] = "online"
            wandb_mode = os.environ.get("WANDB_MODE", "online")
            if wandb_mode != "online":
                print(f"⚠️  WANDB_MODE is set to '{wandb_mode}', forcing to 'online'")
                os.environ["WANDB_MODE"] = "online"
            
            # Ensure W&B directory is set (for syncing)
            if "WANDB_DIR" not in os.environ:
                os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "wandb")
            
            log_info(f"🔧 W&B Configuration:")
            log_info(f"   Mode: {os.environ.get('WANDB_MODE', 'not set')}")
            log_info(f"   Project: linux-ai-acceleration")
            log_info(f"   Entity: kishorelin-tad001-university-of-windsor")
            log_info(f"   Run name: {run_name}")
            
            # Check if tensorboard is available before enabling sync
            try:
                import tensorboard
                sync_tb = True
            except ImportError:
                sync_tb = False
                log_warning("⚠️  TensorBoard not available, disabling tensorboard sync")
            
            wandb.init(
                project="linux-ai-acceleration",
                entity="kishorelin-tad001-university-of-windsor",
                name=run_name,  # Descriptive run name
                sync_tensorboard=sync_tb,  # Only enable if tensorboard is installed
                mode="online",  # Force online mode
                settings=wandb.Settings(start_method="fork"),  # Avoid reinit deprecation warning
                tags=[f"{num_agents}agents", f"{num_tasks}tasks", "resource-monitoring"],  # Tags for filtering
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
                },
                notes=f"Run with {num_agents} agents and {num_tasks} tasks. Resource monitoring enabled."
            )
            wandb_initialized = True
            if wandb.run:
                log_success(f"✅ W&B initialized: {wandb.run.project}")
                log_info(f"   Run name: {run_name}")
                log_info(f"   Run ID: {wandb.run.id}")
                log_info(f"   Config: {num_agents} agents, {num_tasks} tasks")
                log_info(f"   View at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
            else:
                log_success("✅ W&B initialized: linux-ai-acceleration")
        except Exception as e:
            log_warning(f"⚠️  W&B initialization failed: {e}")
    
    # Start resource monitoring in background thread
    # Add a small delay to ensure wandb.init() completes first
    import time
    time.sleep(2)  # Increased delay to ensure proper initialization
    
    # Verify W&B is in online mode and properly initialized
    if WANDB_AVAILABLE and wandb and hasattr(wandb, 'run') and wandb.run:
        # Check mode using settings or environment variable (newer wandb versions)
        try:
            # Try to get mode from run settings or use environment variable
            wandb_mode = os.environ.get("WANDB_MODE", "online")
            if hasattr(wandb.run, 'settings'):
                # Newer wandb API
                run_mode = getattr(wandb.run.settings, 'mode', wandb_mode)
            elif hasattr(wandb.run, 'mode'):
                # Older wandb API
                run_mode = wandb.run.mode
            else:
                # Fallback to environment variable
                run_mode = wandb_mode
            
            if run_mode != "online" and wandb_mode == "online":
                log_warning("⚠️  WARNING: W&B may not be in online mode!")
                log_warning("   Metrics may not sync to cloud. Check WANDB_MODE environment variable.")
            else:
                log_success(f"✅ W&B initialized in ONLINE mode (run ID: {wandb.run.id})")
                log_info(f"   View run at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
        except Exception as e:
            # If we can't check mode, assume it's working and just log success
            log_success(f"✅ W&B initialized (run ID: {wandb.run.id})")
            log_info(f"   View run at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
    else:
        log_warning("⚠️  W&B run not initialized properly")
    
    resource_monitor = None
    perf_logger: Optional[PerfLogger] = None
    if RESOURCE_MONITOR_AVAILABLE:
        try:
            # Wait a moment for wandb to fully initialize
            import time
            if wandb_initialized:
                time.sleep(1)  # Give wandb time to set up wandb.run
                print(f"✅ W&B run ready: {wandb.run.id if wandb.run else 'Not ready'}")
            # Monitor every 5 seconds for better granularity
            resource_monitor = start_monitor(interval=5)
            print("✅ Resource monitor started")
        except Exception as e:
            print(f"⚠️  Resource monitor failed to start: {e}")

    out_dir = cfg.get("output_dir", "logs")
    os.makedirs(out_dir, exist_ok=True)

    # Start lightweight perf logger before we kick off GPU work so the log
    # captures warm-up spikes as well.
    perf_logger = start_perf_logger(out_dir)

    # Check if data parallelism is enabled
    use_data_parallelism = cfg.get("use_data_parallelism", False)
    
    # Load model(s)
    if use_data_parallelism and torch.cuda.is_available():
        # DATA PARALLELISM: Load model on each GPU separately
        n_gpus = torch.cuda.device_count()
        log_success(f"🚀 DATA PARALLELISM MODE: Loading model on {n_gpus} GPUs separately")
        
        # Load tokenizer once (shared) - tokenizer doesn't need GPU
        model_id = cfg.get("model_id")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        log_success("✅ Tokenizer loaded (shared across all GPUs)")
        
        # Load model on each GPU with explicit device context
        models = {}
        loader = ModelLoader(cfg)
        
        for gpu_id in range(n_gpus):
            log_info(f"\n{'='*60}")
            log_info(f"🔄 Loading model on GPU {gpu_id}...")
            log_info(f"{'='*60}")
            
            # Set CUDA device context before loading
            torch.cuda.set_device(gpu_id)
            log_info(f"   Set CUDA device to GPU {gpu_id}")
            
            # Load model on this specific GPU
            _, model = loader.load(gpu_id=gpu_id)
            
            # Verify ALL parameters are on the correct GPU
            target_device = torch.device(f"cuda:{gpu_id}")
            all_correct = True
            wrong_device_params = []
            
            try:
                for name, param in model.named_parameters():
                    if param.device != target_device:
                        all_correct = False
                        wrong_device_params.append((name, param.device))
                        # Move parameter to correct device
                        param.data = param.data.to(target_device)
                
                if wrong_device_params:
                    log_warning(f"   ⚠️  Found {len(wrong_device_params)} parameters on wrong device, moved to {target_device}")
                    # Verify all are now correct
                    all_correct = True
                    for name, param in model.named_parameters():
                        if param.device != target_device:
                            all_correct = False
                            break
                
                if all_correct:
                    # Count parameters on correct device
                    param_count = sum(1 for _ in model.parameters())
                    log_success(f"   ✅ Verified: All {param_count} parameters on GPU {gpu_id} ({target_device})")
                else:
                    log_error(f"   ❌ ERROR: Some parameters still not on GPU {gpu_id}!")
                    
            except Exception as e:
                log_error(f"   ❌ Could not verify model device: {e}")
            
            # Additional verification: check first and last parameters
            try:
                params_list = list(model.parameters())
                if params_list:
                    first_device = params_list[0].device
                    last_device = params_list[-1].device
                    if first_device == target_device and last_device == target_device:
                        log_success(f"   ✅ First and last parameters confirmed on {target_device}")
                    else:
                        log_warning(f"   ⚠️  First param on {first_device}, last on {last_device}, expected {target_device}")
            except Exception as e:
                log_warning(f"   ⚠️  Could not verify parameter devices: {e}")
            
            models[gpu_id] = model
            log_success(f"✅ Model {gpu_id} loaded and verified on GPU {gpu_id}")
        
        log_success(f"\n✅ Loaded {n_gpus} model instances (one per GPU)")
        model = None  # Will use models dict instead
    else:
        # MODEL SHARDING: Load single model sharded across GPUs
        tokenizer, model = ModelLoader(cfg).load()
        models = None

    # Prepare tasks
    bench = cfg.get("benchmark", "humaneval")
    if bench != "humaneval":
        print(f"[WARN] Only humaneval stub wired; got {bench}. Using humaneval.")
    runner = HumanEvalRunner(cfg)
    tasks = runner.load_tasks()
    prompt_data_list = runner.extract_prompts(tasks)  # Now returns formatted prompts

    react_full = bool(cfg.get("react_full_evaluation", False))
    react_log_base = Path(cfg.get("react_log_path", "react_log.jsonl"))

    # Optional ReAct smoke test before large-scale evaluation
    react_results = []
    if react_full:
        log_info("\n" + "=" * 60)
        log_info("🧠 ReAct FULL EVALUATION ENABLED")
        log_info("=" * 60)
        log_info("   Replacing baseline AgentManager run with ReAct pipeline for all tasks.")
    elif cfg.get("enable_react_pipeline", False):
        react_task_limit_cfg = int(cfg.get("react_task_limit", 0) or 0)
        if react_task_limit_cfg <= 0:
            react_task_limit = min(len(prompt_data_list), cfg.get("humaneval_limit", len(prompt_data_list)))
        else:
            react_task_limit = min(react_task_limit_cfg, len(prompt_data_list))

        if react_task_limit > 0 and prompt_data_list:
            log_info("\n" + "=" * 60)
            log_info("🧠 ReAct PIPELINE SMOKE TEST")
            log_info("=" * 60)
            log_info(f"   Tasks routed through ReAct: {react_task_limit}")

            # Choose model instance for the ReAct loop
            react_model = None
            if use_data_parallelism and models:
                torch.cuda.set_device(0)
                react_model = models.get(0)
                if react_model is None:
                    log_warning("⚠️  ReAct pipeline skipped: GPU 0 model missing.")
            else:
                react_model = model

            if react_model is not None:
                react_manager_cfg = cfg.copy()
                react_manager_cfg["num_agents"] = max(1, react_manager_cfg.get("num_agents", 1))

                react_manager = AgentManager(
                    tokenizer,
                    react_model,
                    react_manager_cfg,
                    agent_id=-1,
                )
                react_agent = ReActAgent(
                    react_manager,
                    log_path=cfg.get("react_log_path", "react_log.jsonl"),
                )

                react_tasks = prompt_data_list[:react_task_limit]
                react_results = await run_react_pipeline(react_tasks, react_agent)
                successes = sum(1 for r in react_results if r.get("success"))
                log_success(
                    f"✅ ReAct pipeline complete: {successes}/{len(react_results)} "
                    "tasks reported success. Logs -> "
                    f"{cfg.get('react_log_path', 'react_log.jsonl')}"
                )
            else:
                log_warning("⚠️  ReAct pipeline skipped due to missing model.")

    async def run_agent_with_react(agent_id: int, manager: AgentManager, agent_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a full ReAct evaluation for a single agent and package results in the
        same structure returned by AgentManager.run_tasks.
        """
        if not agent_tasks:
            return {
                "agent_id": agent_id,
                "latencies": [],
                "outputs": [],
                "task_ids": [],
                "successes": 0,
                "total": 0,
                "runtime_s": 0.0,
                "throughput_rps": 0.0,
                "max_concurrent_agents": manager.sem._value,
                "configured_agents": manager.sem._value,
                "total_processed": 0,
                "batch_size": 1,
                "num_batches": 0,
            }

        # Construct agent-specific log path to avoid contention across agents/GPUs.
        if react_log_base.suffix:
            log_path = react_log_base.with_name(f"{react_log_base.stem}_agent{agent_id}{react_log_base.suffix}")
        else:
            log_path = Path(f"{react_log_base}_agent{agent_id}.jsonl")

        react_agent = ReActAgent(manager, log_path=str(log_path))
        start_time = time.time()
        react_outputs = await run_react_pipeline(agent_tasks, react_agent)
        runtime = time.time() - start_time

        outputs: List[str] = []
        latencies: List[float] = []
        task_ids: List[str] = []
        successes = 0

        if len(react_outputs) != len(agent_tasks):
            log_warning(
                f"⚠️  ReAct results count mismatch for agent {agent_id}: "
                f"{len(react_outputs)} results vs {len(agent_tasks)} tasks."
            )

        for idx, task in enumerate(agent_tasks):
            task_id = task.get("task_id", f"agent{agent_id}/task{idx}")
            task_ids.append(task_id)

            if idx < len(react_outputs):
                result = react_outputs[idx]
            else:
                result = {}

            completion = result.get("completion", "") or ""
            outputs.append(completion)
            latencies.append(float(result.get("latency_s", 0.0)))
            if result.get("success"):
                successes += 1

        effective_runtime = runtime if runtime > 0 else sum(latencies)
        throughput = len(outputs) / max(effective_runtime, 0.001)

        return {
            "agent_id": agent_id,
            "latencies": latencies,
            "outputs": outputs,
            "task_ids": task_ids,
            "successes": successes,
            "total": len(agent_tasks),
            "runtime_s": effective_runtime,
            "throughput_rps": throughput,
            "max_concurrent_agents": manager.sem._value,
            "configured_agents": manager.sem._value,
            "total_processed": len(agent_tasks),
            "batch_size": 1,
            "num_batches": None,
        }

    async def run_agent_evaluation(agent_id: int, manager: AgentManager, agent_tasks: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        if react_full:
            return await run_agent_with_react(agent_id, manager, agent_tasks)
        return await manager.run_tasks(agent_tasks, use_multi_sample=False, batch_size=batch_size)

    # Monitors
    gpu_csv = os.path.join(out_dir, "gpu_trace.csv")
    perf_txt = os.path.join(out_dir, "perf_trace.txt")
    gpu_mon = GPUMonitor(gpu_csv, interval=cfg.get("gpu_monitor_interval", 1))
    logger = MetricsLogger(out_dir)

    # Optionally run under perf
    if cfg.get("enable_tracing", True) and args.trace:
        # Execute this script again under perf for the critical section
        # Simpler approach: Just start nvidia-smi dmon and run the core coro
        pass

    # Run CROSS-EVALUATION: All agents × All tasks
    print("\n" + "="*60)
    print("Running CROSS-EVALUATION BENCHMARK")
    print("="*60)
    
    num_agents = cfg.get("num_agents", 50)
    num_tasks = len(prompt_data_list)
    
    print(f"📊 Configuration:")
    print(f"   Agents: {num_agents}")
    print(f"   Tasks: {num_tasks}")
    print(f"   Total evaluations: {num_agents} × {num_tasks} = {num_agents * num_tasks}")
    
    baseline_cfg = cfg.copy()
    baseline_cfg["num_samples"] = 1
    baseline_cfg["enable_reflection"] = False
    
    # Create all M×N task-agent pairs
    cross_eval_pairs = []
    for agent_id in range(num_agents):
        for task_idx, prompt_data in enumerate(prompt_data_list):
            # Create a copy with agent_id and original task info
            task_pair = {
                "agent_id": agent_id,
                "task_idx": task_idx,
                "prompt_data": prompt_data.copy(),
            }
            cross_eval_pairs.append(task_pair)
    
    print(f"✅ Created {len(cross_eval_pairs)} task-agent pairs for cross-evaluation")
    
    # Scale semaphore for M×N operations (use num_agents * 2 as max concurrent)
    # This allows more parallelism while still controlling GPU load
    original_semaphore_value = cfg.get("num_agents", 50)
    max_concurrent = min(original_semaphore_value * 2, len(cross_eval_pairs))
    print(f"   Max concurrent operations: {max_concurrent}")
    
    gpu_mon.start()
    
    # Process all cross-evaluation pairs
    all_results = {}  # Group by agent_id: {agent_id: {results...}}
    
    try:
        # Get batch size from config (default: 8 for batching optimization)
        batch_size = baseline_cfg.get("batch_size", 8)
        
        if use_data_parallelism and models is not None:
            # DATA PARALLELISM: Distribute tasks evenly across GPUs using round-robin
            # This ensures balanced load even when agent count doesn't divide evenly
            n_gpus = len(models)
            log_success(f"\n🚀 DATA PARALLELISM: Distributing tasks evenly across {n_gpus} GPUs (round-robin)")
            
            # IMPROVED: Distribute tasks evenly across GPUs (not just agents)
            # This ensures better load balancing even when agent count doesn't divide evenly
            total_task_pairs = len(cross_eval_pairs)
            tasks_per_gpu = total_task_pairs // n_gpus
            extra_tasks = total_task_pairs % n_gpus
            
            # Distribute task-agent pairs evenly across GPUs using round-robin
            # This ensures each GPU gets approximately the same number of tasks
            gpu_task_batches = {i: [] for i in range(n_gpus)}
            
            for idx, pair in enumerate(cross_eval_pairs):
                # Round-robin distribution: assign to GPU based on index
                gpu_id = idx % n_gpus
                gpu_task_batches[gpu_id].append((pair["agent_id"], pair["prompt_data"]))
            
            # Create agent-to-GPU mapping for logging (agents may be split across GPUs)
            gpu_agent_map = {i: set() for i in range(n_gpus)}
            for gpu_id, batch in gpu_task_batches.items():
                for agent_id, _ in batch:
                    gpu_agent_map[gpu_id].add(agent_id)
            
            # Log distribution
            for gpu_id in range(n_gpus):
                agents = sorted(list(gpu_agent_map[gpu_id]))
                total_tasks = len(gpu_task_batches[gpu_id])
                agents_preview = f"{agents[:5]}{'...' if len(agents) > 5 else ''}" if len(agents) > 0 else "[]"
                log_info(f"   GPU {gpu_id}: {len(agents)} unique agent(s), {total_tasks} task-agent pairs")
                if len(agents) <= 10:
                    log_info(f"      Agents: {agents}")
            
            # Process tasks on each GPU in parallel with runtime tracking
            async def process_gpu(gpu_id: int, task_batch: list):
                """Process all tasks assigned to this GPU."""
                gpu_start_time = time.time()
                log_info(f"\n{'='*60}")
                log_info(f"🎯 GPU {gpu_id}: Processing {len(task_batch)} task-agent pairs...")
                log_info(f"{'='*60}")
                
                # Set CUDA device context for this GPU
                torch.cuda.set_device(gpu_id)
                gpu_model = models[gpu_id]
                
                # Verify model is still on correct GPU
                try:
                    first_param = next(iter(gpu_model.parameters()))
                    if first_param.device != torch.device(f"cuda:{gpu_id}"):
                        log_warning(f"   ⚠️  Model not on GPU {gpu_id}, moving...")
                        gpu_model = gpu_model.to(torch.device(f"cuda:{gpu_id}"))
                        models[gpu_id] = gpu_model  # Update in dict
                        log_success(f"   ✅ Model moved to GPU {gpu_id}")
                    else:
                        log_info(f"   ✅ Model confirmed on GPU {gpu_id}")
                except Exception as e:
                    log_warning(f"   ⚠️  Could not verify model device: {e}")
                
                gpu_results = {}  # {agent_id: result}
                
                # Group tasks by agent_id (all tasks for an agent are already on this GPU)
                agent_task_map = {}
                for agent_id, task_data in task_batch:
                    if agent_id not in agent_task_map:
                        agent_task_map[agent_id] = []
                    agent_task_map[agent_id].append(task_data)
                
                # Process each agent's tasks (all tasks for agent are on this GPU)
                for agent_id, agent_task_list in agent_task_map.items():
                    manager = AgentManager(tokenizer, gpu_model, baseline_cfg, agent_id=agent_id)
                    result = await run_agent_evaluation(agent_id, manager, agent_task_list, batch_size)
                    gpu_results[agent_id] = result
                    log_success(f"✅ GPU {gpu_id}: Agent {agent_id} completed ({len(result.get('outputs', []))} outputs)")
                
                gpu_runtime = time.time() - gpu_start_time
                log_success(f"⏱️  GPU {gpu_id} runtime: {gpu_runtime:.2f} seconds")
                
                return gpu_id, gpu_results, gpu_runtime
            
            # Run all GPUs concurrently using asyncio.gather()
            gpu_results_list = await asyncio.gather(*[
                process_gpu(gpu_id, gpu_task_batches[gpu_id]) 
                for gpu_id in range(n_gpus)
            ])
            
            # Collect per-GPU runtimes and merge results
            gpu_runtimes = {}
            for gpu_id, gpu_results, gpu_runtime in gpu_results_list:
                gpu_runtimes[gpu_id] = gpu_runtime
                for agent_id, result in gpu_results.items():
                    # With agent-based distribution, each agent's tasks are on one GPU
                    # No need to merge (but keep code for safety)
                    if agent_id not in all_results:
                        all_results[agent_id] = result
                    else:
                        # This shouldn't happen with agent-based distribution, but handle it
                        all_results[agent_id]["outputs"].extend(result.get("outputs", []))
                        all_results[agent_id]["task_ids"].extend(result.get("task_ids", []))
                        all_results[agent_id]["latencies"].extend(result.get("latencies", []))
            
            # Print per-GPU and total runtime
            print("\n" + "="*60)
            print("📊 DATA PARALLELISM PERFORMANCE METRICS")
            print("="*60)
            for gpu_id, runtime in sorted(gpu_runtimes.items()):
                print(f"   GPU {gpu_id} runtime: {runtime:.2f} seconds")
            total_runtime = max(gpu_runtimes.values())  # Total = max (since they run in parallel)
            print(f"   Total runtime (parallel): {total_runtime:.2f} seconds")
            print(f"   Speedup vs sequential: {sum(gpu_runtimes.values()) / total_runtime:.2f}x")
            print("="*60)
            
        else:
            # MODEL SHARDING: Original approach (single model, all agents share it)
            async def process_agent(agent_id: int):
                """Process all tasks for one agent."""
                log_info(f"\n{'='*60}")
                log_info(f"🤖 Agent {agent_id}/{num_agents-1}: Processing all {num_tasks} tasks...")
                log_info(f"{'='*60}")
                
                # Get all tasks for this agent
                agent_tasks = [pair["prompt_data"] for pair in cross_eval_pairs if pair["agent_id"] == agent_id]
                
                # Create agent manager with agent_id
                manager = AgentManager(tokenizer, model, baseline_cfg, agent_id=agent_id)
                
                # Run with batching enabled (Phase 1 optimization)
                result = await run_agent_evaluation(agent_id, manager, agent_tasks, batch_size)
                
                log_success(f"✅ Agent {agent_id} completed: {len(result.get('outputs', []))} outputs")
                if result.get('num_batches'):
                    log_info(f"   Processed in {result.get('num_batches')} batches (batch_size={batch_size})")
                
                return agent_id, result
            
            # Run all agents in parallel (Phase 2: Parallel Agent Execution)
            if num_agents > 1:
                log_info(f"\n🚀 Running {num_agents} agents in PARALLEL (model sharding mode)...")
                agent_results = await asyncio.gather(*[
                    process_agent(agent_id) 
                    for agent_id in range(num_agents)
                ])
                
                # Group results by agent_id
                for agent_id, result in agent_results:
                    all_results[agent_id] = result
            else:
                # Single agent (sequential is fine)
                log_info(f"\n🚀 Running 1 agent (batching enabled)...")
                agent_id, result = await process_agent(0)
                all_results[agent_id] = result
            
    finally:
        gpu_mon.stop()
    
    # Process cross-evaluation results: Group by agent and evaluate
    print("\n" + "="*60)
    print("PROCESSING CROSS-EVALUATION RESULTS")
    print("="*60)
    
    # Aggregate per-agent results
    agent_summaries = {}
    all_agent_results = []
    
    for agent_id, result in all_results.items():
        outputs = result.get("outputs", [])
        task_ids = result.get("task_ids", [])
        latencies = result.get("latencies", [])
        
        # Format results for HumanEval evaluation
        agent_results = []
        for i, output in enumerate(outputs):
            # Get task_id
            if i < len(task_ids) and task_ids[i]:
                task_id = task_ids[i]
            elif i < len(prompt_data_list):
                task_id = prompt_data_list[i].get("task_id", f"HumanEval/{i}")
            else:
                task_id = f"HumanEval/{i}"
            
            # Format completion
            if output and isinstance(output, str):
                completion = output.strip()
                # Ensure proper indentation
                if completion and not completion.startswith('    '):
                    lines = completion.split('\n')
                    completion = '\n'.join('    ' + line.lstrip() if line.strip() else line for line in lines)
                if len(completion) >= 5:
                    agent_results.append({"task_id": task_id, "completion": completion})
                else:
                    agent_results.append({"task_id": task_id, "completion": completion or "    pass"})
            else:
                agent_results.append({"task_id": task_id, "completion": "    pass"})
        
        # Save per-agent results
        agent_jsonl = os.path.join(out_dir, f"agent_{agent_id}_results.jsonl")
        from human_eval.data import write_jsonl
        write_jsonl(agent_jsonl, agent_results)
        
        # Calculate per-agent metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_time = result.get("runtime_s", 0.0)
        throughput = len(outputs) / max(total_time, 0.001)
        successes = result.get("successes", 0)
        
        agent_summaries[agent_id] = {
            "agent_id": agent_id,
            "total_tasks": len(outputs),
            "successes": successes,
            "avg_latency_s": avg_latency,
            "total_time_s": total_time,
            "throughput_rps": throughput,
            "results_file": agent_jsonl,
        }
        
        all_agent_results.extend([(agent_id, r) for r in agent_results])
        
        print(f"✅ Agent {agent_id}: {len(agent_results)} results saved to {agent_jsonl}")
    
    # Evaluate each agent's results
    print("\n" + "="*60)
    print("EVALUATING AGENT RESULTS")
    print("="*60)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    for agent_id, summary in agent_summaries.items():
        agent_jsonl = summary["results_file"]
        print(f"\n📊 Evaluating Agent {agent_id}...")
        
        try:
            eval_result = runner.evaluate_outputs(
                agent_jsonl,
                k=[1, 10, 100],
                n_workers=cfg.get("eval_workers", 4),
                timeout=cfg.get("eval_timeout", 10.0)
            )
            
            # Add evaluation metrics to agent summary
            agent_summaries[agent_id]["pass@1"] = eval_result.get("pass@1", 0.0)
            agent_summaries[agent_id]["pass@10"] = eval_result.get("pass@10", 0.0)
            agent_summaries[agent_id]["pass@100"] = eval_result.get("pass@100", 0.0)
            
            # Save per-agent evaluation
            agent_eval_file = os.path.join(out_dir, f"agent_{agent_id}_evaluation.json")
            with open(agent_eval_file, "w") as f:
                json.dump(eval_result, f, indent=2)
            
            print(f"   Agent {agent_id} Pass@1: {eval_result.get('pass@1', 0.0):.2%}")
        except Exception as e:
            print(f"   ⚠️  Agent {agent_id} evaluation failed: {e}")
            agent_summaries[agent_id]["pass@1"] = 0.0
            agent_summaries[agent_id]["pass@10"] = 0.0
            agent_summaries[agent_id]["pass@100"] = 0.0
    
    # Save overall summary
    summary_file = os.path.join(out_dir, "agent_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "num_agents": num_agents,
            "num_tasks": num_tasks,
            "total_evaluations": num_agents * num_tasks,
            "agents": list(agent_summaries.values())
        }, f, indent=2)
    
    print(f"\n✅ Agent summary saved to {summary_file}")
    
    # Print summary table
    print("\n" + "="*60)
    print("PER-AGENT SUMMARY")
    print("="*60)
    print(f"{'Agent':<8} {'Pass@1':<10} {'Avg Latency':<15} {'Throughput':<15} {'Tasks':<8}")
    print("-" * 60)
    for agent_id, summary in sorted(agent_summaries.items()):
        print(f"{agent_id:<8} {summary.get('pass@1', 0.0):<10.2%} {summary.get('avg_latency_s', 0.0):<15.2f} "
              f"{summary.get('throughput_rps', 0.0):<15.2f} {summary.get('total_tasks', 0):<8}")
    
    # Calculate overall metrics for W&B logging
    overall_pass1 = sum(s.get('pass@1', 0.0) for s in agent_summaries.values()) / len(agent_summaries) if agent_summaries else 0.0
    overall_latency = sum(s.get('avg_latency_s', 0.0) for s in agent_summaries.values()) / len(agent_summaries) if agent_summaries else 0.0
    overall_throughput = sum(s.get('throughput_rps', 0.0) for s in agent_summaries.values()) / len(agent_summaries) if agent_summaries else 0.0
    
    # For backward compatibility, save a "score.json" with overall metrics
    score = {
        "pass@1": overall_pass1,
        "pass@10": 0.0,  # Calculate if needed
        "pass@100": 0.0,
        "total": num_tasks,
        "rate_est": overall_pass1,
        "num_agents": num_agents,
        "avg_latency_s": overall_latency,
        "avg_throughput_rps": overall_throughput
    }
    with open(os.path.join(out_dir, "score.json"), "w") as f:
        json.dump(score, f, indent=2)
    
    # Log per-agent metrics to W&B
    if WANDB_AVAILABLE and wandb and wandb.run:
        try:
            # Log overall metrics
            wandb.log({
                "overall_pass@1": overall_pass1,
                "overall_avg_latency_s": overall_latency,
                "overall_avg_throughput_rps": overall_throughput,
                "num_agents": num_agents,
                "num_tasks": num_tasks,
                "total_evaluations": num_agents * num_tasks,
            })
            
            # Log per-agent metrics
            for agent_id, summary in agent_summaries.items():
                wandb.log({
                    f"agent_{agent_id}_pass@1": summary.get('pass@1', 0.0),
                    f"agent_{agent_id}_avg_latency_s": summary.get('avg_latency_s', 0.0),
                    f"agent_{agent_id}_throughput_rps": summary.get('throughput_rps', 0.0),
                })
            
            print(f"✅ Per-agent metrics logged to W&B")
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Overall Pass@1: {overall_pass1:.2%}")
    print(f"Overall Avg Latency: {overall_latency:.2f}s")
    print(f"Overall Avg Throughput: {overall_throughput:.2f} req/s")
    print(f"Total evaluations: {num_agents * num_tasks}")
    print(f"Agents: {num_agents}, Tasks: {num_tasks}")
    
    # Save generation parameters for tracking
    gen_params = {
        "max_new_tokens": cfg.get("max_new_tokens"),
        "temperature": cfg.get("temperature"),
        "top_p": cfg.get("top_p"),
        "top_k": cfg.get("top_k"),
        "repetition_penalty": cfg.get("repetition_penalty"),
        "num_agents": cfg.get("num_agents"),
        "num_samples": cfg.get("num_samples", 3),
        "enable_reflection": cfg.get("enable_reflection", False),
    }
    with open(os.path.join(out_dir, "generation_params.json"), "w") as f:
        json.dump(gen_params, f, indent=2)
    
    # Cleanup and finalize W&B
    if WANDB_AVAILABLE and wandb and wandb.run:
        print("\n🔄 Finalizing W&B run and syncing to cloud...")
        try:
            # Ensure all metrics are logged
            wandb.log({})  # Flush any pending logs
            # Finish the run
            wandb.finish()
            print("✅ W&B run finalized and synced to cloud")
            print(f"   View results at: https://wandb.ai/kishorelin-tad001-university-of-windsor/linux-ai-acceleration")
        except Exception as e:
            print(f"⚠️  Error finalizing W&B run: {e}")
            # Try to sync manually
            try:
                import subprocess
                print("   Attempting manual sync...")
                result = subprocess.run(["wandb", "sync", "--no-include-synced"], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print("✅ Manually synced W&B runs")
                else:
                    print(f"⚠️  Manual sync failed: {result.stderr}")
                    print("   You can manually sync with: wandb sync")
            except Exception as sync_error:
                print(f"⚠️  Could not sync W&B: {sync_error}")
                print("   You can manually sync offline runs with: wandb sync")
    else:
        print("⚠️  W&B run not available for finalization")
    
    if resource_monitor:
        resource_monitor.stop()

    if perf_logger:
        perf_logger.stop()
    
    print("\n=== Files Saved ===")
    print(f"✅ Agent summary: {summary_file}")
    print(f"✅ Score: {os.path.join(out_dir, 'score.json')}")
    print(f"✅ Generation parameters: {os.path.join(out_dir, 'generation_params.json')}")
    print(f"✅ Per-agent results: logs/agent_*_results.jsonl")
    print(f"✅ Per-agent evaluations: logs/agent_*_evaluation.json")


if __name__ == "__main__":
    asyncio.run(main())
