# model_loader.py
import torch
import os
from threading import Lock
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Tuple, Dict, Any
from utils import log_info, log_warning, log_success, log_error

class ModelLoader:
    """
    ModelLoader with lightweight caching so we only hit disk once per run.
    Keeping models/tokenizers warm avoids repeated I/O stalls that inflate
    context switches and kernel wake-ups.
    """

    _tokenizer_cache: Dict[str, AutoTokenizer] = {}
    _model_cache: Dict[Tuple[str, bool, str], AutoModelForCausalLM] = {}
    _cache_lock: Lock = Lock()

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load(self, gpu_id: int = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load model with model sharding (default) or on specific GPU (data parallelism).
        
        Args:
            gpu_id: If specified, load model on this specific GPU (data parallelism).
                   If None, use device_map="auto" for model sharding.
        """
        model_id = self.cfg.get("model_id")
        load_in_4bit = bool(self.cfg.get("load_in_4bit", True))

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",  # Use NF4 quantization (best quality)
                bnb_4bit_quant_storage=torch.uint8,  # Storage type
                # Note: llm_int8_enable_fp32_cpu_offload is for 8-bit, not 4-bit
                # For 4-bit, CPU offload is handled by device_map="auto"
            )

        cache_suffix = f"gpu{gpu_id}" if gpu_id is not None else "auto"

        with ModelLoader._cache_lock:
            tokenizer = ModelLoader._tokenizer_cache.get(model_id)

        if tokenizer:
            log_info("🧠 Reusing cached tokenizer to avoid repeated file I/O.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            with ModelLoader._cache_lock:
                ModelLoader._tokenizer_cache[model_id] = tokenizer

        model_cache_key = (model_id, load_in_4bit, cache_suffix)
        with ModelLoader._cache_lock:
            cached_model = ModelLoader._model_cache.get(model_cache_key)

        if cached_model:
            log_info("⚡ Reusing cached model weights to keep GPUs hot.")
            cached_model.eval()
            return tokenizer, cached_model
        
        # Load model with GPU-only placement
        load_kwargs = {
            "quantization_config": bnb_config,
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
        }
        
        if torch.cuda.is_available():
            # DATA PARALLELISM: Load model on specific GPU
            if gpu_id is not None:
                log_info(f"🔄 Loading model on GPU {gpu_id} for DATA PARALLELISM...")
                # Set CUDA device context first
                torch.cuda.set_device(gpu_id)
                
                # Use device_map with explicit device mapping
                # device_map={"": f"cuda:{gpu_id}"} places entire model on that GPU
                load_kwargs["device_map"] = {"": f"cuda:{gpu_id}"}
                log_info(f"   Set CUDA device context to GPU {gpu_id}")
                log_info(f"   Using device_map={{\"\": \"cuda:{gpu_id}\"}} to load model on GPU {gpu_id}")
            else:
                # MODEL SHARDING: Use device_map="auto" for intelligent multi-GPU sharding
                log_info("🔄 Loading model with device_map='auto' for intelligent GPU sharding...")
                n_gpus = torch.cuda.device_count()
                log_info(f"   Using {n_gpus} GPU(s) for model sharding (GPU-only, no CPU offload)")
                
                # Check if CUDA_VISIBLE_DEVICES is set (limits visible GPUs)
                cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_visible:
                    log_warning(f"   ⚠️  CUDA_VISIBLE_DEVICES={cuda_visible} (limiting GPU access)")
                    # Parse visible devices
                    visible_gpus = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
                    if visible_gpus:
                        log_info(f"   Using GPUs: {visible_gpus}")
                
                # Use auto device map with max_memory to prevent CPU offload
                load_kwargs["device_map"] = "auto"
                
                # Calculate max memory per GPU - force GPU-only, no CPU fallback
                max_memory = {}
                for i in range(n_gpus):
                    try:
                        mem_info = torch.cuda.mem_get_info(i)
                        total_mem_gb = mem_info[1] / (1024**3)
                        # Use 90% of GPU memory to maximize GPU usage, but keep on GPU
                        # Reduced from 95% to avoid OOM issues
                        max_memory[i] = f"{int(total_mem_gb * 0.90)}GiB"
                    except Exception as e:
                        log_warning(f"   ⚠️  Could not get memory info for GPU {i}: {e}")
                        # Use conservative estimate
                        max_memory[i] = "14GiB"  # Conservative for A16 (15.61GB)
                
                # CRITICAL: Set CPU to 0 to prevent CPU offload
                max_memory["cpu"] = "0GiB"
                
                load_kwargs["max_memory"] = max_memory
                log_info(f"   Max memory per GPU (GPU-only, CPU=0): {max_memory}")
                log_success(f"   ✅ Model will be sharded across {n_gpus} GPUs (NO CPU offload)")
        else:
            load_kwargs["device_map"] = "cpu"
            log_info("   No GPU available, using CPU")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs
        )
        
        # For data parallelism: Ensure model is on the correct GPU
        if gpu_id is not None and torch.cuda.is_available():
            target_device = torch.device(f"cuda:{gpu_id}")
            # Set device context
            torch.cuda.set_device(gpu_id)
            
            # Verify and move model to correct device if needed
            try:
                # Check if model has device_map (accelerate library)
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    # Model was loaded with device_map, verify all layers are on correct device
                    all_correct = True
                    for layer_name, device in model.hf_device_map.items():
                        if isinstance(device, int):
                            device_str = f"cuda:{device}"
                        elif isinstance(device, str):
                            device_str = device
                        else:
                            device_str = str(device)
                        
                        # Check if device matches target
                        if device_str != str(target_device) and f"cuda:{gpu_id}" not in device_str:
                            all_correct = False
                            log_warning(f"   ⚠️  Layer {layer_name} on {device_str}, expected {target_device}")
                    
                    if not all_correct:
                        log_warning(f"   ⚠️  Some layers not on {target_device}, moving model...")
                        model = model.to(target_device)
                        log_success(f"   ✅ Model moved to {target_device}")
                    else:
                        log_success(f"   ✅ All layers confirmed on {target_device}")
                else:
                    # No device_map, check parameters directly
                    param_iter = iter(model.parameters())
                    first_param = next(param_iter)
                    actual_device = first_param.device
                    
                    if actual_device != target_device:
                        log_warning(f"   ⚠️  Model on {actual_device}, expected {target_device}. Moving...")
                        model = model.to(target_device)
                        log_success(f"   ✅ Model moved to {target_device}")
                    else:
                        log_success(f"   ✅ Model confirmed on {target_device}")
            except Exception as e:
                log_warning(f"   ⚠️  Could not verify model device: {e}")
                # Fallback: try to move model anyway
                try:
                    model = model.to(target_device)
                    log_success(f"   ✅ Model moved to {target_device} (fallback)")
                except Exception as e2:
                    log_error(f"   ❌ Failed to move model to {target_device}: {e2}")
        
        # Print model device placement for debugging (only if verbose)
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            log_info("📊 Model device placement:")
            device_counts = {}
            for layer_name, device in model.hf_device_map.items():
                device_str = str(device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
            
            # Show summary by device
            for device, count in sorted(device_counts.items()):
                log_info(f"   {device}: {count} layers")
            
            # Check if any layers are on CPU (this shouldn't happen with max_memory)
            cpu_layers = [d for d in device_counts.keys() if 'cpu' in str(d).lower()]
            if cpu_layers:
                log_warning(f"   ⚠️  WARNING: {sum(device_counts[d] for d in cpu_layers)} layers on CPU!")
            else:
                log_success(f"   ✅ All layers on GPU(s)")
        
        # Performance optimizations
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        
        model.eval()
        with ModelLoader._cache_lock:
            ModelLoader._model_cache[model_cache_key] = model
        if gpu_id is not None:
            log_success(f"✅ Model loaded successfully on GPU {gpu_id} (data parallelism)")
        else:
            log_success("✅ Model loaded successfully with auto GPU sharding")
        return tokenizer, model
