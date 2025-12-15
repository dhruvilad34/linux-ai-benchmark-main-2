# agent_manager.py
import asyncio, time, re
from typing import List, Dict, Any, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpu_assigner import assign_gpu, check_gpu_availability
from utils import log_info, log_warning, log_error, log_success

class AgentManager:
    def __init__(self, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, cfg: Dict[str, Any], agent_id: int = 0):
        self.tk = tokenizer
        self.model = model
        self.cfg = cfg
        self.agent_id = agent_id  # Agent identifier for cross-evaluation
        
        # Scale semaphore for batching: allow multiple batches concurrently
        # For batching, we want to process multiple batches in parallel
        base_semaphore = int(cfg.get("num_agents", 100))
        batch_size = int(cfg.get("batch_size", 8))
        
        # For batching: allow multiple batches to run concurrently
        # Formula: If batching, allow min(4 * num_batches_estimate, 20) concurrent batches
        # This ensures we can process multiple batches in parallel
        if batch_size > 1:
            # Estimate number of batches (assuming ~164 tasks)
            # Allow up to 10-20 concurrent batches for better GPU utilization
            # But cap at reasonable limit to avoid OOM
            scaled_semaphore = min(20, max(4, batch_size * 2))
        else:
            scaled_semaphore = base_semaphore
        
        self.sem = asyncio.Semaphore(scaled_semaphore)
        log_info(f"🔧 Agent {agent_id} semaphore set to {scaled_semaphore} (batch_size={batch_size}, base={base_semaphore})")
        self.batch_size = int(cfg.get("batch_size", 8))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0.35))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.top_k = int(cfg.get("top_k", 0))  # 0 means disabled
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.0))
        self.do_sample = bool(cfg.get("do_sample", True))
        self.num_beams = int(cfg.get("num_beams", 1))
        self.early_stopping = bool(cfg.get("early_stopping", True))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.num_samples = int(cfg.get("num_samples", 1))  # Multi-sample generation
        self.enable_reflection = bool(cfg.get("enable_reflection", False))  # Reflection/retry with corrective prompt
        
        # Set pad token if not set
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token

    def _generate_sync(self, input_ids: torch.Tensor, generate_kwargs: Dict[str, Any]) -> torch.Tensor:
        """Synchronous generation wrapper for thread pool execution."""
        # Ensure input is on the correct device for sharded models
        device = self._get_model_device()
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        
        with torch.no_grad():
            return self.model.generate(input_ids, **generate_kwargs)
    
    def _get_model_device(self):
        """
        Get the device for model inputs.
        For sharded models (device_map="auto"), find the first GPU device that has model layers.
        For data parallelism (device_map="cuda:X"), return that specific device.
        This is where inputs need to go for the forward pass to work correctly.
        """
        # Check if model is sharded (has hf_device_map)
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            # Find first GPU device in the device map (inputs go to first layer's device)
            device_values = list(self.model.hf_device_map.values())
            
            # Look for GPU devices first
            gpu_devices = []
            for device in device_values:
                if isinstance(device, int) and device >= 0:
                    # Device ID (e.g., 0, 1, 2)
                    gpu_devices.append(device)
                elif isinstance(device, str) and ("cuda" in device.lower() or device.isdigit()):
                    if device.isdigit():
                        gpu_devices.append(int(device))
                    elif "cuda:" in device.lower():
                        gpu_devices.append(int(device.split(":")[-1]))
            
            if gpu_devices:
                # Use the first GPU device found (where the first layer is)
                first_gpu = min(gpu_devices) if gpu_devices else 0
                return torch.device(f"cuda:{first_gpu}")
            
            # Fallback: use first available GPU
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            return torch.device("cpu")
        
        # For data parallelism: Check where model parameters actually are
        # When using device_map="cuda:X", parameters are on that specific device
        try:
            # Get device of first parameter (most reliable method)
            # Use iter() to get proper iterator
            param_iter = iter(self.model.parameters())
            first_param = next(param_iter)
            if first_param is not None:
                param_device = first_param.device
                if param_device.type == 'cuda':
                    log_info(f"🔧 Agent {self.agent_id}: Detected model device from parameters: {param_device}")
                    return param_device
        except (StopIteration, AttributeError, TypeError) as e:
            log_warning(f"⚠️  Could not get device from parameters: {e}")
            pass
        
        # Single device model - check model.device attribute
        if hasattr(self.model, 'device'):
            device = self.model.device
            # Ensure it's a GPU device
            if isinstance(device, torch.device) and device.type == 'cuda':
                log_info(f"🔧 Agent {self.agent_id}: Detected model device from model.device: {device}")
                return device
        
        # Check if model has named_parameters (alternative method)
        try:
            param_iter = iter(self.model.named_parameters())
            name, first_param = next(param_iter)
            if first_param is not None:
                param_device = first_param.device
                if param_device.type == 'cuda':
                    log_info(f"🔧 Agent {self.agent_id}: Detected model device from named_parameters: {param_device}")
                    return param_device
        except (StopIteration, AttributeError, TypeError) as e:
            log_warning(f"⚠️  Could not get device from named_parameters: {e}")
            pass
        
        # Fallback - this should not happen for properly loaded models
        log_warning(f"⚠️  Agent {self.agent_id}: Could not detect model device, defaulting to cuda:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    
    def format_prompt_for_generation(self, prompt_data: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """
        Format prompt using chat template and return tokenized input_ids.
        Returns: (input_ids tensor, original_length for slicing)
        """
        # Get the correct device for this model (handles sharded models)
        device = self._get_model_device()
        
        # Use chat template format for LLaMA-3.1-Instruct
        system_msg = prompt_data.get("system_msg", "You are an expert Python coding assistant. Write correct, executable functions.")
        user_msg = prompt_data.get("user_msg", prompt_data.get("prompt_text", ""))
        
        # Check if tokenizer has chat template (LLaMA-3.1-Instruct should have one)
        if hasattr(self.tk, "apply_chat_template") and self.tk.chat_template is not None:
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]
                # Apply chat template and tokenize in one step
                # CRITICAL: Tokenizer creates tensors on CPU by default, must move to device
                input_ids = self.tk.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                # Explicitly move to device and verify
                if input_ids.device != device:
                    input_ids = input_ids.to(device)
                assert input_ids.device == device, f"Input IDs must be on {device}, got {input_ids.device}"
                
                return input_ids, input_ids.shape[1]
            except Exception as e:
                print(f"Warning: Chat template failed, using raw prompt: {e}")
        
        # Fallback: tokenize raw prompt
        prompt_text = prompt_data.get("user_msg", prompt_data.get("prompt_text", ""))
        inputs = self.tk(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        # CRITICAL: Tokenizer creates tensors on CPU, must move to device
        input_ids = inputs.input_ids.to(device)
        assert input_ids.device == device, f"Input IDs must be on {device}, got {input_ids.device}"
        return input_ids, input_ids.shape[1]

    def extract_completion(self, full_output: str, prompt_text: str, entry_point: str) -> str:
        """
        Extract only the completion part (function body) from the full generated text.
        
        HumanEval format: 
        - prompt = function signature + docstring (ends with triple quotes)
        - completion = function body (starts with 4 spaces, preserves indentation)
        - Final program = prompt + completion + test + check(entry_point)
        
        The completion should be the EXACT code that goes after the docstring.
        """
        completion = full_output.strip()
        
        # Step 1: Remove code fences and prefixes
        completion = re.sub(r'^```python\s*', '', completion, flags=re.IGNORECASE)
        completion = re.sub(r'```\s*$', '', completion, flags=re.IGNORECASE)
        completion = re.sub(r'^(Here\s+is|Here\'s|The\s+function|The\s+implementation|Implementation)[:.]?\s*', '', completion, flags=re.IGNORECASE)
        
        # Remove any special characters that shouldn't be in code (like ! markers)
        # These might be added by the model or tokenizer
        completion = re.sub(r'!+', '', completion)  # Remove exclamation marks
        completion = completion.strip()
        
        # Step 2: Remove function signature if model regenerated it
        if entry_point and f"def {entry_point}" in completion:
            # Find function definition
            parts = completion.split(f"def {entry_point}", 1)
            if len(parts) > 1:
                after_def = parts[1]
                # Skip signature line and docstring
                lines = after_def.split('\n')
                body_lines = []
                in_docstring = False
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        if not in_docstring:
                            in_docstring = True
                            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                                in_docstring = False
                            continue
                        else:
                            in_docstring = False
                            continue
                    elif in_docstring:
                        continue
                    else:
                        body_lines.append(line)
                
                if body_lines:
                    completion = '\n'.join(body_lines)
        
        # Step 3: Remove imports and function definitions at start
        lines = completion.split('\n')
        body_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('from ', 'import ', 'def ', 'class ')):
                continue
            body_lines.append(line)
        
        completion = '\n'.join(body_lines).strip()
        
        # Step 4: Remove docstring if present at start
        if completion.strip().startswith('"""'):
            lines = completion.split('\n')
            body_lines = []
            in_docstring = True
            for line in lines:
                stripped = line.strip()
                if in_docstring:
                    if '"""' in stripped:
                        in_docstring = False
                    continue
                body_lines.append(line)
            completion = '\n'.join(body_lines).strip()
        
        # Step 5: Remove trailing explanations (stop at explanatory text or code fences)
        lines = completion.split('\n')
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('```'):
                break
            if stripped and not stripped.startswith('#'):
                if re.match(r'^(This|The|It|Here|The function|This function|This implementation)', stripped, re.IGNORECASE):
                    break
            code_lines.append(line)
        
        completion = '\n'.join(code_lines).strip()
        
        # Step 6: Clean up any remaining artifacts
        # Remove lines that are clearly not code (explanatory text, etc.)
        lines = completion.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines, pure punctuation, or lines that are clearly not code
            if not stripped:
                continue
            # Skip lines that are just punctuation or special characters
            if re.match(r'^[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>?/`~]+$', stripped):
                continue
            # Skip lines that look like explanatory text (start with common words)
            if re.match(r'^(This|The|It|Here|Note|Example|Explanation|Solution|Code|Function)', stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        
        completion = '\n'.join(cleaned_lines).strip()
        
        # Step 7: Validate we have code
        if not completion:
            return ""
        
        has_code = any(
            keyword in line.strip() 
            for line in completion.split('\n') 
            for keyword in ['return', 'if ', 'for ', 'while ', '=', 'def ', 'class ']
            if line.strip() and not line.strip().startswith('#')
        )
        
        if not has_code:
            return ""
        
        # Step 7: Fix indentation - CRITICAL for HumanEval
        # Parse Python structure and fix indentation based on control flow
        lines = completion.split('\n')
        if not lines:
            return ""
        
        # Remove empty lines at start/end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop(-1)
        
        if not lines:
            return ""
        
        # Find minimum indentation
        min_indent = min(
            (len(line) - len(line.lstrip()) for line in lines if line.strip()),
            default=0
        )
        
        # Step 1: Normalize base indentation to 4 spaces
        offset = 4 - min_indent
        normalized_lines = []
        for line in lines:
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                new_indent = current_indent + offset
                if new_indent < 4:
                    new_indent = 4
                normalized_lines.append(' ' * new_indent + line.lstrip())
            else:
                normalized_lines.append('')
        
        # Step 2: Fix control flow indentation - CRITICAL FIX
        # After any line ending with ':' (if/for/while/try/except/with/elif/else),
        # the next non-empty line MUST be indented 4 more spaces
        # This handles cases where model generates all lines at same indent level
        fixed_lines = []
        i = 0
        indent_stack = []  # Stack of body_indent values (indent level for body of current block)
        
        while i < len(normalized_lines):
            line = normalized_lines[i]
            stripped = line.strip()
            
            if not stripped:
                fixed_lines.append('')
                i += 1
                continue
            
            current_indent = len(line) - len(stripped)
            
            # Check if this is a control flow statement (ends with ':')
            is_control_flow = stripped.endswith(':') and any(
                stripped.startswith(kw) for kw in ['if ', 'for ', 'while ', 'elif ', 'else:', 'try:', 'except', 'with ', 'def ', 'class ']
            )
            
            # Determine expected indent for this line based on control flow structure
            # If we're in a block (stack not empty), we should be at the body indent
            if indent_stack:
                expected_indent = indent_stack[-1]  # Body indent of innermost block
            else:
                expected_indent = 4  # Function body level
            
            # Before fixing, check if we've dedented past blocks
            # Only check dedentation if current indent is STRICTLY less than expected
            # This prevents false positives when all lines start at same indent
            if not is_control_flow and indent_stack and current_indent < expected_indent - 4:
                # Current indent is significantly less than expected - we've dedented
                # Pop all blocks we've dedented past
                while indent_stack:
                    control_indent = indent_stack[-1] - 4  # Control was 4 spaces before body
                    if current_indent <= control_indent:
                        # We've dedented past this block - pop it
                        indent_stack.pop()
                    else:
                        break
                # Recalculate expected indent after popping
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 4
            
            # Fix indentation to expected level
            if current_indent != expected_indent:
                fixed_line = ' ' * expected_indent + stripped
                fixed_lines.append(fixed_line)
                actual_indent = expected_indent
            else:
                fixed_lines.append(line)
                actual_indent = current_indent
            
            # If control flow, push the body indent (4 more spaces) onto stack for next line
            if is_control_flow:
                next_body_indent = actual_indent + 4
                indent_stack.append(next_body_indent)
            
            i += 1
        
        # Step 3: Final validation - ensure all lines have at least 4 spaces
        # Don't strip the entire completion - preserve leading spaces on first line
        final_lines = []
        for line in fixed_lines:
            if line.strip():
                stripped = line.lstrip()
                current_indent = len(line) - len(stripped)
                if current_indent < 4:
                    final_lines.append('    ' + stripped)
                else:
                    final_lines.append(line)
            else:
                final_lines.append(line)
        
        completion = '\n'.join(final_lines)
        # Only strip trailing whitespace, not leading (preserve first line indent)
        completion = completion.rstrip()
        
        return completion

    def create_reflection_prompt(self, prompt_data: Dict[str, Any], error_msg: str = "") -> Dict[str, Any]:
        """
        Create a corrective prompt for reflection/retry.
        """
        original_user_msg = prompt_data.get("user_msg", "")
        prompt_text = prompt_data.get("prompt_text", "")
        
        # Create corrective prompt
        reflection_msg = f"""You are an expert Python developer.

Complete the following function so that it passes all test cases.

{prompt_text}

# Write your solution below:

The previous attempt had issues. Please write a corrected solution.
Requirements:
- Start each line with 4 spaces for the function body
- Use 8 spaces for nested blocks (inside if/for/while)
- Use 12 spaces for nested blocks inside nested blocks
- Do NOT include the function signature, imports, or docstring
- Write complete, executable code that solves the problem
- Ensure all edge cases are handled correctly"""
        
        # Create new prompt_data with reflection message
        reflection_prompt_data = prompt_data.copy()
        reflection_prompt_data["user_msg"] = reflection_msg
        return reflection_prompt_data
    
    async def run_one(self, prompt_data: Dict[str, Any], retry_count: int = 0, is_reflection: bool = False) -> Tuple[str, float, bool]:
        """
        Run a single generation task with retry logic.
        Returns: (completion, latency, success)
        
        Args:
            prompt_data: Dictionary with prompt information
            retry_count: Number of retries attempted
            is_reflection: Whether this is a reflection/retry attempt
        """
        start_time = time.time()
        
        try:
            # Format prompt using chat template and get tokenized input_ids
            input_ids, input_length = self.format_prompt_for_generation(prompt_data)
            
            # Generate with configured parameters
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self.tk.pad_token_id,
                "eos_token_id": self.tk.eos_token_id,
                "num_beams": self.num_beams,
            }
            
            # early_stopping is only valid when num_beams > 1
            if self.num_beams > 1:
                generate_kwargs["early_stopping"] = self.early_stopping
            
            # Add sampling parameters only if do_sample is True
            if self.do_sample:
                generate_kwargs["temperature"] = self.temperature
                generate_kwargs["top_p"] = self.top_p
                if self.top_k > 0:  # Only add if top_k is enabled
                    generate_kwargs["top_k"] = self.top_k
            
            # Add repetition penalty if not 1.0 (neutral)
            if self.repetition_penalty != 1.0:
                generate_kwargs["repetition_penalty"] = self.repetition_penalty
            
            # Run generation in thread pool to allow true async concurrency
            # This allows multiple agents to queue GPU requests concurrently
            # The semaphore already limits concurrency, but we need async execution
            loop = asyncio.get_event_loop()
            output_ids = await loop.run_in_executor(
                None,
                lambda: self._generate_sync(input_ids, generate_kwargs)
            )
            
            # Decode only the new tokens (completion part)
            new_tokens = output_ids[0, input_length:]
            completion_raw = self.tk.decode(new_tokens, skip_special_tokens=True)
            
            # Extract clean completion - but preserve ALL generated code
            completion = self.extract_completion(
                completion_raw, 
                prompt_data.get("prompt_text", ""),
                prompt_data.get("entry_point", "")
            )
            
            # CRITICAL: If completion is too short or seems incomplete, it might have been cut off
            # HumanEval expects complete function body, not fragments
            if len(completion.strip()) < 20:
                # Very short completion - likely incomplete, try to get more
                if retry_count < self.max_retries:
                    # Retry with longer generation
                    await asyncio.sleep(0.1)
                    return await self.run_one(prompt_data, retry_count + 1, is_reflection)
            
            latency = time.time() - start_time
            
            # Validate completion
            if not completion or len(completion.strip()) < 10:
                if retry_count < self.max_retries:
                    # Retry with slightly higher temperature
                    await asyncio.sleep(0.1)  # Small delay
                    return await self.run_one(prompt_data, retry_count + 1, is_reflection)
                else:
                    return ("", latency, False)
            
            return (completion, latency, True)
            
        except Exception as e:
            latency = time.time() - start_time
            if retry_count < self.max_retries:
                await asyncio.sleep(0.1)
                return await self.run_one(prompt_data, retry_count + 1, is_reflection)
            return (f"[ERROR] {e}", latency, False)
    
    async def run_one_multi_sample(self, prompt_data: Dict[str, Any]) -> Tuple[List[str], float, bool]:
        """
        Generate multiple samples for a single task (for pass@k calculation).
        Returns: (list of completions, average_latency, success)
        """
        start_time = time.time()
        completions = []
        
        for sample_idx in range(self.num_samples):
            completion, latency, success = await self.run_one(prompt_data, retry_count=0)
            if success and completion:
                completions.append(completion)
            await asyncio.sleep(0.05)  # Small delay between samples
        
        avg_latency = (time.time() - start_time) / max(1, len(completions))
        return (completions, avg_latency, len(completions) > 0)

    async def run_tasks(self, prompt_data_list: List[Dict[str, Any]], use_multi_sample: bool = False, batch_size: int = 8) -> Dict[str, Any]:
        """
        Run all tasks with improved generation and error handling.
        Supports multi-sample generation, reflection logic, and GPU batching.
        Returns results with agent_id included for cross-evaluation.
        
        Args:
            prompt_data_list: List of tasks to process
            use_multi_sample: Whether to generate multiple samples per task
            batch_size: Number of tasks to process together in a batch (default: 8)
        """
        t0 = time.time()
        
        # Use batching if batch_size > 1 and not using multi-sample
        use_batching = batch_size > 1 and not use_multi_sample
        
        if use_batching:
            return await self._run_tasks_batched(prompt_data_list, batch_size)
        else:
            return await self._run_tasks_individual(prompt_data_list, use_multi_sample)
    
    async def _run_tasks_batched(self, prompt_data_list: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        """Process tasks in batches for better GPU utilization."""
        t0 = time.time()
        all_outputs, all_task_ids, all_latencies = [], [], []
        total_successes = 0
        
        # Group tasks into batches
        batches = [prompt_data_list[i:i+batch_size] 
                  for i in range(0, len(prompt_data_list), batch_size)]
        
        log_info(f"\n🚀 Agent {self.agent_id}: Processing {len(prompt_data_list)} tasks in {len(batches)} batches (batch_size={batch_size})...")
        
        async def process_batch(batch, batch_idx):
            """Process a batch of tasks together."""
            batch_start = time.time()
            
            # GPU throttling: wait if all GPUs are near capacity
            while not check_gpu_availability(threshold=0.9):
                log_warning(f"⚠️  All GPUs near capacity (90%+), waiting for free memory... (Agent {self.agent_id}, Batch {batch_idx})")
                await asyncio.sleep(5)
            
            # Use semaphore to limit concurrent batches (not individual tasks)
            # This allows multiple batches to process in parallel
            async with self.sem:
                try:
                    # Tokenize all prompts in batch
                    batch_inputs = []
                    batch_lengths = []
                    batch_task_ids = []
                    
                    # Get device first (GPU for sharded models) - CRITICAL for GPU placement
                    device = self._get_model_device()
                    log_info(f"🔧 Agent {self.agent_id}: Using device {device} for batch processing")
                    
                    for prompt_data in batch:
                        input_ids, input_length = self.format_prompt_for_generation(prompt_data)
                        # CRITICAL: Ensure input_ids are on the correct GPU device
                        if input_ids.device != device:
                            log_warning(f"⚠️  Moving input_ids from {input_ids.device} to {device}")
                            input_ids = input_ids.to(device)
                        # Double-check device
                        assert input_ids.device == device, f"Input must be on {device}, got {input_ids.device}"
                        batch_inputs.append(input_ids)
                        batch_lengths.append(input_length)
                        batch_task_ids.append(prompt_data.get("task_id", ""))
                    
                    # Pad to same length for batching
                    if len(batch_inputs) > 1:
                        max_len = max(inp.shape[1] for inp in batch_inputs)
                        padded_batch = []
                        attention_masks = []
                        
                        for input_ids in batch_inputs:
                            # CRITICAL: Ensure on correct GPU before any operations
                            if input_ids.device != device:
                                log_warning(f"⚠️  Moving input_ids from {input_ids.device} to {device} during padding")
                                input_ids = input_ids.to(device)
                            
                            pad_length = max_len - input_ids.shape[1]
                            if pad_length > 0:
                                # Create padding tensor on the SAME GPU device
                                pad_tensor = torch.zeros(
                                    1, pad_length, 
                                    dtype=input_ids.dtype, 
                                    device=device  # Force GPU device
                                )
                                # Ensure input_ids is on same device before concatenation
                                if input_ids.device != device:
                                    input_ids = input_ids.to(device)
                                padded = torch.cat([input_ids, pad_tensor], dim=1)
                                # Verify padded is on correct device
                                assert padded.device == device, f"Padded tensor must be on {device}, got {padded.device}"
                            else:
                                padded = input_ids
                                if padded.device != device:
                                    padded = padded.to(device)
                            
                            padded_batch.append(padded)
                            # Create attention mask on the SAME device as padded
                            # torch.ones_like() inherits device from input, but we verify it
                            attention_mask_item = torch.ones_like(padded)
                            # Ensure it's on correct device (should already be, but verify)
                            if attention_mask_item.device != device:
                                attention_mask_item = attention_mask_item.to(device)
                            attention_masks.append(attention_mask_item)
                        
                        # CRITICAL: Verify all tensors are on the same device before concatenation
                        for i, tensor in enumerate(padded_batch):
                            if tensor.device != device:
                                log_error(f"❌ Tensor {i} in padded_batch is on {tensor.device}, expected {device}")
                                padded_batch[i] = tensor.to(device)
                        
                        for i, tensor in enumerate(attention_masks):
                            if tensor.device != device:
                                log_error(f"❌ Tensor {i} in attention_masks is on {tensor.device}, expected {device}")
                                attention_masks[i] = tensor.to(device)
                        
                        # Stack into batch tensor on GPU
                        batch_tensor = torch.cat(padded_batch, dim=0)  # [batch_size, seq_len]
                        # Verify batch_tensor is on correct device
                        if batch_tensor.device != device:
                            log_warning(f"⚠️  Moving batch_tensor from {batch_tensor.device} to {device}")
                            batch_tensor = batch_tensor.to(device)
                        
                        attention_mask = torch.cat(attention_masks, dim=0)
                        if attention_mask.device != device:
                            log_warning(f"⚠️  Moving attention_mask from {attention_mask.device} to {device}")
                            attention_mask = attention_mask.to(device)
                    else:
                        # Single item batch
                        batch_tensor = batch_inputs[0]
                        # Ensure single item is on correct device
                        if batch_tensor.device != device:
                            log_warning(f"⚠️  Moving single batch_tensor from {batch_tensor.device} to {device}")
                            batch_tensor = batch_tensor.to(device)
                        attention_mask = torch.ones_like(batch_tensor)
                        # Ensure attention mask is on correct device
                        if attention_mask.device != device:
                            attention_mask = attention_mask.to(device)
                        assert attention_mask.device == device, f"Attention mask must be on {device}, got {attention_mask.device}"
                    
                    # Generate for entire batch
                    generate_kwargs = {
                        "max_new_tokens": self.max_new_tokens,
                        "do_sample": self.do_sample,
                        "pad_token_id": self.tk.pad_token_id,
                        "eos_token_id": self.tk.eos_token_id,
                        "attention_mask": attention_mask,
                        "num_beams": self.num_beams,
                    }
                    
                    # early_stopping is only valid when num_beams > 1
                    if self.num_beams > 1:
                        generate_kwargs["early_stopping"] = self.early_stopping
                    
                    # Add sampling parameters
                    if self.do_sample:
                        generate_kwargs["temperature"] = self.temperature
                        generate_kwargs["top_p"] = self.top_p
                        if self.top_k > 0:
                            generate_kwargs["top_k"] = self.top_k
                    
                    if self.repetition_penalty != 1.0:
                        generate_kwargs["repetition_penalty"] = self.repetition_penalty
                    
                    # Single GPU call for entire batch
                    loop = asyncio.get_event_loop()
                    batch_outputs = await loop.run_in_executor(
                        None,
                        lambda: self._generate_sync_batch(batch_tensor, generate_kwargs)
                    )
                    
                    # batch_outputs is [batch_size, seq_len] tensor
                    # Decode and extract individual completions
                    batch_completions = []
                    batch_latencies = []
                    batch_successes = 0
                    
                    batch_latency = (time.time() - batch_start) / len(batch)
                    
                    for i in range(batch_outputs.shape[0]):
                        # Extract only new tokens (generated part) for this batch item
                        input_length = batch_lengths[i]
                        output_seq = batch_outputs[i]  # Get sequence for this batch item [seq_len]
                        
                        # Extract only new tokens (skip input tokens)
                        seq_len = output_seq.shape[0]
                        if seq_len > input_length:
                            new_tokens = output_seq[input_length:]
                        else:
                            new_tokens = output_seq
                        
                        completion_raw = self.tk.decode(new_tokens, skip_special_tokens=True)
                        
                        # Extract clean completion
                        prompt_text = batch[i].get("prompt_text", "")
                        entry_point = batch[i].get("entry_point", "")
                        completion = self.extract_completion(completion_raw, prompt_text, entry_point)
                        
                        batch_completions.append(completion)
                        batch_latencies.append(batch_latency)
                        
                        if completion and len(completion.strip()) >= 10:
                            batch_successes += 1
                    
                    return batch_completions, batch_task_ids, batch_latencies, batch_successes
                    
                except Exception as e:
                    log_error(f"⚠️  Batch {batch_idx} error: {e}")
                    # Fallback to individual processing for this batch
                    batch_completions = []
                    batch_task_ids = []
                    batch_latencies = []
                    batch_successes = 0
                    
                    for prompt_data in batch:
                        try:
                            completion, latency, success = await self.run_one(prompt_data)
                            batch_completions.append(completion)
                            batch_task_ids.append(prompt_data.get("task_id", ""))
                            batch_latencies.append(latency)
                            if success:
                                batch_successes += 1
                        except Exception as e2:
                            log_error(f"⚠️  Individual task error: {e2}")
                            batch_completions.append("")
                            batch_task_ids.append(prompt_data.get("task_id", ""))
                            batch_latencies.append(0.0)
                    
                    return batch_completions, batch_task_ids, batch_latencies, batch_successes
        
        # Process all batches concurrently (with semaphore)
        batch_results = await asyncio.gather(*[
            process_batch(batch, idx) 
            for idx, batch in enumerate(batches)
        ])
        
        # Flatten results
        for completions, task_ids, latencies, successes in batch_results:
            all_outputs.extend(completions)
            all_task_ids.extend(task_ids)
            all_latencies.extend(latencies)
            total_successes += successes
        
        total = time.time() - t0
        
        return {
            "agent_id": self.agent_id,
            "latencies": all_latencies,
            "outputs": all_outputs,
            "task_ids": all_task_ids,
            "successes": total_successes,
            "total": len(prompt_data_list),
            "runtime_s": total,
            "throughput_rps": len(all_outputs) / max(total, 0.001),
            "max_concurrent_agents": min(batch_size, len(prompt_data_list)),
            "configured_agents": self.sem._value,
            "total_processed": len(prompt_data_list),
            "batch_size": batch_size,
            "num_batches": len(batches)
        }
    
    def _generate_sync_batch(self, input_ids: torch.Tensor, generate_kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Synchronous batch generation wrapper.
        
        For sharded models (device_map="auto"), this processes the entire batch through the pipeline:
        - Batch flows through GPU0 → GPU1 → GPU2 → GPU3 sequentially
        - But all batch items process together at each stage (better than individual calls)
        """
        # Ensure input is on the correct device (first GPU for sharded models)
        device = self._get_model_device()
        
        # CRITICAL: Verify and force GPU placement
        if input_ids.device.type != 'cuda':
            log_warning(f"⚠️  WARNING: Batch input not on GPU! Device: {input_ids.device}, Moving to {device}")
            input_ids = input_ids.to(device)
        elif input_ids.device != device:
            # Device mismatch - move to correct GPU
            input_ids = input_ids.to(device)
        
        # Verify device before generation
        assert input_ids.device.type == 'cuda', f"Input must be on GPU, got {input_ids.device}"
        
        with torch.no_grad():
            # Generate for entire batch - model.generate() handles batching automatically
            # For sharded models, batch flows through pipeline sequentially
            # But all batch items process together at each GPU stage
            outputs = self.model.generate(input_ids, **generate_kwargs)
            # Return full batch tensor [batch_size, seq_len]
            return outputs
    
    async def _run_tasks_individual(self, prompt_data_list: List[Dict[str, Any]], use_multi_sample: bool) -> Dict[str, Any]:
        """Process tasks individually (original implementation for multi-sample or batch_size=1)."""
        t0 = time.time()
        latencies, outputs, task_ids, successes = [], [], [], 0
        
        # Track active agents
        active_agents = 0
        max_active_agents = 0
        total_processed = 0
        
        async def process_task(prompt_data: Dict[str, Any], task_idx: int):
            nonlocal successes, active_agents, max_active_agents, total_processed
            
            # GPU throttling: wait if all GPUs are near capacity
            while not check_gpu_availability(threshold=0.9):
                log_warning(f"⚠️  All GPUs near capacity (90%+), waiting for free memory... (Agent {self.agent_id}, Task {task_idx})")
                await asyncio.sleep(5)
            
            async with self.sem:
                active_agents += 1
                max_active_agents = max(max_active_agents, active_agents)
                total_processed += 1
                
                # Print progress every 10 tasks (only if verbose)
                if total_processed % 10 == 0:
                    log_info(f"  Agent {self.agent_id} Progress: {total_processed}/{len(prompt_data_list)} tasks | "
                          f"Active: {active_agents}/{self.sem._value} | "
                          f"Max concurrent: {max_active_agents}")
                
                if use_multi_sample and self.num_samples > 1:
                    # Multi-sample generation
                    completions, latency, success = await self.run_one_multi_sample(prompt_data)
                    for comp in completions:
                        outputs.append(comp)
                        task_ids.append(prompt_data.get("task_id", ""))
                        latencies.append(latency)
                    if success:
                        successes += len(completions)
                else:
                    # Single sample with optional reflection
                    completion, latency, success = await self.run_one(prompt_data)
                    outputs.append(completion)
                    task_ids.append(prompt_data.get("task_id", ""))
                    latencies.append(latency)
                    if success:
                        successes += 1
                    
                    # Reflection: if first attempt failed and reflection enabled, retry with corrective prompt
                    if not success and self.enable_reflection:
                        reflection_prompt = self.create_reflection_prompt(prompt_data)
                        completion_reflect, latency_reflect, success_reflect = await self.run_one(reflection_prompt, is_reflection=True)
                        if success_reflect and completion_reflect:
                            outputs[-1] = completion_reflect
                            latencies[-1] = latency_reflect
                            successes += 1
                
                # Decrement active agents counter
                active_agents -= 1

        # Start ALL tasks concurrently - let semaphore control concurrency
        log_info(f"\n🚀 Agent {self.agent_id}: Starting {len(prompt_data_list)} tasks with {self.sem._value} concurrent operations...")
        log_info(f"   Dynamic GPU assignment enabled - agents will be balanced across GPUs")
        await asyncio.gather(*[process_task(pd, idx) for idx, pd in enumerate(prompt_data_list)])

        total = time.time() - t0
        
        return {
            "agent_id": self.agent_id,
            "latencies": latencies,
            "outputs": outputs,
            "task_ids": task_ids,
            "successes": successes,
            "total": len(prompt_data_list),
            "runtime_s": total,
            "throughput_rps": len(outputs) / max(total, 0.001),
            "max_concurrent_agents": max_active_agents,
            "configured_agents": self.sem._value,
            "total_processed": total_processed
        }
