# agent_manager.py
import asyncio
import time
import re
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from gpu_assigner import check_gpu_availability
from utils import log_info, log_warning, log_error


class AgentManager:
    """
    Agent manager for async execution + optional batching.

    Key guarantees:
    - run_tasks() exists (expected by agent_concurrency.py / main.py)
    - attention_mask is ALWAYS provided to generate()
    - batching pads with pad_token_id and uses correct attention masks
    - on CUDA failure, returns empty completion (does not inject error text into code)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        cfg: Dict[str, Any],
        agent_id: int = 0,
    ):
        self.tk = tokenizer
        self.model = model
        self.cfg = cfg
        self.agent_id = agent_id

        # config
        self.batch_size = int(cfg.get("batch_size", 8))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 512))
        self.temperature = float(cfg.get("temperature", 0.35))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.top_k = int(cfg.get("top_k", 0))  # 0 = disabled
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.0))
        self.do_sample = bool(cfg.get("do_sample", True))
        self.num_beams = int(cfg.get("num_beams", 1))
        self.early_stopping = bool(cfg.get("early_stopping", True))
        self.max_retries = int(cfg.get("max_retries", 2))
        self.num_samples = int(cfg.get("num_samples", 1))
        self.enable_reflection = bool(cfg.get("enable_reflection", False))

        # Ensure pad token is set (important for batching + stable generation)
        if self.tk.pad_token is None:
            self.tk.pad_token = self.tk.eos_token
        if self.tk.pad_token_id is None and self.tk.pad_token is not None:
            self.tk.pad_token_id = self.tk.convert_tokens_to_ids(self.tk.pad_token)

        # Concurrency control
        base_semaphore = int(cfg.get("num_agents", 100))
        batch_size = self.batch_size

        if batch_size > 1:
            # allow multiple batches concurrently but cap to avoid memory pressure
            scaled = min(20, max(4, batch_size * 2))
        else:
            scaled = base_semaphore

        self.sem = asyncio.Semaphore(scaled)
        log_info(
            f"🔧 Agent {agent_id} semaphore set to {scaled} (batch_size={batch_size}, base={base_semaphore})"
        )

    # ----------------------------
    # Device helpers
    # ----------------------------
    def _get_model_device(self) -> torch.device:
        """
        Determines where inputs should be placed.
        - If hf_device_map exists, pick the first CUDA device used.
        - Else use the first parameter device.
        """
        if hasattr(self.model, "hf_device_map") and getattr(self.model, "hf_device_map", None):
            vals = list(self.model.hf_device_map.values())
            gpu_ids = []
            for v in vals:
                if isinstance(v, int) and v >= 0:
                    gpu_ids.append(v)
                elif isinstance(v, str):
                    s = v.lower()
                    if s.startswith("cuda:"):
                        try:
                            gpu_ids.append(int(s.split(":")[-1]))
                        except Exception:
                            pass
                    elif s.isdigit():
                        gpu_ids.append(int(s))
            if gpu_ids and torch.cuda.is_available():
                return torch.device(f"cuda:{min(gpu_ids)}")
            return torch.device("cpu")

        # non-sharded: use first param device
        try:
            p = next(iter(self.model.parameters()))
            return p.device
        except Exception:
            return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # ----------------------------
    # Prompt formatting
    # ----------------------------
    def format_prompt_for_generation(self, prompt_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
        - input_ids  [1, seq_len] on correct device
        - attention_mask [1, seq_len] on correct device
        - input_length (seq_len)
        """
        device = self._get_model_device()
        system_msg = prompt_data.get(
            "system_msg",
            "You are an expert Python coding assistant. Write correct, executable functions.",
        )
        user_msg = prompt_data.get("user_msg", prompt_data.get("prompt_text", ""))

        # Try chat template if available
        if hasattr(self.tk, "apply_chat_template") and getattr(self.tk, "chat_template", None) is not None:
            try:
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                input_ids = self.tk.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                input_ids = input_ids.to(device)

                attention_mask = torch.ones_like(input_ids, device=device)
                return input_ids, attention_mask, int(input_ids.shape[1])
            except Exception as e:
                log_warning(f"⚠️  Chat template failed, falling back to raw prompt: {e}")

        # Fallback: raw tokenize
        prompt_text = user_msg
        inputs = self.tk(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.cfg.get("max_input_tokens", 2048)),
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else torch.ones_like(input_ids, device=device)
        return input_ids, attention_mask, int(input_ids.shape[1])

    # ----------------------------
    # Generation wrappers
    # ----------------------------
    def _build_generate_kwargs(self, attention_mask: torch.Tensor) -> Dict[str, Any]:
        kw: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tk.pad_token_id,
            "eos_token_id": self.tk.eos_token_id,
            "attention_mask": attention_mask,
            "num_beams": self.num_beams,
        }

        if self.num_beams > 1:
            kw["early_stopping"] = self.early_stopping

        if self.do_sample:
            kw["temperature"] = self.temperature
            kw["top_p"] = self.top_p
            if self.top_k > 0:
                kw["top_k"] = self.top_k

        if self.repetition_penalty and self.repetition_penalty != 1.0:
            kw["repetition_penalty"] = self.repetition_penalty

        return kw

    def _generate_sync(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = self._get_model_device()
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        generate_kwargs = self._build_generate_kwargs(attention_mask)

        with torch.no_grad():
            return self.model.generate(input_ids, **generate_kwargs)

    def _generate_sync_batch(self, batch_input_ids: torch.Tensor, batch_attention_mask: torch.Tensor) -> torch.Tensor:
        device = self._get_model_device()
        if batch_input_ids.device != device:
            batch_input_ids = batch_input_ids.to(device)
        if batch_attention_mask.device != device:
            batch_attention_mask = batch_attention_mask.to(device)

        generate_kwargs = self._build_generate_kwargs(batch_attention_mask)

        with torch.no_grad():
            return self.model.generate(batch_input_ids, **generate_kwargs)

    # ----------------------------
    # Completion extraction (your version, kept but safer)
    # ----------------------------
    def extract_completion(self, full_output: str, prompt_text: str, entry_point: str) -> str:
        completion = (full_output or "").strip()

        # remove code fences
        completion = re.sub(r"^```python\s*", "", completion, flags=re.IGNORECASE)
        completion = re.sub(r"^```\s*", "", completion)
        completion = re.sub(r"```$", "", completion).strip()

        # remove common prefaces
        completion = re.sub(
            r"^(Here\s+is|Here\'s|The\s+function|The\s+implementation|Implementation)[:.]?\s*",
            "",
            completion,
            flags=re.IGNORECASE,
        ).strip()

        # If model re-prints def entry_point, strip everything before body
        if entry_point and f"def {entry_point}" in completion:
            after = completion.split(f"def {entry_point}", 1)[1]
            lines = after.split("\n")

            body_lines = []
            in_doc = False
            for line in lines:
                s = line.strip()
                if s.startswith('"""') or s.startswith("'''"):
                    if not in_doc:
                        in_doc = True
                        # one-line docstring
                        if s.count('"""') == 2 or s.count("'''") == 2:
                            in_doc = False
                        continue
                    else:
                        in_doc = False
                        continue
                if in_doc:
                    continue
                body_lines.append(line)
            completion = "\n".join(body_lines).strip()

        # drop import/def/class lines at start
        cleaned = []
        for line in completion.split("\n"):
            s = line.strip()
            if s.startswith(("from ", "import ", "def ", "class ")):
                continue
            cleaned.append(line)
        completion = "\n".join(cleaned).strip()

        # remove trailing “explanation” lines
        out_lines = []
        for line in completion.split("\n"):
            s = line.strip()
            if s.startswith("```"):
                break
            if re.match(r"^(This|The|It|Here|Note|Explanation)", s, flags=re.IGNORECASE):
                break
            out_lines.append(line)
        completion = "\n".join(out_lines).rstrip()

        if not completion.strip():
            return ""

        # Ensure function-body indentation (HumanEval expects 4-space indentation)
        lines = [ln.rstrip() for ln in completion.split("\n") if ln.strip() != ""]
        if not lines:
            return ""

        # Normalize minimum indentation to at least 4 spaces
        min_indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
        normalized = []
        for ln in completion.split("\n"):
            if not ln.strip():
                continue
            curr = len(ln) - len(ln.lstrip())
            new_indent = max(4, curr - min_indent + 4)
            normalized.append((" " * new_indent) + ln.lstrip())

        return "\n".join(normalized).rstrip()

    # ----------------------------
    # Reflection prompt (optional)
    # ----------------------------
    def create_reflection_prompt(self, prompt_data: Dict[str, Any], error_msg: str = "") -> Dict[str, Any]:
        prompt_text = prompt_data.get("prompt_text", "")
        reflection_msg = f"""You are an expert Python developer.

Complete the following function so that it passes all test cases.

{prompt_text}

# Write your solution below:

The previous attempt had issues. Please write a corrected solution.
Requirements:
- Start each line with 4 spaces for the function body
- Use 8 spaces for nested blocks (inside if/for/while)
- Do NOT include the function signature, imports, or docstring
- Write complete, executable code that solves the problem
- Handle edge cases carefully
"""
        new_pd = dict(prompt_data)
        new_pd["user_msg"] = reflection_msg
        return new_pd

    # ----------------------------
    # Single task execution
    # ----------------------------
    async def run_one(
        self,
        prompt_data: Dict[str, Any],
        retry_count: int = 0,
        is_reflection: bool = False,
    ) -> Tuple[str, float, bool]:
        start = time.time()

        try:
            input_ids, attention_mask, input_len = self.format_prompt_for_generation(prompt_data)

            loop = asyncio.get_event_loop()
            output_ids = await loop.run_in_executor(
                None,
                lambda: self._generate_sync(input_ids, attention_mask),
            )

            new_tokens = output_ids[0, input_len:]
            completion_raw = self.tk.decode(new_tokens, skip_special_tokens=True)

            completion = self.extract_completion(
                completion_raw,
                prompt_data.get("prompt_text", ""),
                prompt_data.get("entry_point", ""),
            )

            latency = time.time() - start

            if not completion or len(completion.strip()) < 10:
                if retry_count < self.max_retries:
                    await asyncio.sleep(0.1)
                    return await self.run_one(prompt_data, retry_count + 1, is_reflection)
                return ("", latency, False)

            return (completion, latency, True)

        except RuntimeError as e:
            # Common CUDA failure: device-side assert triggered
            latency = time.time() - start
            log_error(f"⚠️  Agent {self.agent_id} generation RuntimeError: {e}")

            # IMPORTANT: don't return error text as completion (breaks HumanEval syntax)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if retry_count < self.max_retries:
                await asyncio.sleep(0.2)
                return await self.run_one(prompt_data, retry_count + 1, is_reflection)

            return ("", latency, False)

        except Exception as e:
            latency = time.time() - start
            log_error(f"⚠️  Agent {self.agent_id} generation error: {e}")

            if retry_count < self.max_retries:
                await asyncio.sleep(0.2)
                return await self.run_one(prompt_data, retry_count + 1, is_reflection)

            return ("", latency, False)

    async def run_one_multi_sample(self, prompt_data: Dict[str, Any]) -> Tuple[List[str], float, bool]:
        start = time.time()
        completions: List[str] = []

        for _ in range(self.num_samples):
            comp, _, ok = await self.run_one(prompt_data, retry_count=0)
            if ok and comp:
                completions.append(comp)
            await asyncio.sleep(0.05)

        avg_latency = (time.time() - start) / max(1, len(completions))
        return (completions, avg_latency, len(completions) > 0)

    # ----------------------------
    # Public entrypoint (FIXES your "no attribute run_tasks")
    # ----------------------------
    async def run_tasks(
        self,
        prompt_data_list: List[Dict[str, Any]],
        use_multi_sample: bool = False,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        use_batching = (batch_size > 1) and (not use_multi_sample)
        if use_batching:
            return await self._run_tasks_batched(prompt_data_list, batch_size)
        return await self._run_tasks_individual(prompt_data_list, use_multi_sample)

    # ----------------------------
    # Batched execution
    # ----------------------------
    async def _run_tasks_batched(self, prompt_data_list: List[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
        t0 = time.time()
        all_outputs: List[str] = []
        all_task_ids: List[str] = []
        all_latencies: List[float] = []
        total_successes = 0

        batches = [prompt_data_list[i : i + batch_size] for i in range(0, len(prompt_data_list), batch_size)]
        log_info(f"\n🚀 Agent {self.agent_id}: Processing {len(prompt_data_list)} tasks in {len(batches)} batches (batch_size={batch_size})...")

        async def process_batch(batch: List[Dict[str, Any]], batch_idx: int):
            batch_start = time.time()

            while not check_gpu_availability(threshold=0.9):
                log_warning(f"⚠️  All GPUs near capacity (90%+), waiting... (Agent {self.agent_id}, Batch {batch_idx})")
                await asyncio.sleep(5)

            async with self.sem:
                device = self._get_model_device()
                pad_id = int(self.tk.pad_token_id)

                # tokenize each item
                item_input_ids = []
                item_attention = []
                item_lengths = []
                item_task_ids = []

                for pd in batch:
                    in_ids, attn, in_len = self.format_prompt_for_generation(pd)
                    item_input_ids.append(in_ids)
                    item_attention.append(attn)
                    item_lengths.append(in_len)
                    item_task_ids.append(pd.get("task_id", ""))

                # Pad to max length
                max_len = max(x.shape[1] for x in item_input_ids)
                padded_ids = []
                padded_attn = []

                for in_ids, attn in zip(item_input_ids, item_attention):
                    seq_len = in_ids.shape[1]
                    pad_len = max_len - seq_len

                    if pad_len > 0:
                        pad_tensor = torch.full((1, pad_len), pad_id, dtype=in_ids.dtype, device=device)
                        in_ids = torch.cat([in_ids.to(device), pad_tensor], dim=1)

                        pad_mask = torch.zeros((1, pad_len), dtype=attn.dtype, device=device)
                        attn = torch.cat([attn.to(device), pad_mask], dim=1)
                    else:
                        in_ids = in_ids.to(device)
                        attn = attn.to(device)

                    padded_ids.append(in_ids)
                    padded_attn.append(attn)

                batch_input_ids = torch.cat(padded_ids, dim=0).to(device)
                batch_attention_mask = torch.cat(padded_attn, dim=0).to(device)

                loop = asyncio.get_event_loop()
                try:
                    batch_outputs = await loop.run_in_executor(
                        None,
                        lambda: self._generate_sync_batch(batch_input_ids, batch_attention_mask),
                    )
                except Exception as e:
                    log_error(f"⚠️  Batch {batch_idx} error: {e}")
                    # fall back to individual for this batch
                    outs, tids, lats, succ = [], [], [], 0
                    for pd in batch:
                        c, l, ok = await self.run_one(pd)
                        outs.append(c)
                        tids.append(pd.get("task_id", ""))
                        lats.append(l)
                        succ += 1 if ok else 0
                    return outs, tids, lats, succ

                per_item_latency = (time.time() - batch_start) / max(1, len(batch))
                outs, lats, succ = [], [], 0

                for i in range(batch_outputs.shape[0]):
                    out_seq = batch_outputs[i]
                    in_len = item_lengths[i]
                    new_tokens = out_seq[in_len:] if out_seq.shape[0] > in_len else out_seq
                    completion_raw = self.tk.decode(new_tokens, skip_special_tokens=True)

                    completion = self.extract_completion(
                        completion_raw,
                        batch[i].get("prompt_text", ""),
                        batch[i].get("entry_point", ""),
                    )

                    outs.append(completion)
                    lats.append(per_item_latency)
                    if completion and len(completion.strip()) >= 10:
                        succ += 1

                return outs, item_task_ids, lats, succ

        batch_results = await asyncio.gather(*[process_batch(b, idx) for idx, b in enumerate(batches)])

        for outs, tids, lats, succ in batch_results:
            all_outputs.extend(outs)
            all_task_ids.extend(tids)
            all_latencies.extend(lats)
            total_successes += succ

        total = time.time() - t0
        return {
            "agent_id": self.agent_id,
            "latencies": all_latencies,
            "outputs": all_outputs,
            "task_ids": all_task_ids,
            "successes": total_successes,
            "total": len(prompt_data_list),
            "runtime_s": total,
            "throughput_rps": len(all_outputs) / max(total, 1e-6),
            "max_concurrent_agents": min(batch_size, len(prompt_data_list)),
            "configured_agents": getattr(self.sem, "_value", None),
            "total_processed": len(prompt_data_list),
            "batch_size": batch_size,
            "num_batches": len(batches),
        }

    # ----------------------------
    # Individual execution
    # ----------------------------
    async def _run_tasks_individual(self, prompt_data_list: List[Dict[str, Any]], use_multi_sample: bool) -> Dict[str, Any]:
        t0 = time.time()
        latencies: List[float] = []
        outputs: List[str] = []
        task_ids: List[str] = []
        successes = 0

        active = 0
        max_active = 0
        processed = 0

        async def process_task(pd: Dict[str, Any], idx: int):
            nonlocal successes, active, max_active, processed

            while not check_gpu_availability(threshold=0.9):
                log_warning(f"⚠️  All GPUs near capacity (90%+), waiting... (Agent {self.agent_id}, Task {idx})")
                await asyncio.sleep(5)

            async with self.sem:
                active += 1
                max_active = max(max_active, active)
                processed += 1

                if processed % 10 == 0:
                    log_info(f"  Agent {self.agent_id} Progress: {processed}/{len(prompt_data_list)} | Active: {active} | Max: {max_active}")

                if use_multi_sample and self.num_samples > 1:
                    comps, latency, ok = await self.run_one_multi_sample(pd)
                    for c in comps:
                        outputs.append(c)
                        task_ids.append(pd.get("task_id", ""))
                        latencies.append(latency)
                    if ok:
                        successes += len(comps)
                else:
                    c, latency, ok = await self.run_one(pd)
                    outputs.append(c)
                    task_ids.append(pd.get("task_id", ""))
                    latencies.append(latency)
                    successes += 1 if ok else 0

                    if (not ok) and self.enable_reflection:
                        rp = self.create_reflection_prompt(pd)
                        c2, l2, ok2 = await self.run_one(rp, is_reflection=True)
                        if ok2 and c2:
                            outputs[-1] = c2
                            latencies[-1] = l2
                            successes += 1

                active -= 1

        log_info(f"\n🚀 Agent {self.agent_id}: Starting {len(prompt_data_list)} tasks with concurrency={getattr(self.sem, '_value', None)}")
        await asyncio.gather(*[process_task(pd, i) for i, pd in enumerate(prompt_data_list)])

        total = time.time() - t0
        return {
            "agent_id": self.agent_id,
            "latencies": latencies,
            "outputs": outputs,
            "task_ids": task_ids,
            "successes": successes,
            "total": len(prompt_data_list),
            "runtime_s": total,
            "throughput_rps": len(outputs) / max(total, 1e-6),
            "max_concurrent_agents": max_active,
            "configured_agents": getattr(self.sem, "_value", None),
            "total_processed": processed,
        }
