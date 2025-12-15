"""
ReAct Agent Framework built on top of the existing AgentManager.

This module introduces a lightweight reasoning layer that wraps AgentManager's
GPU-optimized generation routines without altering their internal behaviour.
The ReActAgent class executes the Reasoning → Action → Observation → Reflection
cycle while streaming detailed traces to a JSONL log file for post-run analysis.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from utils import log_info, log_success, log_warning


class ReActAgent:
    """
    Orchestrates the ReAct reasoning loop using the existing AgentManager.

    Each call performs:
    1. Reasoning: lightweight chain-of-thought prompt recorded for transparency.
    2. Action: defers to AgentManager.run_one which handles GPU inference,
       batching, and semaphore throttling.
    3. Observation: inspects the completion and determines success criteria.
    4. Reflection: optional corrective call using AgentManager.create_reflection_prompt
       when the first attempt fails, mirroring the project's reflection workflow.
    """

    def __init__(
        self,
        manager: Any,
        log_path: str | Path = "react_log.jsonl",
        enable_reflection: Optional[bool] = None,
        max_reflections: int = 1,
    ):
        self.manager = manager
        self.log_path = Path(log_path)
        self.enable_reflection = (
            manager.enable_reflection if enable_reflection is None else enable_reflection
        )
        self.max_reflections = max(0, int(max_reflections))
        self._log_lock = asyncio.Lock()

        # Ensure the directory exists to avoid IO errors during first write.
        if self.log_path.parent and not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full ReAct cycle for a single task.

        Returns a result dictionary containing the final completion, latency
        metrics, and bookkeeping flags. The actual GPU-bound generation remains
        delegated to AgentManager to preserve performance characteristics.
        """
        task_id = task.get("task_id", "unknown-task")
        prompt_snippet = task.get("prompt_text", "")[:120]
        reasoning_prompt = f"Let's reason step by step to solve: {prompt_snippet}"

        self._console_log("Reasoning", task_id, reasoning_prompt)
        await self._record_trace(
            task_id,
            stage="reasoning",
            payload={"reasoning": reasoning_prompt},
        )

        completion, latency, success = await self.manager.run_one(task)
        self._console_log(
            "Action",
            task_id,
            f"Completed in {latency:.2f}s | success={success}",
        )
        await self._record_trace(
            task_id,
            stage="action",
            payload={
                "latency_s": latency,
                "success": success,
                "completion_preview": completion[:160],
            },
        )

        total_latency = latency
        reflection_attempts = 0
        reflection_completion = ""
        observation_message = ""

        if success:
            observation_message = "Result appears valid on first attempt."
            self._console_log("Observation", task_id, observation_message)
            await self._record_trace(
                task_id,
                stage="observation",
                payload={"message": observation_message, "need_reflection": False},
            )
            return {
                "task_id": task_id,
                "completion": completion,
                "success": True,
                "latency_s": total_latency,
                "reflection_attempts": reflection_attempts,
                "reflection_completion": reflection_completion,
            }

        observation_message = "Initial output failed checks; preparing reflection."
        self._console_log("Observation", task_id, observation_message)
        await self._record_trace(
            task_id,
            stage="observation",
            payload={"message": observation_message, "need_reflection": True},
        )

        if not self.enable_reflection or self.max_reflections <= 0:
            self._console_log(
                "Reflection",
                task_id,
                "Reflection disabled. Returning empty completion.",
            )
            await self._record_trace(
                task_id,
                stage="reflection",
                payload={
                    "attempt": 0,
                    "enabled": False,
                    "completion_preview": "",
                    "success": False,
                },
            )
            return {
                "task_id": task_id,
                "completion": "",
                "success": False,
                "latency_s": total_latency,
                "reflection_attempts": reflection_attempts,
                "reflection_completion": reflection_completion,
            }

        current_task = task
        reflection_success = False

        while reflection_attempts < self.max_reflections and not reflection_success:
            reflection_attempts += 1
            reflection_task = self.manager.create_reflection_prompt(current_task)
            reflection_task["task_id"] = f"{task_id}::reflection-{reflection_attempts}"

            self._console_log(
                "Reflection",
                task_id,
                f"Attempt {reflection_attempts}: dispatching corrective prompt.",
            )
            reflection_completion, reflection_latency, reflection_success = await self.manager.run_one(
                reflection_task, is_reflection=True
            )
            total_latency += reflection_latency

            self._console_log(
                "Reflection",
                task_id,
                f"Completed in {reflection_latency:.2f}s | success={reflection_success}",
            )
            await self._record_trace(
                task_id,
                stage="reflection",
                payload={
                    "attempt": reflection_attempts,
                    "latency_s": reflection_latency,
                    "success": reflection_success,
                    "completion_preview": reflection_completion[:160],
                },
            )

            current_task = reflection_task  # Allows iterative refinement if needed.

        final_completion = reflection_completion if reflection_success else ""
        final_success = reflection_success

        if final_success:
            log_success(f"[ReAct][{task_id}] Reflection succeeded.")
        else:
            log_warning(f"[ReAct][{task_id}] Reflection exhausted without success.")

        return {
            "task_id": task_id,
            "completion": final_completion,
            "success": final_success,
            "latency_s": total_latency,
            "reflection_attempts": reflection_attempts,
            "reflection_completion": reflection_completion,
        }

    async def run_batch(self, tasks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks concurrently while respecting AgentManager's
        internal concurrency controls. The outer gather ensures we saturate the
        event loop whereas GPU contention remains regulated by AgentManager.
        """
        tasks_list = list(tasks)
        if not tasks_list:
            return []
        self._console_log(
            "Batch",
            "react",
            f"Launching {len(tasks_list)} tasks via ReAct pipeline.",
        )
        return list(await asyncio.gather(*(self.run_task(task) for task in tasks_list)))

    async def _record_trace(
        self,
        task_id: str,
        stage: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Append a structured trace entry to the JSONL log.

        The write path uses asyncio.to_thread to avoid blocking the event loop
        during filesystem IO, while a module-level lock serialises writers.
        """
        entry = {
            "timestamp": time.time(),
            "task_id": task_id,
            "stage": stage,
            "payload": payload,
        }
        line = json.dumps(entry, ensure_ascii=True)

        async with self._log_lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _console_log(self, step: str, task_id: str, message: str) -> None:
        """
        Mirror the reasoning trace to stdout and project log utilities.
        """
        formatted = f"[{step}] ({task_id}) {message}"
        print(formatted)
        log_info(formatted)


async def run_react_pipeline(
    task_list: Iterable[Dict[str, Any]],
    agent: ReActAgent,
) -> List[Dict[str, Any]]:
    """
    Run a collection of tasks through a shared ReActAgent and return all results.

    The orchestrator relies on asyncio.gather to launch all ReAct cycles in
    parallel. AgentManager's semaphore and batching logic remain unchanged,
    ensuring kernel-level performance parity with existing execution paths.
    """
    tasks_list = list(task_list)
    if not tasks_list:
        return []

    log_info(f"[ReAct] Starting pipeline for {len(tasks_list)} tasks.")
    results = await asyncio.gather(*(agent.run_task(task) for task in tasks_list))
    log_info(f"[ReAct] Completed pipeline for {len(tasks_list)} tasks.")
    return list(results)


__all__ = ["ReActAgent", "run_react_pipeline"]

