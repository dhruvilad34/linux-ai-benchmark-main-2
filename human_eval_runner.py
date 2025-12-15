# human_eval_runner.py
from typing import List, Dict, Any, Optional
import json, os
import textwrap
import ast
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

# Optional black formatter
try:
    import black
    BLACK_AVAILABLE = True
except ImportError:
    BLACK_AVAILABLE = False
    black = None

class HumanEvalRunner:
    """
    Loads and runs HumanEval tasks using the official OpenAI HumanEval repository.
    """
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.limit = int(cfg.get("humaneval_limit", 100))

    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        Load HumanEval tasks and return as list of task dictionaries.
        Each task contains: task_id, prompt, entry_point, canonical_solution, test
        """
        problems = read_problems()
        tasks = list(problems.values())
        # Limit to specified number of tasks
        if self.limit:
            tasks = tasks[:self.limit]
        return tasks

    def extract_prompts(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract prompt strings from HumanEval tasks and format for LLaMA-3.1-Instruct.
        Returns list of dicts with 'prompt', 'task_id', and 'entry_point' for proper parsing.
        """
        formatted_prompts = []
        for task in tasks:
            task_id = task.get("task_id", "")
            prompt_raw = task.get("prompt", "")
            entry_point = task.get("entry_point", "")
            
            # Enhanced prompt template for better accuracy
            system_msg = "You are an expert Python developer specializing in writing correct, efficient, and test-passing code."
            
            # Improved user prompt with clear instructions
            user_msg = f"""You are an expert Python developer.

Complete the following function so that it passes all test cases.

{prompt_raw}

# Write your solution below:

Write ONLY the function body code (the indented code that goes inside the function). 
Requirements:
- Start each line with 4 spaces for the function body
- Use 8 spaces for nested blocks (inside if/for/while)
- Use 12 spaces for nested blocks inside nested blocks
- Do NOT include the function signature, imports, or docstring
- Write complete, executable code that solves the problem"""
            
            formatted_prompts.append({
                "prompt_text": prompt_raw,  # Keep original for token counting
                "task_id": task_id,
                "entry_point": entry_point,
                "system_msg": system_msg,
                "user_msg": user_msg,
                "full_task": task  # Keep full task for later use
            })
        return formatted_prompts

    def score_dummy(self, outputs: List[str]) -> Dict[str, Any]:
        """
        Stub scoring; for full evaluation, use evaluate_functional_correctness
        """
        passed = sum(1 for o in outputs if "def " in o or "return" in o)
        return {"pass_like": passed, "total": len(outputs), "rate_est": passed/max(1,len(outputs))}
    
    def sanitize_completion(self, completion: str, task_id: str, use_black: bool = False) -> Optional[str]:
        """
        Sanitize and validate a completion.
        
        1. Uses textwrap.dedent() to normalize indentation
        2. Automatically adds a def header if completion doesn't start with 'def '
        3. Optionally formats with black
        4. Runs ast.parse() to validate syntax
        5. Returns None for invalid completions instead of crashing
        
        Args:
            completion: The code completion string to sanitize
            task_id: Task identifier for error logging
            use_black: Whether to format code with black (default: False)
            
        Returns:
            Sanitized completion string, or None if invalid
        """
        if not completion or not completion.strip():
            return None
        
        try:
            # Step 1: Normalize indentation using textwrap.dedent()
            completion_stripped = completion.strip()
            normalized = textwrap.dedent(completion_stripped)
            
            # Step 2: Automatically add def header if completion doesn't start with 'def '
            # For HumanEval, completions are function bodies, so we wrap them
            if not normalized.strip().startswith('def '):
                # Check if it looks like a function body (has indentation)
                lines = normalized.split('\n')
                if lines and lines[0].startswith((' ', '\t')):
                    # Has indentation - it's a function body, wrap it
                    # Remove common leading whitespace and add function wrapper
                    # Find minimum indentation
                    min_indent = min(
                        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
                        default=0
                    )
                    # Remove common indent
                    dedented_lines = [line[min_indent:] if line.strip() else line for line in lines]
                    body = '\n'.join(dedented_lines)
                    # Wrap in function with proper indentation
                    sanitized = f"def _completion():\n    {body.replace(chr(10), chr(10) + '    ')}"
                else:
                    # No indentation - might be top-level code, wrap it
                    sanitized = f"def _completion():\n    {normalized.replace(chr(10), chr(10) + '    ')}"
            else:
                # Already has def - use as is but ensure it's valid
                sanitized = normalized
            
            # Step 3: Optionally format with black
            if use_black and BLACK_AVAILABLE:
                try:
                    # Format with black (mode=black.Mode())
                    sanitized = black.format_str(sanitized, mode=black.Mode())
                except Exception as e:
                    # If black fails, continue with unformatted code
                    pass
            
            # Step 4: Run ast.parse() to validate syntax
            try:
                ast.parse(sanitized)
                # If validation passed, return the sanitized code
                # For HumanEval, we need to extract just the function body
                # If we wrapped it, extract the body part
                if sanitized.startswith('def _completion():\n'):
                    # Extract the body (everything after the function definition)
                    body_lines = sanitized.split('\n')[1:]  # Skip def line
                    # Remove the wrapper indentation (4 spaces) from each line
                    extracted_body = []
                    for line in body_lines:
                        if line.startswith('    '):
                            extracted_body.append(line[4:])
                        elif line.strip() == '':
                            extracted_body.append('')
                        else:
                            # Line doesn't have expected indentation, keep as-is
                            extracted_body.append(line)
                    body = '\n'.join(extracted_body)
                    # Ensure it has proper base indentation for HumanEval (4 spaces)
                    if body and not body.startswith(' '):
                        # Add base indentation
                        body = '    ' + body.replace('\n', '\n    ')
                    return body.strip()
                else:
                    # Return as-is if it was already a function
                    return sanitized
            except SyntaxError as e:
                # Step 5: Return None for invalid completions instead of crashing
                error_msg = f"SyntaxError: {e.msg}"
                if e.lineno:
                    error_msg += f" on line {e.lineno}"
                if e.text:
                    error_msg += f" ({e.text.strip()})"
                print(f"❌ Syntax error in {task_id}: {error_msg}")
                return None
                
        except Exception as e:
            # Catch any other errors and return None
            print(f"❌ Error sanitizing {task_id}: {str(e)}")
            return None
    
    def validate_completion_syntax(self, completion: str, task_id: str) -> tuple[bool, str]:
        """
        Validate that a completion is syntactically valid Python code.
        
        HumanEval completions are function bodies that should be valid when inserted
        into a function. We validate by wrapping in a function context.
        
        Args:
            completion: The code completion string to validate
            task_id: Task identifier for error logging
            
        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        if not completion or not completion.strip():
            return False, "Empty completion"
        
        try:
            # Step 1: Normalize using textwrap.dedent() to handle inconsistent indentation
            # This removes common leading whitespace while preserving relative structure
            completion_stripped = completion.strip()
            
            # Step 2: Normalize indentation of the completion itself
            # Use dedent to remove common leading whitespace
            try:
                normalized_completion = textwrap.dedent(completion_stripped)
                # If dedent removed all indentation, the completion might be at top level
                # For HumanEval, we expect function body code (should be indented)
                # But we'll try both ways
                if not normalized_completion.startswith((' ', '\t')):
                    # No indentation after dedent - try with base indentation
                    normalized_completion = '    ' + normalized_completion.replace('\n', '\n    ')
            except:
                normalized_completion = completion_stripped
            
            # Step 3: For HumanEval, completions are function bodies
            # Wrap in a dummy function to validate syntax in proper context
            # The completion should already be indented (4 spaces for function body)
            # So we just add the function definition above it
            wrapped_code = f"def _validate_completion():\n{normalized_completion}"
            
            # Step 4: Parse with ast.parse() to catch syntax errors
            try:
                ast.parse(wrapped_code)
                return True, ""
            except SyntaxError as e:
                # Extract clear error message
                # Adjust line number (subtract 1 for the wrapper function line)
                error_msg = f"SyntaxError: {e.msg}"
                if e.lineno and e.lineno > 1:
                    # Line number in completion (not including wrapper)
                    completion_line = e.lineno - 1
                    error_msg += f" on line {completion_line}"
                elif e.lineno:
                    error_msg += f" on line {e.lineno}"
                if e.text:
                    error_msg += f" ({e.text.strip()})"
                return False, error_msg
                
        except Exception as e:
            # Catch any other parsing errors
            return False, f"Parse error: {str(e)}"
    
    def prevalidate_completions(self, results_file: str, use_black: bool = False) -> tuple[str, int]:
        """
        Pre-validate and sanitize all completions in the results file.
        Uses sanitize_completion() to clean and validate code.
        Creates a new file with sanitized completions.
        
        Args:
            results_file: Path to input JSONL file
            use_black: Whether to format code with black (default: False)
            
        Returns:
            Tuple of (validated_file_path: str, invalid_count: int)
        """
        from human_eval.data import stream_jsonl, write_jsonl
        
        validated_file = results_file.replace('.jsonl', '_validated.jsonl')
        invalid_count = 0
        valid_samples = []
        
        print(f"\n🔍 Sanitizing and validating completions...")
        if use_black and BLACK_AVAILABLE:
            print("   Using black formatter")
        elif use_black:
            print("   ⚠️  black not available, skipping formatting")
        
        for sample in stream_jsonl(results_file):
            task_id = sample.get("task_id", "unknown")
            completion = sample.get("completion", "")
            
            # Sanitize completion (returns None if invalid)
            sanitized = self.sanitize_completion(completion, task_id, use_black=use_black)
            
            if sanitized is not None:
                # Valid and sanitized - update completion
                sample["completion"] = sanitized
                valid_samples.append(sample)
            else:
                # Invalid - mark as failed
                invalid_count += 1
                # Mark as failed but keep in results for tracking
                sample["syntax_error"] = "syntax_error"
                sample["passed"] = False
                sample["result"] = "syntax_error"
                # Still add to valid_samples so it's counted but marked as failed
                valid_samples.append(sample)
        
        # Write validated results
        write_jsonl(validated_file, valid_samples)
        
        if invalid_count > 0:
            print(f"⚠️  Found {invalid_count} completion(s) with syntax errors (marked as failed)")
        else:
            print(f"✅ All {len(valid_samples)} completions are syntactically valid")
        
        return validated_file, invalid_count
    
    def evaluate_outputs(self, results_file: str, k: List[int] = [1, 10, 100], 
                         n_workers: int = 4, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Run real HumanEval functional correctness evaluation with syntax pre-validation.
        
        Args:
            results_file: Path to JSONL file with completions (format: {"task_id": "...", "completion": "..."})
            k: List of k values for Pass@k calculation
            n_workers: Number of parallel workers for test execution
            timeout: Timeout per test in seconds
            
        Returns:
            Dictionary with Pass@k results
        """
        print(f"\nRunning HumanEval functional correctness evaluation...")
        print(f"Results file: {results_file}")
        print(f"k values: {k}, workers: {n_workers}, timeout: {timeout}s")
        
        # Step 1: Pre-validate and sanitize all completions for syntax errors
        use_black = self.cfg.get("use_black_formatting", False)
        validated_file, invalid_count = self.prevalidate_completions(results_file, use_black=use_black)
        
        # Step 2: Read the validated results to get which tasks were attempted
        from human_eval.data import stream_jsonl
        attempted_task_ids = set()
        for sample in stream_jsonl(validated_file):
            attempted_task_ids.add(sample["task_id"])
        
        # Create a filtered problem file with only attempted tasks
        all_problems = read_problems()
        filtered_problems = {task_id: all_problems[task_id] for task_id in attempted_task_ids if task_id in all_problems}
        
        # Write filtered problems to a temporary file
        import tempfile
        temp_problem_file = os.path.join(os.path.dirname(results_file), "filtered_problems.jsonl.gz")
        write_jsonl(temp_problem_file, filtered_problems.values())
        
        print(f"Evaluating {len(attempted_task_ids)} tasks (out of {len(all_problems)} total)")
        
        # evaluate_functional_correctness writes results to {validated_file}_results.jsonl.gz
        # Use validated_file instead of original results_file
        results = evaluate_functional_correctness(
            sample_file=validated_file,
            k=k,
            n_workers=n_workers,
            timeout=timeout,
            problem_file=temp_problem_file  # Use filtered problem file
        )
        
        # Add syntax error statistics to results
        if invalid_count > 0:
            results["syntax_errors"] = invalid_count
            results["total_attempted"] = len(attempted_task_ids)
            results["syntax_error_rate"] = invalid_count / len(attempted_task_ids) if attempted_task_ids else 0.0
            print(f"\n📊 Syntax Validation Summary:")
            print(f"   Total completions: {len(attempted_task_ids)}")
            print(f"   Syntax errors: {invalid_count} ({results['syntax_error_rate']:.1%})")
        
        # Clean up temporary files
        try:
            os.remove(temp_problem_file)
        except:
            pass
        
        # Optionally clean up validated file (or keep for debugging)
        # Uncomment the following to remove validated file after evaluation:
        # try:
        #     os.remove(validated_file)
        # except:
        #     pass
        
        return results
