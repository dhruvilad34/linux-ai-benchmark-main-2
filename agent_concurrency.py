#!/usr/bin/env python3
"""
agent_concurrency.py
Wrapper script for main.py that supports kernel optimization via --optimize flag.
This script can automatically apply kernel tuning before running the main AI concurrency workload.
"""

import sys
import os
import subprocess
import argparse
import asyncio

# Add project directory to path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

def apply_kernel_optimization(use_sudo=True, fallback_no_sudo=True):
    """Apply kernel optimization by running optimize_kernel.sh with sudo.
    
    Args:
        use_sudo: Try to use sudo for full kernel optimization
        fallback_no_sudo: If sudo fails, try user-level optimizations
    """
    # First, try sudo-based optimization if requested
    if use_sudo:
        script_path = os.path.join(project_dir, "optimize_kernel.sh")
        
        if not os.path.exists(script_path):
            print(f"[WARN] optimize_kernel.sh not found at {script_path}")
        else:
            if not os.access(script_path, os.X_OK):
                print(f"[WARN] optimize_kernel.sh is not executable, attempting to make it executable...")
                try:
                    os.chmod(script_path, 0o755)
                except Exception as e:
                    print(f"[WARN] Could not make script executable: {e}")
            
            print("[INFO] Attempting kernel optimization with sudo...")
            
            try:
                # Run the optimization script with sudo
                result = subprocess.run(
                    ["sudo", script_path],
                    check=False,  # Don't raise exception on non-zero exit
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print("[SUCCESS] Kernel optimization applied successfully!")
                    if result.stdout:
                        print(result.stdout)
                    return True
                else:
                    print(f"[WARN] Sudo-based kernel optimization failed (code {result.returncode})")
                    if "not in the sudoers file" in result.stderr or "password" in result.stderr.lower():
                        print("[INFO] No sudo access available. Trying user-level optimizations...")
                    elif result.stderr:
                        print(f"[INFO] {result.stderr[:200]}")
            except FileNotFoundError:
                print("[INFO] 'sudo' command not found. Trying user-level optimizations...")
            except subprocess.TimeoutExpired:
                print("[WARN] Sudo command timed out. Trying user-level optimizations...")
            except Exception as e:
                print(f"[INFO] Sudo attempt failed: {e}. Trying user-level optimizations...")
    
    # Fallback to user-level optimizations (no sudo required)
    if fallback_no_sudo:
        no_sudo_script = os.path.join(project_dir, "optimize_kernel_no_sudo.sh")
        
        if os.path.exists(no_sudo_script):
            print("[INFO] Applying user-level optimizations (no sudo required)...")
            try:
                # Source the script to set environment variables in current process
                result = subprocess.run(
                    ["bash", "-c", f"source {no_sudo_script} && env"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    # Apply environment variables from the script
                    for line in result.stdout.split('\n'):
                        if '=' in line and line.startswith(('OMP_', 'MKL_', 'NUMEXPR_', 'THP_', 'PYTHON')):
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                    
                    print("[SUCCESS] User-level optimizations applied!")
                    print("[INFO] Set environment variables for threading and process optimization")
                    return True
                else:
                    print(f"[WARN] User-level optimization script failed: {result.stderr[:200]}")
            except Exception as e:
                print(f"[WARN] Could not apply user-level optimizations: {e}")
        else:
            print("[INFO] User-level optimization script not found. Skipping...")
    
    return False

def main():
    """Main entry point that handles --optimize flag and delegates to main.py"""
    parser = argparse.ArgumentParser(
        description="AI Concurrency Agent Runner with optional kernel optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_concurrency.py                    # Run normally
  python agent_concurrency.py --optimize          # Apply kernel tuning, then run
  python agent_concurrency.py --config config/config.yaml --num_tasks 10
        """
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply kernel optimization before running (requires sudo)"
    )
    
    # Pass through all other arguments to main.py
    args, unknown_args = parser.parse_known_args()
    
    # If --optimize is set, apply kernel optimization first
    if args.optimize:
        success = apply_kernel_optimization()
        if not success:
            print("[WARN] Kernel optimization failed, but continuing with normal execution...")
        print()  # Add blank line for readability
    
    # Import and run main.py's main function
    # main.py uses its own ArgumentParser, so we need to ensure it gets the right args
    # Remove --optimize from sys.argv since main.py doesn't know about it
    original_argv = sys.argv.copy()
    if "--optimize" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--optimize"]
    
    try:
        # Import main module and run its async main function
        # main.py will parse its own arguments from sys.argv
        from main import main as main_async
        
        # Run the async main function
        asyncio.run(main_async())
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Failed to run main script: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    main()

