# utils.py
"""
Utility functions for logging and verbosity control.
"""
import sys
from typing import Optional

# Global verbosity setting
_verbose = False

def set_verbose(enabled: bool):
    """Set global verbosity level."""
    global _verbose
    _verbose = enabled

def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _verbose

def log(message: str, level: str = "info", force: bool = False):
    """
    Log a message based on verbosity level.
    
    Args:
        message: Message to log
        level: Log level ("info", "warning", "error", "success")
        force: Force output even if not verbose (for errors/warnings)
    """
    global _verbose
    
    # Always show errors, warnings, and success messages
    if force or level in ["error", "warning", "success"]:
        print(message, file=sys.stderr if level == "error" else sys.stdout)
    elif _verbose and level == "info":
        print(message)

def log_info(message: str):
    """Log info message (only if verbose)."""
    log(message, level="info")

def log_warning(message: str):
    """Log warning message (always shown)."""
    log(message, level="warning", force=True)

def log_error(message: str):
    """Log error message (always shown)."""
    log(message, level="error", force=True)

def log_success(message: str):
    """Log success message (always shown)."""
    log(message, level="success", force=True)












