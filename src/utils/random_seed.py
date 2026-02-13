"""Global random seed management for reproducibility."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set random seed globally for all random generators.
    
    This ensures reproducible results across NumPy, Python's random module,
    and other libraries that use randomness.
    
    Args:
        seed: Random seed value (e.g., 42)
        
    Usage:
        from src.utils.random_seed import set_global_seed
        set_global_seed(42)  # Call once at program start
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # TensorFlow (if installed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch (if installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
