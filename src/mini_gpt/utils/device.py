"""
Device detection (CUDA / MPS / CPU) and seed setup utilities.
"""

import random
import numpy as np
import torch


def get_device() -> torch.device:
    """
    Automatically detects available hardware accelerator.
    Order of preference: CUDA > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, PyTorch, and CUDA for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
