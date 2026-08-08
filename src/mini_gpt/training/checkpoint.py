"""
Checkpoint handling functions for saving, loading, and resuming training state.
"""

from typing import Dict, Any, Tuple, Optional
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from mini_gpt.config import Config


def save_checkpoint(
    checkpoint_dir: str,
    filename: str,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    config: Config,
) -> str:
    """
    Save model, optimizer, scheduler states and training metadata to disk.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config.to_dict(),
    }

    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load saved checkpoint file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint
