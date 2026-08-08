"""
Optimizer configuration with parameter weight decay separation and
Cosine Annealing with Linear Warmup learning rate scheduler.
"""

from typing import Tuple, Dict
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from mini_gpt.config import TrainingConfig


def configure_optimizers(
    model: nn.Module, config: TrainingConfig
) -> Tuple[AdamW, LambdaLR]:
    """
    Separate model parameters into 2D weights (decayed) and 1D weights/biases (no decay),
    and construct AdamW optimizer with Cosine Annealing + Warmup learning rate scheduler.
    """
    # Separate parameter groups
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases and 1D norm weights do not get weight decay
        if param.dim() < 2 or name.endswith(".bias") or "ln_" in name or "norm" in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Cosine scheduler with linear warmup
    def lr_lambda(current_step: int) -> float:
        warmup_steps = config.warmup_steps
        max_steps = config.max_epochs * 1000  # Default step scale fallback if needed

        if current_step < warmup_steps:
            # Linear warmup phase
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = 0.1
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    return optimizer, scheduler
