"""
Dataset and DataLoader utilities for Mini GPT autoregressive training.
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """
    PyTorch Dataset for autoregressive language modeling.
    Given a sequence of token IDs, extracts contiguous input (x) and target (y) pairs
    where y is shifted by 1 position relative to x.
    """

    def __init__(self, token_ids: torch.Tensor, context_length: int):
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        if token_ids.ndim != 1:
            token_ids = token_ids.squeeze()

        if len(token_ids) <= context_length:
            raise ValueError(
                f"Token sequence length ({len(token_ids)}) must be greater than "
                f"context_length ({context_length})."
            )

        self.token_ids = token_ids.long()
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.context_length]
        y = self.token_ids[idx + 1 : idx + 1 + self.context_length]
        return x, y


def create_dataloaders(
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    context_length: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation datasets.
    """
    train_dataset = GPTDataset(train_tokens, context_length=context_length)
    val_dataset = GPTDataset(val_tokens, context_length=context_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader
