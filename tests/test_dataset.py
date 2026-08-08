"""
Unit tests for GPTDataset and DataLoader creation.
"""

import pytest
import torch
from mini_gpt.dataset import GPTDataset, create_dataloaders


def test_gpt_dataset_shapes():
    token_ids = torch.arange(100, dtype=torch.long)
    context_length = 10

    dataset = GPTDataset(token_ids, context_length=context_length)
    assert len(dataset) == 100 - context_length

    x, y = dataset[0]
    assert x.shape == (context_length,)
    assert y.shape == (context_length,)

    # Input x should be [0..9], target y should be [1..10]
    assert torch.equal(x, torch.arange(0, 10))
    assert torch.equal(y, torch.arange(1, 11))


def test_gpt_dataset_invalid_length():
    short_tokens = torch.arange(5, dtype=torch.long)
    context_length = 10

    with pytest.raises(ValueError):
        GPTDataset(short_tokens, context_length=context_length)


def test_create_dataloaders():
    train_tokens = torch.arange(100, dtype=torch.long)
    val_tokens = torch.arange(50, dtype=torch.long)

    train_loader, val_loader = create_dataloaders(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        context_length=8,
        batch_size=4,
    )

    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (4, 8)
    assert y_batch.shape == (4, 8)

    # Shift assertion
    assert torch.equal(x_batch[:, 1:], y_batch[:, :-1])
