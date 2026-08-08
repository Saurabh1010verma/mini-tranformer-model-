"""
Unit tests for MiniGPT model, embedding, and backward pass gradient flow.
"""

import pytest
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model.gpt import MiniGPT


def test_minigpt_forward_shape():
    cfg = ModelConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
    )
    model = MiniGPT(cfg)

    x = torch.randint(0, 100, (4, 16))
    logits, loss = model(x)

    assert logits.shape == (4, 16, 100)
    assert loss is None


def test_minigpt_loss_computation_and_backward():
    cfg = ModelConfig(
        vocab_size=100,
        context_length=32,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
    )
    model = MiniGPT(cfg)

    x = torch.randint(0, 100, (2, 16))
    y = torch.randint(0, 100, (2, 16))

    logits, loss = model(x, targets=y)

    assert loss is not None
    assert loss.item() > 0.0

    # Backpropagation check
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} is None"


def test_minigpt_parameter_count():
    cfg = ModelConfig(
        vocab_size=500,
        context_length=64,
        embedding_dim=128,
        num_heads=4,
        num_layers=3,
    )
    model = MiniGPT(cfg)
    num_params = model.get_num_params()
    assert isinstance(num_params, int)
    assert num_params > 0
