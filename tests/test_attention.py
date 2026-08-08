"""
Unit tests for Scaled Dot-Product Attention and Causal Multi-Head Self-Attention.
"""

import pytest
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.model.attention import scaled_dot_product_attention, CausalSelfAttention


def test_scaled_dot_product_attention_shapes():
    B, num_heads, T, head_dim = 2, 4, 16, 32
    q = torch.randn(B, num_heads, T, head_dim)
    k = torch.randn(B, num_heads, T, head_dim)
    v = torch.randn(B, num_heads, T, head_dim)

    out, attn_weights = scaled_dot_product_attention(q, k, v)

    assert out.shape == (B, num_heads, T, head_dim)
    assert attn_weights.shape == (B, num_heads, T, T)


def test_causal_masking_property():
    """Verify that upper triangular attention weights are zero (strictly masked)."""
    B, num_heads, T, head_dim = 1, 1, 4, 8
    q = torch.randn(B, num_heads, T, head_dim)
    k = torch.randn(B, num_heads, T, head_dim)
    v = torch.randn(B, num_heads, T, head_dim)

    # Causal lower triangular mask (1 for attend, 0 for mask)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    _, attn_weights = scaled_dot_product_attention(q, k, v, attn_mask=mask)

    # Convert to 2D matrix
    w = attn_weights[0, 0]

    # Upper triangular values (above main diagonal) must be zero
    upper_tri = torch.triu(w, diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri))


def test_causal_self_attention_module():
    cfg = ModelConfig(context_length=32, embedding_dim=64, num_heads=4, dropout=0.0)
    attn = CausalSelfAttention(cfg)

    x = torch.randn(2, 16, 64)
    out, weights = attn(x, return_attn_weights=True)

    assert out.shape == (2, 16, 64)
    assert weights.shape == (2, 4, 16, 16)
