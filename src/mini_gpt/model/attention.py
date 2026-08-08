"""
Scaled Dot-Product Attention and Causal Multi-Head Self-Attention for Mini GPT.
Implemented from scratch using standard PyTorch linear projections and tensor operations.
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_gpt.config import ModelConfig


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax( (Q K^T / sqrt(d_k)) + mask ) V

    Args:
        query: Tensor of shape (B, num_heads, T, head_dim)
        key: Tensor of shape (B, num_heads, T, head_dim)
        value: Tensor of shape (B, num_heads, T, head_dim)
        attn_mask: Optional boolean or float mask of shape (1, 1, T, T)
        dropout_p: Dropout probability on attention weights
        training: Boolean flag indicating training mode

    Returns:
        Tuple of (output_tensor, attention_weights)
    """
    d_k = query.size(-1)
    # Scaled dot-product query @ key^T -> (B, num_heads, T, T)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if attn_mask is not None:
        # Mask out upper triangular region with -inf before softmax
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=training)

    output = torch.matmul(attn_weights, value)  # (B, num_heads, T, head_dim)
    return output, attn_weights


class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention layer.
    Ensures autoregressive property via triangular lower causal mask.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.dropout = config.dropout

        # Combined Q, K, V projection for speed and standard GPT layout
        self.c_attn = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix of ones (1, 1, context_length, context_length)
        # 1 means attend, 0 means mask out
        causal_mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer(
            "bias",
            causal_mask.view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Multi-Head Self-Attention.
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            return_attn_weights: Option to return attention matrix for visualization
        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dim (channels)

        # Calculate Query, Key, Value for all heads in a single batch matrix multiplication
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.embedding_dim, dim=2)  # Each is (B, T, C)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Slice causal mask up to current sequence length T
        causal_mask = self.bias[:, :, :T, :T]

        # Scaled dot-product attention
        y, attn_weights = scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            training=self.training,
        )

        # Re-assemble all head outputs side-by-side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output linear projection + residual dropout
        out = self.resid_dropout(self.c_proj(y))

        if return_attn_weights:
            return out, attn_weights
        return out, None
