"""
Transformer Decoder Block for Mini GPT.
Follows modern Pre-Layer Normalization architecture with residual connections.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
from mini_gpt.config import ModelConfig
from mini_gpt.model.attention import CausalSelfAttention
from mini_gpt.model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    """
    Standard GPT Decoder Block:
    1. Pre-LayerNorm -> Causal Multi-Head Self-Attention -> Residual Add
    2. Pre-LayerNorm -> Feed-Forward Network -> Residual Add
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embedding_dim, elementwise_affine=True, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embedding_dim, elementwise_affine=True, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(
        self, x: torch.Tensor, return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for single Transformer block.
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            return_attn_weights: Option to return attention matrix
        Returns:
            Tuple of (output tensor, attention weights optional)
        """
        # Attention sub-block with residual connection
        attn_out, attn_weights = self.attn(self.ln_1(x), return_attn_weights=return_attn_weights)
        x = x + attn_out

        # Feed-Forward sub-block with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x, attn_weights
