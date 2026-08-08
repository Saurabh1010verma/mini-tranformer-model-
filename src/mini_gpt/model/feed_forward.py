"""
Feed-Forward Neural Network (FFN) for Mini GPT Transformer Blocks.
"""

import torch
import torch.nn as nn
from mini_gpt.config import ModelConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN):
    FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))
    Expands representation dimension by 4x before projecting back.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Feed-Forward layer.
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim)
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
