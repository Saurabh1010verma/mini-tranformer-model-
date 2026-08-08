"""
Token and Positional Embedding modules for Mini GPT.
"""

import torch
import torch.nn as nn
from mini_gpt.config import ModelConfig


class GPTEmbeddings(nn.Module):
    """
    Combines learned Token Embeddings and Positional Embeddings with Dropout.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for embedding layer.
        Args:
            idx: Tensor of token indices with shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = idx.shape

        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds model context_length "
                f"({self.config.context_length})"
            )

        # Generate positional indices [0, 1, 2, ..., seq_len - 1]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)

        # Compute embeddings
        tok_emb = self.token_embedding(idx)  # (batch_size, seq_len, embedding_dim)
        pos_emb = self.position_embedding(pos)  # (seq_len, embedding_dim)

        x = self.dropout(tok_emb + pos_emb)
        return x
