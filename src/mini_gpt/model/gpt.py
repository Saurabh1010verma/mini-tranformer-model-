"""
MiniGPT Model class assembling Embeddings, Transformer Blocks, Final Norm, and LM Head.
Supports custom weight initialization, optional weight-tying, and loss computation.
"""

from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_gpt.config import ModelConfig
from mini_gpt.model.embeddings import GPTEmbeddings
from mini_gpt.model.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    """
    Mini GPT Language Model implemented from scratch using PyTorch.
    Follows autoregressive Transformer decoder architecture.
    """

    def __init__(self, config: ModelConfig, tie_weights: bool = True):
        super().__init__()
        config.validate()
        self.config = config
        self.tie_weights = tie_weights

        # 1. Embeddings (Token + Position)
        self.transformer = nn.ModuleDict(
            dict(
                emb=GPTEmbeddings(config),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)]),
                ln_f=nn.LayerNorm(config.embedding_dim, elementwise_affine=True, bias=config.bias),
            )
        )

        # 2. Linear Language Modeling Head
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # 3. Optional Weight Tying
        if self.tie_weights:
            self.lm_head.weight = self.transformer.emb.token_embedding.weight

        # 4. Initialize parameters according to GPT conventions
        self.apply(self._init_weights)

        # Scale residual projections special initialization
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding parameter count, position embeddings are subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.emb.position_embedding.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for MiniGPT model.

        Args:
            idx: Tensor of shape (batch_size, seq_len) with input token IDs.
            targets: Optional tensor of shape (batch_size, seq_len) with target token IDs.
            return_attn_weights: If True, returns list of attention maps per block.

        Returns:
            Tuple of (logits, loss). Loss will be None if targets is None.
        """
        device = idx.device
        b, t = idx.size()

        if t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.context_length}"
            )

        # 1. Forward embeddings
        x = self.transformer.emb(idx)  # (batch_size, seq_len, embedding_dim)

        # 2. Forward stack of Transformer decoder blocks
        attn_maps = []
        for block in self.transformer.h:
            x, attn_weights = block(x, return_attn_weights=return_attn_weights)
            if return_attn_weights:
                attn_maps.append(attn_weights)

        # 3. Final LayerNorm
        x = self.transformer.ln_f(x)

        # 4. Project to vocabulary logits
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        # 5. Compute loss if targets are supplied
        loss = None
        if targets is not None:
            # Flatten logits and targets to compute CrossEntropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss
