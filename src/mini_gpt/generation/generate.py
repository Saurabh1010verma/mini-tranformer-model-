"""
Autoregressive text generation module for Mini GPT.
Supports Greedy Decoding, Temperature Scaling, and Top-K Sampling.
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from mini_gpt.tokenizer import Tokenizer


@torch.no_grad()
def generate_tokens(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressively generate next token IDs using temperature and top-k sampling.

    Args:
        model: Trained MiniGPT PyTorch model.
        idx: Initial prompt token IDs of shape (batch_size, seq_len).
        max_new_tokens: Maximum number of tokens to append.
        temperature: Temperature parameter for scaling logits. Higher = more random, lower = more deterministic.
        top_k: If set > 0, restricts sampling pool to top k logits.
        eos_id: Optional End-Of-Sequence token ID to trigger early exit.

    Returns:
        Tensor of shape (batch_size, seq_len + generated_length) with prompt + generated token IDs.
    """
    model.eval()
    context_length = getattr(model.config, "context_length", 256)

    for _ in range(max_new_tokens):
        # Crop context to fit inside model's maximum context length window
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]

        # Forward pass to get output logits
        logits, _ = model(idx_cond)

        # Pluck logits at final time step: (batch_size, vocab_size)
        logits = logits[:, -1, :]

        # 1. Greedy decoding if temperature is 0
        if temperature <= 0.0:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # 2. Temperature scaling
            logits = logits / temperature

            # 3. Top-K filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                # Set all logits below top k threshold to -infinity
                logits[logits < v[:, [-1]]] = float("-inf")

            # 4. Softmax & Multinomial Sampling
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Append generated token to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

        # Early termination if End-Of-Sequence token generated
        if eos_id is not None and (idx_next == eos_id).all():
            break

    return idx


def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    device: Optional[torch.device] = None,
) -> str:
    """
    High-level text generation helper. Encodes prompt, runs autoregressive sampling,
    and returns decoded text string.
    """
    if device is None:
        device = next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.unk_id]

    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    out_idx = generate_tokens(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )

    generated_ids = out_idx[0].tolist()
    decoded_text = tokenizer.decode(generated_ids)
    return decoded_text
