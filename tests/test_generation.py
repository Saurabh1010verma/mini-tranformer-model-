"""
Unit tests for text generation and autoregressive sampling.
"""

import pytest
import torch
from mini_gpt.config import ModelConfig
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.model.gpt import MiniGPT
from mini_gpt.generation.generate import generate_tokens, generate_text


@pytest.fixture
def dummy_setup():
    cfg = ModelConfig(vocab_size=50, context_length=32, embedding_dim=32, num_heads=2, num_layers=2)
    model = MiniGPT(cfg)
    model.eval()

    text = "The quick brown fox jumps over the lazy dog."
    tokenizer = Tokenizer(mode="char")
    tokenizer.fit(text)
    return model, tokenizer


def test_generate_tokens_greedy(dummy_setup):
    model, _ = dummy_setup
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = generate_tokens(model, idx, max_new_tokens=10, temperature=0.0)
    assert out.shape == (1, 3 + 10)


def test_generate_tokens_temperature_topk(dummy_setup):
    model, _ = dummy_setup
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long)

    out = generate_tokens(model, idx, max_new_tokens=15, temperature=0.7, top_k=5)
    assert out.shape == (1, 3 + 15)


def test_generate_text_high_level(dummy_setup):
    model, tokenizer = dummy_setup
    prompt = "The quick"

    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=10,
    )

    assert isinstance(result, str)
    assert result.startswith(prompt)
