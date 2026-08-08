"""
Unit tests for custom Tokenizer.
"""

import os
import tempfile
import pytest
from mini_gpt.tokenizer import Tokenizer, SPECIAL_TOKENS


def test_tokenizer_char_mode():
    text = "Hello world!"
    tokenizer = Tokenizer(mode="char")
    tokenizer.fit(text)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert len(encoded) == len(text)
    assert decoded == text


def test_tokenizer_bpe_mode():
    text = "To be, or not to be, that is the question."
    tokenizer = Tokenizer(mode="bpe")
    tokenizer.fit(text, target_vocab_size=100)

    assert tokenizer.vocab_size <= 100
    assert "<|unk|>" in tokenizer.vocab
    assert "<|endoftext|>" in tokenizer.vocab

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert isinstance(encoded, list)
    assert len(encoded) > 0
    assert decoded == text


def test_tokenizer_special_tokens():
    tokenizer = Tokenizer()
    assert tokenizer.vocab["<|unk|>"] == 0
    assert tokenizer.vocab["<|endoftext|>"] == 1
    assert tokenizer.unk_id == 0
    assert tokenizer.eos_id == 1


def test_tokenizer_save_load():
    text = "Save and load tokenizer test data string."
    tokenizer = Tokenizer(mode="bpe")
    tokenizer.fit(text, target_vocab_size=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.save(save_path)

        assert os.path.exists(save_path)

        loaded_tokenizer = Tokenizer.load(save_path)
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert loaded_tokenizer.vocab == tokenizer.vocab

        encoded_orig = tokenizer.encode(text)
        encoded_loaded = loaded_tokenizer.encode(text)
        assert encoded_orig == encoded_loaded
