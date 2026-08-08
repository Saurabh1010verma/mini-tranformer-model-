"""
Custom Tokenizer implementation built from scratch for Mini GPT.
Supports Character-level tokenization and Byte-Pair Encoding (BPE) subword tokenization
with full serialization, special token handling, and encoding/decoding capabilities.
"""

from typing import List, Dict, Tuple, Union, Optional
import json
import os
import re


SPECIAL_TOKENS = {
    "<|unk|>": 0,
    "<|endoftext|>": 1,
}


class Tokenizer:
    """
    Modular Tokenizer supporting BPE subword and Character-level modes.
    """

    def __init__(
        self,
        unk_token: str = "<|unk|>",
        eos_token: str = "<|endoftext|>",
        mode: str = "bpe",
    ):
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.mode = mode.lower()

        # Special token mapping
        self.special_tokens = {
            self.unk_token: 0,
            self.eos_token: 1,
        }
        self.special_ids = {idx: tok for tok, idx in self.special_tokens.items()}

        # Core vocabulary: string token -> int id, int id -> string token
        self.vocab: Dict[str, int] = dict(self.special_tokens)
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # BPE merge rules: Tuple[str, str] -> merged string token
        self.merges: List[Tuple[str, str]] = []
        self._cache: Dict[str, List[str]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def unk_id(self) -> int:
        return self.special_tokens[self.unk_token]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[self.eos_token]

    def _get_stats(self, word_counts: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Compute frequency of adjacent symbol pairs across all words."""
        stats: Dict[Tuple[str, str], int] = {}
        for word, freq in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                stats[pair] = stats.get(pair, 0) + freq
        return stats

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        """Merge all occurrences of pair in a single word sequence."""
        first, second = pair
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def fit(self, text: str, target_vocab_size: int = 1000) -> None:
        """
        Train vocabulary on input text string.
        """
        if self.mode == "char":
            self._fit_char(text)
        else:
            self._fit_bpe(text, target_vocab_size)

    def _fit_char(self, text: str) -> None:
        """Build character-level vocabulary."""
        unique_chars = sorted(list(set(text)))
        self.vocab = dict(self.special_tokens)

        for ch in unique_chars:
            if ch not in self.vocab:
                idx = len(self.vocab)
                self.vocab[ch] = idx

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = []

    def _fit_bpe(self, text: str, target_vocab_size: int = 1000) -> None:
        """Train Byte-Pair Encoding merges up to target_vocab_size."""
        # 1. Initialize vocabulary with special tokens
        self.vocab = dict(self.special_tokens)

        # 2. Extract initial words split as individual characters
        # We split text by whitespace/punctuation to bound merge scopes
        words = text.split()
        word_counts: Dict[Tuple[str, ...], int] = {}

        initial_chars = set()
        for w in words:
            char_tuple = tuple(list(w))
            word_counts[char_tuple] = word_counts.get(char_tuple, 0) + 1
            for ch in char_tuple:
                initial_chars.add(ch)

        # Add initial characters to vocab
        for ch in sorted(list(initial_chars)):
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)

        # Also add whitespace symbol space representation or newline support if needed
        # We also handle newline as special char token if present in text
        for extra in ["\n", " "]:
            if extra in text and extra not in self.vocab:
                self.vocab[extra] = len(self.vocab)

        self.merges = []
        num_merges = max(0, target_vocab_size - len(self.vocab))

        # Iteratively find top pair and merge
        for _ in range(num_merges):
            stats = self._get_stats(word_counts)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # Stop if pair occurs only once
                break

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)
                self.merges.append(best_pair)

            # Update word representations
            new_word_counts: Dict[Tuple[str, ...], int] = {}
            for word, freq in word_counts.items():
                new_word = self._merge_word(word, best_pair)
                new_word_counts[new_word] = freq
            word_counts = new_word_counts

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Encode text string into token IDs.
        """
        if self.mode == "char":
            return [self.vocab.get(ch, self.unk_id) for ch in text]

        # BPE Encoding
        ids: List[int] = []
        # Segment into text fragments while preserving special tokens
        tokens = self._bpe_encode_text(text)
        for tok in tokens:
            ids.append(self.vocab.get(tok, self.unk_id))
        return ids

    def _bpe_encode_word(self, word: str) -> List[str]:
        """Apply learned BPE merges to a single word."""
        if not word:
            return []

        if word in self._cache:
            return list(self._cache[word])

        word_tokens = tuple(list(word))

        for pair in self.merges:
            if len(word_tokens) <= 1:
                break
            word_tokens = self._merge_word(word_tokens, pair)

        res = list(word_tokens)
        self._cache[word] = res
        return list(res)

    def _bpe_encode_text(self, text: str) -> List[str]:
        """Encode full text preserving whitespace/newlines."""
        tokens: List[str] = []
        # Split text while preserving delimiters like space and newline
        parts = re.split(r'(\s+)', text)

        for part in parts:
            if not part:
                continue
            if part in self.vocab:
                tokens.append(part)
            elif part.isspace():
                # Handle space sequence
                for char in part:
                    tokens.append(char if char in self.vocab else self.unk_token)
            else:
                tokens.extend(self._bpe_encode_word(part))
        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text string.
        """
        tokens = [self.inverse_vocab.get(i, self.unk_token) for i in ids]
        # Join tokens, skipping eos token or unk marker formatting if desired
        decoded = "".join(tokens)
        return decoded

    def save(self, path: str) -> None:
        """Save vocabulary and merge rules to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "mode": self.mode,
            "unk_token": self.unk_token,
            "eos_token": self.eos_token,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        """Load vocabulary and merge rules from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            unk_token=data.get("unk_token", "<|unk|>"),
            eos_token=data.get("eos_token", "<|endoftext|>"),
            mode=data.get("mode", "bpe"),
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.inverse_vocab = {int(v): k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(m) for m in data.get("merges", [])]
        return tokenizer
