"""
Data Preparation Script for Mini GPT.
Downloads raw text dataset, trains tokenizer, and generates train/val token tensors.
"""

import argparse
import os
import requests
import torch
from mini_gpt.config import Config
from mini_gpt.tokenizer import Tokenizer


def download_dataset(url: str, output_path: str) -> None:
    """Download text dataset if not already present."""
    if os.path.exists(output_path):
        print(f"[Dataset] Raw dataset already exists at: {output_path}")
        return

    print(f"[Dataset] Downloading raw dataset from: {url}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"[Dataset] Download completed. Size: {len(response.text)} characters.")


def prepare_data(config_path: str = "config.yaml") -> None:
    cfg = Config.from_yaml(config_path)

    raw_file = os.path.join(cfg.data.raw_dir, f"{cfg.data.dataset_name}.txt")
    processed_dir = cfg.data.processed_dir
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Download raw data
    download_dataset(cfg.data.data_url, raw_file)

    with open(raw_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"[Data Prep] Processing dataset '{cfg.data.dataset_name}' ({len(raw_text):,} chars)...")

    # 2. Train Tokenizer
    tokenizer = Tokenizer(mode="bpe")
    tokenizer.fit(raw_text, target_vocab_size=cfg.model.vocab_size)

    tokenizer_path = os.path.join(processed_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"[Tokenizer] Saved trained tokenizer (vocab_size={tokenizer.vocab_size}) to {tokenizer_path}")

    # 3. Tokenize full corpus
    print("[Data Prep] Tokenizing full text corpus...")
    token_ids = tokenizer.encode(raw_text)
    print(f"[Data Prep] Total tokens generated: {len(token_ids):,}")

    # 4. Split into train and val
    val_split = cfg.data.val_split
    val_size = int(len(token_ids) * val_split)
    train_size = len(token_ids) - val_size

    train_tokens = torch.tensor(token_ids[:train_size], dtype=torch.long)
    val_tokens = torch.tensor(token_ids[train_size:], dtype=torch.long)

    train_path = os.path.join(processed_dir, "train_ids.pt")
    val_path = os.path.join(processed_dir, "val_ids.pt")

    torch.save(train_tokens, train_path)
    torch.save(val_tokens, val_path)

    print(f"[Data Prep] Saved train tokens ({len(train_tokens):,}) -> {train_path}")
    print(f"[Data Prep] Saved val tokens ({len(val_tokens):,}) -> {val_path}")
    print("[Data Prep] Data preparation successfully finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset and tokenizer for Mini GPT")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    prepare_data(args.config)
