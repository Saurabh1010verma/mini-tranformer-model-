"""
CLI Evaluation Script for Mini GPT.
Calculates validation loss and perplexity (PPL) on validation dataset split.
Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
"""

import argparse
import math
import os
import sys

# Ensure src directory is in Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch

from mini_gpt.config import ModelConfig, Config
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.dataset import GPTDataset
from mini_gpt.model.gpt import MiniGPT
from mini_gpt.training.checkpoint import load_checkpoint
from mini_gpt.utils.device import get_device
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mini GPT checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--data_path", type=str, default="data/processed/val_ids.pt", help="Path to validation token tensor")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    device = get_device()
    print(f"[Evaluate] Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Validation data tensor not found at: {args.data_path}")

    # 1. Load Checkpoint
    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)
    model_cfg = ModelConfig(**ckpt["config"]["model"])

    # 2. Instantiate Model and Load Weights
    model = MiniGPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Load Validation Dataset
    val_tokens = torch.load(args.data_path)
    dataset = GPTDataset(val_tokens, context_length=model_cfg.context_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[Evaluate] Loaded model with {model.get_num_params():,} params.")
    print(f"[Evaluate] Evaluating on {len(dataset):,} validation sequences...")

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(avg_loss)

    print("\n================ EVALUATION RESULTS ================")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Epoch/Step:      Epoch {ckpt.get('epoch', 'N/A')} | Step {ckpt.get('step', 'N/A')}")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity:      {perplexity:.2f}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
