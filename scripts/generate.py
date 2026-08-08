"""
CLI Text Generation Tool for Mini GPT.
Usage:
    python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "To be or not to be"
    python scripts/generate.py --prompt "The future of AI" --temperature 0.7 --top_k 40 --max_tokens 150
"""

import argparse
import os
import sys

# Ensure src directory is in Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch

from mini_gpt.config import ModelConfig
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.model.gpt import MiniGPT
from mini_gpt.generation.generate import generate_text
from mini_gpt.training.checkpoint import load_checkpoint
from mini_gpt.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained Mini GPT checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt", help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default="data/processed/tokenizer.json", help="Path to tokenizer JSON file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to complete")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0.0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling pool size")
    args = parser.parse_args()

    device = get_device()
    print(f"[Generate] Using device: {device}")

    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer}. Run prepare_data.py first.")

    tokenizer = Tokenizer.load(args.tokenizer)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Run train.py first.")

    ckpt = load_checkpoint(args.checkpoint, map_location=device.type)

    # Reconstruct model from saved checkpoint config
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = MiniGPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\n--- PROMPT ---\n{args.prompt}\n")
    print("--- GENERATING ---")

    generated_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(f"\n--- OUTPUT ---\n{generated_output}\n")


if __name__ == "__main__":
    main()
