"""
CLI Training script for Mini GPT.
Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config config.yaml
"""

import argparse
import os
import sys

# Ensure src directory is in Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch

from mini_gpt.config import Config
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.dataset import create_dataloaders
from mini_gpt.model.gpt import MiniGPT
from mini_gpt.training.trainer import Trainer
from mini_gpt.training.checkpoint import load_checkpoint
from mini_gpt.utils.logging import setup_logger
from mini_gpt.utils.device import get_device


def train(config_path: str, resume_path: str = None) -> None:
    logger = setup_logger("TrainScript")

    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return

    cfg = Config.from_yaml(config_path)

    # 1. Check data availability, run prepare_data if missing
    train_pt = os.path.join(cfg.data.processed_dir, "train_ids.pt")
    val_pt = os.path.join(cfg.data.processed_dir, "val_ids.pt")
    tok_json = os.path.join(cfg.data.processed_dir, "tokenizer.json")

    if not (os.path.exists(train_pt) and os.path.exists(val_pt) and os.path.exists(tok_json)):
        logger.info("Processed dataset/tokenizer not found. Running prepare_data.py...")
        from scripts.prepare_data import prepare_data

        prepare_data(config_path)

    # 2. Load tokenizer and dataset
    tokenizer = Tokenizer.load(tok_json)
    train_tokens = torch.load(train_pt)
    val_tokens = torch.load(val_pt)

    # Update config vocab_size if tokenizer differs
    if cfg.model.vocab_size != tokenizer.vocab_size:
        logger.info(
            f"Updating model vocab_size from config ({cfg.model.vocab_size}) "
            f"to match tokenizer ({tokenizer.vocab_size})"
        )
        cfg.model.vocab_size = tokenizer.vocab_size

    # 3. Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_tokens=train_tokens,
        val_tokens=val_tokens,
        context_length=cfg.model.context_length,
        batch_size=cfg.training.batch_size,
    )

    # 4. Instantiate MiniGPT model
    model = MiniGPT(cfg.model)
    logger.info(f"Model initialized | Parameters: {model.get_num_params():,}")

    # 5. Initialize Trainer
    device = get_device()
    trainer = Trainer(
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
    )

    # 6. Optional Resume checkpoint
    if resume_path:
        logger.info(f"Resuming training from checkpoint: {resume_path}")
        ckpt = load_checkpoint(resume_path, map_location=device.type)
        model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") and trainer.scheduler:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer.global_step = ckpt.get("step", 0)

    # 7. Start training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mini GPT language model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume")
    args = parser.parse_args()

    train(args.config, args.resume)
