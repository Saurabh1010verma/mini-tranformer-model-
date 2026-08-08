"""
Trainer class orchestrating training, validation, mixed precision, gradient clipping,
perplexity computation, and text generation callbacks.
"""

from typing import Optional, Dict
import math
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mini_gpt.config import Config
from mini_gpt.model.gpt import MiniGPT
from mini_gpt.training.optimizer import configure_optimizers
from mini_gpt.training.checkpoint import save_checkpoint
from mini_gpt.utils.device import get_device, set_seed
from mini_gpt.utils.logging import setup_logger


class Trainer:
    """
    Main Trainer for Mini GPT.
    """

    def __init__(
        self,
        model: MiniGPT,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.logger = setup_logger("Trainer")

        # Hardware setup
        self.device = device or get_device()
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

        # Seed setup
        set_seed(config.training.seed)

        # Optimizer and Scheduler
        self.optimizer, self.scheduler = configure_optimizers(model, config.training)

        # Mixed Precision Scaler
        self.use_amp = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("Automatic Mixed Precision (AMP) enabled.")

        self.global_step = 0
        self.best_val_loss = float("inf")

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation dataloader.
        Returns dictionary containing val_loss and perplexity.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, loss = self.model(x, targets=y)

            total_loss += loss.item()
            total_batches += 1

            if total_batches >= self.config.training.eval_iters:
                break

        avg_val_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_val_loss, 20.0))  # Cap exponent to prevent overflow

        self.model.train()
        return {"val_loss": avg_val_loss, "perplexity": perplexity}

    def generate_sample(self, prompt: str = "The ", max_new_tokens: int = 30) -> str:
        """Helper callback to generate sample text during training."""
        if self.tokenizer is None:
            return ""

        from mini_gpt.generation.generate import generate_text

        sample_output = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=20,
            device=self.device,
        )
        return sample_output

    def train(self) -> None:
        """
        Run the complete training loop across max_epochs.
        """
        self.logger.info(
            f"Starting training for {self.config.training.max_epochs} epochs | "
            f"Model Parameters: {self.model.get_num_params():,}"
        )

        start_time = time.time()

        for epoch in range(1, self.config.training.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for step, (x, y) in enumerate(self.train_loader, start=1):
                self.global_step += 1
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                # Forward pass with AMP if CUDA available
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    logits, loss = self.model(x, targets=y)

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.training.grad_clip > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)

                # Step optimizer and scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                epoch_loss += loss.item()

                # Evaluation & Logging step
                if self.global_step % self.config.training.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    val_loss = eval_metrics["val_loss"]
                    ppl = eval_metrics["perplexity"]

                    self.logger.info(
                        f"Epoch {epoch:02d}/{self.config.training.max_epochs:02d} | "
                        f"Step {self.global_step:05d} | "
                        f"Train Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Perplexity: {ppl:.2f}"
                    )

                    # Sample generation printout
                    if self.tokenizer is not None:
                        sample = self.generate_sample(prompt="The ")
                        self.logger.info(f"Sample Generation: '{sample}'")

                    # Checkpoint saving for best val loss
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        save_checkpoint(
                            checkpoint_dir=self.config.checkpoint.dir,
                            filename="best.pt",
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=self.global_step,
                            train_loss=loss.item(),
                            val_loss=val_loss,
                            config=self.config,
                        )

            # End of epoch summary
            avg_train_loss = epoch_loss / len(self.train_loader)
            eval_metrics = self.evaluate()
            self.logger.info(
                f"=== Epoch {epoch} Complete | Avg Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {eval_metrics['val_loss']:.4f} | PPL: {eval_metrics['perplexity']:.2f} ==="
            )

            # Save latest checkpoint after every epoch
            save_checkpoint(
                checkpoint_dir=self.config.checkpoint.dir,
                filename="latest.pt",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                step=self.global_step,
                train_loss=avg_train_loss,
                val_loss=eval_metrics["val_loss"],
                config=self.config,
            )

        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {elapsed / 60.0:.2f} minutes.")
