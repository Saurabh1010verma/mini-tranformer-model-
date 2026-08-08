"""
Configuration classes and YAML loading utilities for Mini GPT.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import yaml
import os


@dataclass
class ModelConfig:
    vocab_size: int = 1000
    context_length: int = 256
    embedding_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    bias: bool = False

    def validate(self) -> None:
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"num_heads ({self.num_heads}). Head dimension = {self.embedding_dim / self.num_heads}"
            )


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_interval: int = 200
    eval_iters: int = 20
    seed: int = 42
    mixed_precision: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "tinyshakespeare"
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    val_split: float = 0.1


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    save_interval: int = 500


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def validate(self) -> None:
        self.model.validate()

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**raw_cfg.get("model", {}))
        training_cfg = TrainingConfig(**raw_cfg.get("training", {}))
        data_cfg = DataConfig(**raw_cfg.get("data", {}))
        checkpoint_cfg = CheckpointConfig(**raw_cfg.get("checkpoint", {}))

        cfg = cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            checkpoint=checkpoint_cfg,
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "checkpoint": self.checkpoint.__dict__,
        }
