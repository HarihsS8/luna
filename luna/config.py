"""
LUNA Configuration Dataclass
Defines architectural hyperparameters for the BERT-like transformer backbone
and dual classification heads.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class LunaConfig:
    """Configuration class storing all hyperparameters of a LunaModel."""
    vocab_size: int = 30522
    hidden_size: int = 256              # 768 for Base, 256 for fast edge/experimentation
    num_hidden_layers: int = 6          # 12 for Base, 6 for fast training
    num_attention_heads: int = 8        # 12 for Base, 8 for fast training
    intermediate_size: int = 1024       # FFN dimension (typically 4x hidden_size)
    hidden_act: str = "gelu"            # Activation: gelu, relu, or silu
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2            # Segment token types (Sentence A / Sentence B)
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    unk_token_id: int = 100
    cls_token_id: int = 101
    sep_token_id: int = 102
    mask_token_id: int = 103

    # Multi-task Classification Head settings
    num_sentiment_classes: int = 3      # 0: Negative, 1: Neutral, 2: Positive
    toxic_labels: List[str] = field(default_factory=lambda: [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ])

    @property
    def num_toxic_classes(self) -> int:
        return len(self.toxic_labels)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def to_json_string(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "LunaConfig":
        return cls(**data)
