"""
LUNA: Language Understanding Neural Attention
A BERT-like bidirectional transformer for multi-task sentiment classification
and toxic comment detection.
"""

from .config import LunaConfig
from .tokenizer import LunaTokenizer
from .model import (
    LunaModel,
    LunaEmbeddings,
    LunaSelfAttention,
    LunaLayer,
    LunaEncoder,
    LunaPooler
)
from .heads import (
    LunaForMultiTaskClassification,
    LunaSentimentClassifier,
    LunaToxicClassifier
)

__version__ = "1.0.0"
__author__ = "LUNA NLP Architecture"
__all__ = [
    "LunaConfig",
    "LunaTokenizer",
    "LunaModel",
    "LunaEmbeddings",
    "LunaSelfAttention",
    "LunaLayer",
    "LunaEncoder",
    "LunaPooler",
    "LunaForMultiTaskClassification",
    "LunaSentimentClassifier",
    "LunaToxicClassifier",
]
