"""
LUNA Classification Heads
Implements:
1. LunaSentimentClassifier: Single-label 3-class sentiment (Negative, Neutral, Positive)
2. LunaToxicClassifier: Multi-label 6-class toxic comment detection (BCEWithLogitsLoss)
3. LunaForMultiTaskClassification: Joint model sharing the LUNA transformer backbone
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from .config import LunaConfig
from .model import LunaModel


class LunaSentimentClassifier(nn.Module):
    """Sentiment Classification Head."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_sentiment_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


class LunaToxicClassifier(nn.Module):
    """Multi-label Toxic Comment Classification Head."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_toxic_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


class LunaForMultiTaskClassification(nn.Module):
    """Joint multi-task transformer architecture."""

    def __init__(self, config: LunaConfig, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.config = config
        self.alpha = alpha
        self.beta = beta

        self.luna = LunaModel(config)
        self.sentiment_head = LunaSentimentClassifier(config)
        self.toxic_head = LunaToxicClassifier(config)

        self.sentiment_loss_fn = nn.CrossEntropyLoss()
        self.toxic_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        sentiment_labels: Optional[torch.Tensor] = None,
        toxic_labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Dict[str, Any]:
        outputs = self.luna(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        pooled_output = outputs["pooled_output"]

        sentiment_logits = self.sentiment_head(pooled_output)
        toxic_logits = self.toxic_head(pooled_output)

        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)
        toxic_probs = torch.sigmoid(toxic_logits)

        loss: Optional[torch.Tensor] = None
        loss_sentiment: Optional[torch.Tensor] = None
        loss_toxic: Optional[torch.Tensor] = None

        if sentiment_labels is not None or toxic_labels is not None:
            loss = torch.tensor(0.0, device=input_ids.device)

            if sentiment_labels is not None:
                loss_sentiment = self.sentiment_loss_fn(sentiment_logits, sentiment_labels)
                loss = loss + self.alpha * loss_sentiment

            if toxic_labels is not None:
                loss_toxic = self.toxic_loss_fn(toxic_logits, toxic_labels.float())
                loss = loss + self.beta * loss_toxic

        return {
            "loss": loss,
            "loss_sentiment": loss_sentiment,
            "loss_toxic": loss_toxic,
            "sentiment_logits": sentiment_logits,
            "sentiment_probs": sentiment_probs,
            "toxic_logits": toxic_logits,
            "toxic_probs": toxic_probs,
            "attentions": outputs["attentions"],
            "pooled_output": pooled_output,
        }
