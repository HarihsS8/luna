"""
LUNA Evaluation & Inference Engine
Enables single-text inference, attention weights extraction, and token attribution.
"""

from typing import Dict, Any, Optional
import torch
from .config import LunaConfig
from .tokenizer import LunaTokenizer
from .heads import LunaForMultiTaskClassification


class LunaInferencePipeline:
    """High-level pipeline for running predictions with a trained LUNA model."""

    SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]
    TOXIC_NAMES = [
        "Toxic",
        "Severe Toxic",
        "Obscene",
        "Threat",
        "Insult",
        "Identity Attack"
    ]

    def __init__(
        self,
        model: LunaForMultiTaskClassification,
        tokenizer: LunaTokenizer,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs multi-task inference and returns structured probabilities."""
        encoded = self.tokenizer.encode(text, max_length=128, padding=False)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )

        sent_probs = outputs["sentiment_probs"][0].cpu().numpy()
        toxic_probs = outputs["toxic_probs"][0].cpu().numpy()

        best_sent_idx = int(sent_probs.argmax())

        toxic_breakdown = {}
        for name, prob in zip(self.TOXIC_NAMES, toxic_probs):
            toxic_breakdown[name] = {
                "probability": float(prob),
                "flagged": bool(prob >= 0.5)
            }

        return {
            "text": text,
            "tokens": encoded["tokens"],
            "sentiment": {
                "label": self.SENTIMENT_NAMES[best_sent_idx],
                "confidence": float(sent_probs[best_sent_idx]),
                "probabilities": {
                    "negative": float(sent_probs[0]),
                    "neutral": float(sent_probs[1]),
                    "positive": float(sent_probs[2]),
                }
            },
            "toxicity": {
                "is_toxic": bool(any(item["flagged"] for item in toxic_breakdown.values())),
                "max_score": float(toxic_probs.max()),
                "categories": toxic_breakdown
            },
            "attention_layers_count": len(outputs["attentions"]) if outputs["attentions"] else 0
        }
