"""
LUNA Dataset & DataLoader
Provides:
- LunaDataset: PyTorch Dataset supporting multi-task annotations
- create_synthetic_benchmark: High-quality curated corpus for rapid reproducible training
"""

from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset
from .tokenizer import LunaTokenizer


class LunaDataset(Dataset):
    """PyTorch Dataset for multi-task sentiment and toxicity training."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: LunaTokenizer,
        max_length: int = 128
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]

        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )

        item = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(encoded["token_type_ids"], dtype=torch.long),
        }

        if "sentiment" in sample:
            item["sentiment_labels"] = torch.tensor(sample["sentiment"], dtype=torch.long)

        if "toxic" in sample:
            item["toxic_labels"] = torch.tensor(sample["toxic"], dtype=torch.float)

        return item


def create_synthetic_benchmark() -> List[Dict[str, Any]]:
    """Curates a diverse multi-task benchmark."""
    return [
        {
            "text": "The camera build is stellar and battery life easily lasts two full days. Outstanding!",
            "sentiment": 2,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "Shut up you absolute idiot, you have no brain and should delete your useless account.",
            "sentiment": 0,
            "toxic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        },
        {
            "text": "The scheduled delivery package arrived on Tuesday afternoon via postal service.",
            "sentiment": 1,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "I will find where you live and destroy you, prepare to die.",
            "sentiment": 0,
            "toxic": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        },
        {
            "text": "Customer service was fantastic and resolved my broken order within ten minutes.",
            "sentiment": 2,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "Terrible service, completely broken update that ruined all my settings. Worst ever.",
            "sentiment": 0,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "I am completely killing it at the gym today, feeling like an absolute beast!",
            "sentiment": 2,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "You are a pathetic loser and a disgusting moron.",
            "sentiment": 0,
            "toxic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        },
        {
            "text": "Could someone please verify if the conference room audio device is working?",
            "sentiment": 1,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "Brilliant product, really love the sleek design and fast response time.",
            "sentiment": 2,
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    ]
