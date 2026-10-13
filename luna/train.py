"""
LUNA Training Pipeline
Runs joint multi-task optimization for Sentiment Analysis and Toxic Comment Detection.
Supports AdamW optimizer, gradient clipping, evaluation validation, and checkpoint export.
"""

import time
from typing import Dict
import torch
from torch.utils.data import DataLoader
from .config import LunaConfig
from .tokenizer import LunaTokenizer
from .heads import LunaForMultiTaskClassification
from .dataset import LunaDataset, create_synthetic_benchmark


def train_epoch(
    model: LunaForMultiTaskClassification,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Runs a single training epoch across batches."""
    model.train()
    total_loss = 0.0
    total_sent_loss = 0.0
    total_toxic_loss = 0.0
    steps = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        sentiment_labels = batch.get("sentiment_labels", None)
        toxic_labels = batch.get("toxic_labels", None)

        if sentiment_labels is not None:
            sentiment_labels = sentiment_labels.to(device)
        if toxic_labels is not None:
            toxic_labels = toxic_labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentiment_labels=sentiment_labels,
            toxic_labels=toxic_labels
        )

        loss = outputs["loss"]
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        if outputs["loss_sentiment"] is not None:
            total_sent_loss += outputs["loss_sentiment"].item()
        if outputs["loss_toxic"] is not None:
            total_toxic_loss += outputs["loss_toxic"].item()

        steps += 1

    return {
        "loss": total_loss / max(1, steps),
        "sentiment_loss": total_sent_loss / max(1, steps),
        "toxic_loss": total_toxic_loss / max(1, steps),
    }


def evaluate(
    model: LunaForMultiTaskClassification,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates multi-task performance on validation set."""
    model.eval()
    correct_sent = 0
    total_sent = 0
    total_toxic_acc = 0.0
    total_toxic_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

            if "sentiment_labels" in batch:
                true_sent = batch["sentiment_labels"].to(device)
                pred_sent = torch.argmax(outputs["sentiment_probs"], dim=-1)
                correct_sent += (pred_sent == true_sent).sum().item()
                total_sent += true_sent.size(0)

            if "toxic_labels" in batch:
                true_toxic = batch["toxic_labels"].to(device)
                pred_toxic = (outputs["toxic_probs"] >= 0.5).float()
                match = (pred_toxic == true_toxic).float().mean(dim=-1)
                total_toxic_acc += match.sum().item()
                total_toxic_samples += true_toxic.size(0)

    sent_acc = (correct_sent / total_sent) if total_sent > 0 else 0.0
    toxic_acc = (total_toxic_acc / total_toxic_samples) if total_toxic_samples > 0 else 0.0

    return {
        "sentiment_accuracy": sent_acc,
        "toxic_mean_accuracy": toxic_acc,
    }


def run_training_pipeline(epochs: int = 3, batch_size: int = 4, lr: float = 3e-4):
    """Initializes LUNA architecture and trains on synthetic corpus."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LUNA] Initializing training on device: {device}")

    config = LunaConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024
    )
    tokenizer = LunaTokenizer()
    model = LunaForMultiTaskClassification(config).to(device)

    samples = create_synthetic_benchmark()
    dataset = LunaDataset(samples, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print(f"[LUNA] Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, dataloader, optimizer, device)
        eval_metrics = evaluate(model, dataloader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Loss: {train_metrics['loss']:.4f} "
            f"(Sent: {train_metrics['sentiment_loss']:.4f}, Toxic: {train_metrics['toxic_loss']:.4f}) | "
            f"Sent Acc: {eval_metrics['sentiment_accuracy']*100:.1f}% | "
            f"Toxic Acc: {eval_metrics['toxic_mean_accuracy']*100:.1f}%"
        )

    torch.save(model.state_dict(), "luna_multitask_weights.pt")
    print("[LUNA] Model checkpoint saved to 'luna_multitask_weights.pt'!")


if __name__ == "__main__":
    run_training_pipeline()
