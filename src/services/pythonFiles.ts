import { PythonCodeFile } from '../types/luna';

export const LUNA_PYTHON_FILES: PythonCodeFile[] = [
  {
    path: 'luna/__init__.py',
    filename: '__init__.py',
    module: 'luna',
    description: 'Package entry point exposing core classes and configuration.',
    code: `"""
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
`
  },
  {
    path: 'luna/config.py',
    filename: 'config.py',
    module: 'luna',
    description: 'Hyperparameter configuration for LUNA transformer architecture.',
    code: `"""
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
`
  },
  {
    path: 'luna/tokenizer.py',
    filename: 'tokenizer.py',
    module: 'luna',
    description: 'WordPiece subword tokenizer replicating BERT vocabulary processing.',
    code: `"""
LUNA WordPiece Tokenizer
Handles subword splitting, special token insertion ([CLS], [SEP], [PAD]),
and token-to-ID tensor conversions.
"""

import re
from typing import List, Dict, Optional, Tuple


class LunaTokenizer:
    """
    Subword WordPiece tokenizer for LUNA.
    Splits text into words, applies subword splitting with '##' prefix,
    and maps tokens to integer vocabulary IDs.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is None:
            self.vocab = self._build_default_vocab()
        else:
            self.vocab = vocab

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.cls_token_id = self.vocab[self.cls_token]
        self.sep_token_id = self.vocab[self.sep_token]
        self.mask_token_id = self.vocab[self.mask_token]

    def _build_default_vocab(self) -> Dict[str, int]:
        """Builds a representative vocabulary containing common English and sentiment tokens."""
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        common_words = [
            "the", "a", "an", "is", "was", "are", "were", "this", "that", "these", "those",
            "it", "its", "i", "you", "he", "she", "they", "we", "my", "your", "their", "our",
            "not", "no", "never", "very", "really", "so", "good", "great", "excellent", "amazing",
            "love", "best", "awesome", "stellar", "fantastic", "brilliant", "helpful", "bad",
            "terrible", "awful", "horrible", "hate", "worst", "poor", "broken", "disappointed",
            "disappointing", "useless", "idiot", "stupid", "moron", "dumb", "loser", "trash",
            "kill", "die", "destroy", "threat", "punch", "attack", "shut", "up", "ugly",
            "game", "play", "work", "product", "service", "device", "team", "delivery", "fast",
            "slow", "price", "customer", "support", "update", "bug", "quality", "camera", "battery"
        ]
        punct = [".", ",", "!", "?", ";", ":", "'", '"', "-", "(", ")"]

        vocab: Dict[str, int] = {}
        # Special tokens
        vocab["[PAD]"] = 0
        vocab["[UNK]"] = 100
        vocab["[CLS]"] = 101
        vocab["[SEP]"] = 102
        vocab["[MASK]"] = 103

        idx = 1000
        for p in punct:
            vocab[p] = idx
            idx += 1

        for w in common_words:
            vocab[w] = idx
            idx += 1
            # Add subwords
            if len(w) > 4:
                vocab["##" + w[2:]] = idx
                idx += 1

        return vocab

    def tokenize(self, text: str) -> List[str]:
        """Splits raw text into WordPiece tokens."""
        text = text.strip()
        if not text:
            return []

        # Split on whitespace and punctuation
        raw_tokens = re.findall(r"\\w+|[^\\w\\s]", text.lower(), re.UNICODE)
        output_tokens: List[str] = []

        for token in raw_tokens:
            if token in self.vocab:
                output_tokens.append(token)
                continue

            # WordPiece subword search
            is_bad = False
            start = 0
            sub_tokens: List[str] = []
            while start < len(token):
                end = len(token)
                cur_substr = None
                while start < end:
                    substr = token[start:end]
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1

                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens

    def encode(
        self,
        text: str,
        max_length: int = 128,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Encodes a single sentence into input_ids, token_type_ids, and attention_mask.
        Includes [CLS] and [SEP].
        """
        tokens = self.tokenize(text)
        if truncation and len(tokens) > max_length - 2:
            tokens = tokens[: max_length - 2]

        tokens = [self.cls_token] + tokens + [self.sep_token]
        input_ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        if padding and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "tokens": tokens,
        }

    def decode(self, token_ids: List[int]) -> str:
        """Converts token IDs back into readable string."""
        words: List[str] = []
        for tid in token_ids:
            if tid == self.pad_token_id:
                continue
            tok = self.inv_vocab.get(tid, self.unk_token)
            if tok in [self.cls_token, self.sep_token]:
                continue
            if tok.startswith("##"):
                if words:
                    words[-1] += tok[2:]
                else:
                    words.append(tok[2:])
            else:
                words.append(tok)
        return " ".join(words)
`
  },
  {
    path: 'luna/model.py',
    filename: 'model.py',
    module: 'luna',
    description: 'PyTorch implementation of LUNA BERT-like bidirectional transformer backbone.',
    code: `"""
LUNA Transformer Architecture
Full bidirectional BERT-like transformer implemented in PyTorch:
- LunaEmbeddings (Token + Position + Segment + LayerNorm)
- LunaSelfAttention (Scaled dot-product multi-head self-attention)
- LunaIntermediate & LunaOutput (Feed-Forward network with GELU)
- LunaLayer (Single transformer block)
- LunaEncoder (Stack of N layers with attention inspection)
- LunaPooler ([CLS] representation pooler)
- LunaModel (Base transformer backbone)
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from .config import LunaConfig


class LunaEmbeddings(nn.Module):
    """
    Constructs the embeddings from word, position, and token_type embeddings.
    Formula:
        Embedding = LayerNorm(E_token + E_position + E_segment)
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LunaSelfAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Self-Attention:
    Attention(Q, K, V) = Softmax( (Q K^T) / sqrt(d_k) ) V
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by "
                f"num_attention_heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, seq_len, hidden_size] -> [batch, num_heads, seq_len, head_dim]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Raw attention scores: [batch, num_heads, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply attention mask (additive mask where 0 becomes large negative number)
            attention_scores = attention_scores + attention_mask

        # Normalize with Softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Context output: [batch, num_heads, seq_len, head_dim]
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer, None)
        return outputs


class LunaSelfOutput(nn.Module):
    """Linear projection + LayerNorm + Residual connection."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LunaAttention(nn.Module):
    """Wraps LunaSelfAttention and LunaSelfOutput."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.self = LunaSelfAttention(config)
        self.output = LunaSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_outputs = self.self(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class LunaIntermediate(nn.Module):
    """Position-wise Feed-Forward expansion: Hidden -> Intermediate with GELU."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LunaOutput(nn.Module):
    """Feed-Forward contraction: Intermediate -> Hidden + Residual + LayerNorm."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LunaLayer(nn.Module):
    """A single Transformer Encoder block."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.attention = LunaAttention(config)
        self.intermediate = LunaIntermediate(config)
        self.output = LunaOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self_attn_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attn_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        outputs = (layer_output,) + self_attn_outputs[1:]
        return outputs


class LunaEncoder(nn.Module):
    """Stack of N LunaLayers."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.layers = nn.ModuleList([LunaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        all_hidden_states: List[torch.Tensor] = []
        all_attentions: List[torch.Tensor] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer(hidden_states, attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions and layer_outputs[1] is not None:
                all_attentions.append(layer_outputs[1])

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states if output_hidden_states else None,
            "attentions": all_attentions if output_attentions else None,
        }


class LunaPooler(nn.Module):
    """Extracts [CLS] token representation (index 0) and applies Linear + Tanh."""

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LunaModel(nn.Module):
    """
    The base LUNA bidirectional transformer model.
    Outputs last hidden states, pooled [CLS] vector, and optional attention matrices.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.config = config
        self.embeddings = LunaEmbeddings(config)
        self.encoder = LunaEncoder(config)
        self.pooler = LunaPooler(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights standard normal with mean=0, std=initializer_range."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        # Create additive attention mask: [batch, 1, 1, seq_len]
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        else:
            extended_attention_mask = None

        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = encoder_outputs["last_hidden_state"]
        pooled_output = self.pooler(sequence_output)

        return {
            "last_hidden_state": sequence_output,
            "pooled_output": pooled_output,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
        }
`
  },
  {
    path: 'luna/heads.py',
    filename: 'heads.py',
    module: 'luna',
    description: 'Multi-task classification heads for sentiment analysis and toxic comment detection.',
    code: `"""
LUNA Classification Heads
Implements:
1. LunaSentimentClassifier: Single-label 3-class sentiment (Negative, Neutral, Positive)
2. LunaToxicClassifier: Multi-label 6-class toxic comment detection (BCEWithLogitsLoss)
3. LunaForMultiTaskClassification: Joint model sharing the LUNA transformer backbone
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from .config import LunaConfig
from .model import LunaModel


class LunaSentimentClassifier(nn.Module):
    """
    Sentiment Classification Head.
    Takes pooled [CLS] representation -> Dropout -> Linear -> 3 Logits.
    Loss: CrossEntropyLoss
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_sentiment_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


class LunaToxicClassifier(nn.Module):
    """
    Multi-label Toxic Comment Classification Head.
    Takes pooled [CLS] representation -> Dropout -> Linear -> 6 Logits.
    Loss: BCEWithLogitsLoss (independent binary probabilities per category).
    Categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.
    """

    def __init__(self, config: LunaConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_toxic_classes)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


class LunaForMultiTaskClassification(nn.Module):
    """
    Joint multi-task transformer architecture.
    Shares the bidirectional LUNA transformer encoder representation and branches into:
      1) Sentiment classification head
      2) Toxic comment multi-label head
    Computes weighted multi-task loss during training:
      Loss_total = alpha * Loss_sentiment + beta * Loss_toxic
    """

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
        sentiment_labels: Optional[torch.Tensor] = None,   # [batch] int (0, 1, 2)
        toxic_labels: Optional[torch.Tensor] = None,       # [batch, 6] float (0.0 or 1.0)
        output_attentions: bool = False
    ) -> Dict[str, Any]:
        # Forward pass through shared LUNA backbone
        outputs = self.luna(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

        pooled_output = outputs["pooled_output"]

        # Forward through both task heads
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
`
  },
  {
    path: 'luna/dataset.py',
    filename: 'dataset.py',
    module: 'luna',
    description: 'PyTorch Dataset and synthetic corpus generator for sentiment & toxic comments.',
    code: `"""
LUNA Dataset & DataLoader
Provides:
- LunaDataset: PyTorch Dataset supporting multi-task annotations
- create_synthetic_benchmark: High-quality curated corpus for rapid reproducible training
- collate_batch: Dynamic batching and tensor padding
"""

from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from .tokenizer import LunaTokenizer


class LunaDataset(Dataset):
    """
    PyTorch Dataset for multi-task sentiment and toxicity training.
    """

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
            # List of 6 binary floats: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
            item["toxic_labels"] = torch.tensor(sample["toxic"], dtype=torch.float)

        return item


def create_synthetic_benchmark() -> List[Dict[str, Any]]:
    """
    Curates a diverse multi-task benchmark covering:
    - Highly positive & grateful comments
    - Polite neutral inquiries & status updates
    - Severely toxic insults & violent threats
    - Tricky sarcasm and false positive idiom traps ('killing it at work')
    """
    return [
        {
            "text": "The camera build is stellar and battery life easily lasts two full days. Outstanding!",
            "sentiment": 2, # Positive
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "Shut up you absolute idiot, you have no brain and should delete your useless account.",
            "sentiment": 0, # Negative
            "toxic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] # toxic, insult
        },
        {
            "text": "The scheduled delivery package arrived on Tuesday afternoon via postal service.",
            "sentiment": 1, # Neutral
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "I will find where you live and destroy you, prepare to die.",
            "sentiment": 0, # Negative
            "toxic": [1.0, 1.0, 0.0, 1.0, 0.0, 0.0] # toxic, severe_toxic, threat
        },
        {
            "text": "Customer service was fantastic and resolved my broken order within ten minutes.",
            "sentiment": 2, # Positive
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "Terrible service, completely broken update that ruined all my settings. Worst ever.",
            "sentiment": 0, # Negative
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # Negative sentiment but not toxic
        },
        {
            "text": "I am completely killing it at the gym today, feeling like an absolute beast!",
            "sentiment": 2, # Positive idiom trap
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "You are a pathetic loser and a disgusting moron.",
            "sentiment": 0, # Negative
            "toxic": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        },
        {
            "text": "Could someone please verify if the conference room audio device is working?",
            "sentiment": 1, # Neutral
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        {
            "text": "Brilliant product, really love the sleek design and fast response time.",
            "sentiment": 2, # Positive
            "toxic": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    ]
`
  },
  {
    path: 'luna/train.py',
    filename: 'train.py',
    module: 'luna',
    description: 'PyTorch training loop with AdamW, linear warmup, and multi-task evaluation.',
    code: `"""
LUNA Training Pipeline
Runs joint multi-task optimization for Sentiment Analysis and Toxic Comment Detection.
Supports AdamW optimizer, gradient clipping, evaluation validation, and checkpoint export.
"""

import time
import math
import torch
from torch.utils.data import DataLoader
from .config import LunaConfig
from .tokenizer import LunaTokenizer
from .model import LunaModel
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

        # Gradient clipping to prevent exploding gradients
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

            # Sentiment accuracy
            if "sentiment_labels" in batch:
                true_sent = batch["sentiment_labels"].to(device)
                pred_sent = torch.argmax(outputs["sentiment_probs"], dim=-1)
                correct_sent += (pred_sent == true_sent).sum().item()
                total_sent += true_sent.size(0)

            # Toxicity accuracy (threshold at 0.5)
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

    # Save trained checkpoint
    torch.save(model.state_dict(), "luna_multitask_weights.pt")
    print("[LUNA] Model checkpoint saved to 'luna_multitask_weights.pt'!")


if __name__ == "__main__":
    run_training_pipeline()
`
  },
  {
    path: 'luna/eval.py',
    filename: 'eval.py',
    module: 'luna',
    description: 'Inference, token saliency attribution, and attention rollout inspector.',
    code: `"""
LUNA Evaluation & Inference Engine
Enables single-text inference, attention weights extraction, and token attribution.
"""

from typing import Dict, Any, List
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
`
  },
  {
    path: 'quickstart.py',
    filename: 'quickstart.py',
    module: 'root',
    description: 'Self-contained runnable demonstration script for testing LUNA.',
    code: `"""
LUNA Quickstart Script
Run this file directly: python quickstart.py
Demonstrates:
1. Tokenization with subwords and special markers
2. Instantiation of LUNA Transformer backbone and multi-task heads
3. Mini forward pass and loss computation
4. Live inference on sentiment and toxic comment samples
"""

import torch
from luna.config import LunaConfig
from luna.tokenizer import LunaTokenizer
from luna.heads import LunaForMultiTaskClassification
from luna.eval import LunaInferencePipeline


def main():
    print("==================================================")
    print("   LUNA: Language Understanding Neural Attention   ")
    print("   BERT-like Transformer for Sentiment & Toxicity   ")
    print("==================================================")

    # 1. Initialize Configuration
    config = LunaConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    print(f"\\n1. Initialized LUNA Architecture:")
    print(f"   - Layers: {config.num_hidden_layers}")
    print(f"   - Attention Heads: {config.num_attention_heads}")
    print(f"   - Hidden Size: {config.hidden_size}")
    print(f"   - FFN Dimension: {config.intermediate_size}")

    # 2. Tokenizer Demonstration
    tokenizer = LunaTokenizer()
    sample_text = "The camera build is stellar, but customer support was terrible!"
    encoded = tokenizer.encode(sample_text, max_length=32)
    print(f"\\n2. Tokenization:")
    print(f"   - Raw text: '{sample_text}'")
    print(f"   - Tokens:   {encoded['tokens']}")
    print(f"   - IDs:      {encoded['input_ids'][:len(encoded['tokens'])]}")

    # 3. Model Instantiation
    model = LunaForMultiTaskClassification(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\\n3. Transformer Parameters: {total_params:,}")

    # 4. Forward Pass with Dummy Batch
    dummy_input = torch.tensor([encoded['input_ids']], dtype=torch.long)
    dummy_mask = torch.tensor([encoded['attention_mask']], dtype=torch.long)
    dummy_sent = torch.tensor([0], dtype=torch.long) # Negative target
    dummy_toxic = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float)

    outputs = model(
        input_ids=dummy_input,
        attention_mask=dummy_mask,
        sentiment_labels=dummy_sent,
        toxic_labels=dummy_toxic,
        output_attentions=True
    )

    print(f"\\n4. Forward Pass Verification:")
    print(f"   - Multi-Task Loss: {outputs['loss'].item():.4f}")
    print(f"   - Sentiment Logits: {outputs['sentiment_logits'].detach().numpy()[0]}")
    print(f"   - Toxic Logits:     {outputs['toxic_logits'].detach().numpy()[0]}")
    print(f"   - Attention Layers: {len(outputs['attentions'])}")

    # 5. Pipeline Demonstration
    pipeline = LunaInferencePipeline(model, tokenizer)
    test_sentences = [
        "Outstanding customer care, resolved my issue in five minutes!",
        "Shut up you absolute idiot, delete your useless account right now.",
        "The package arrived on Tuesday afternoon via standard mail.",
        "I am completely killing it at the gym today, feeling great!"
    ]

    print(f"\\n5. Running Test Inferences:")
    for text in test_sentences:
        res = pipeline.predict(text)
        print(f"\\n   [Text] '{text}'")
        print(f"   [Sentiment] {res['sentiment']['label']} (confidence: {res['sentiment']['confidence']:.2%})")
        print(f"   [Toxicity]  {'⚠️ FLAGGED TOXIC' if res['toxicity']['is_toxic'] else '✅ Clean'} (max score: {res['toxicity']['max_score']:.2f})")

    print("\\n==================================================")
    print("   LUNA Transformer setup complete and verified!  ")
    print("==================================================")


if __name__ == "__main__":
    main()
`
  },
  {
    path: 'requirements.txt',
    filename: 'requirements.txt',
    module: 'root',
    description: 'Python package dependencies for running LUNA locally or in Google Colab.',
    code: `torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
scikit-learn>=1.2.0
`
  },
  {
    path: 'README.md',
    filename: 'README.md',
    module: 'root',
    description: 'Comprehensive technical documentation and mathematical formulation for LUNA.',
    code: `# LUNA: Language Understanding Neural Attention

LUNA is a custom **BERT-like bidirectional Transformer architecture** designed in Python for joint multi-task **Sentiment Classification** (Negative, Neutral, Positive) and **Toxic Comment Detection** (6 sub-labels: Toxic, Severe Toxic, Insult, Threat, Obscene, Identity Hate).

---

## 1. Architectural Highlights

- **Bidirectional Attention**: Unlike causal autoregressive models (GPT), LUNA attends to both left and right context tokens at every layer.
- **Shared Transformer Backbone**: 6 Encoder layers with 8 Multi-Head Attention heads ($d_{model} = 256$, $d_{ff} = 1024$).
- **WordPiece Tokenizer**: Subword vocabulary with special tokens \`[PAD]=0\`, \`[UNK]=100\`, \`[CLS]=101\`, \`[SEP]=102\`, \`[MASK]=103\`.
- **Dual Classification Heads**:
  - **Sentiment Head**: Softmax over 3 classes with Cross-Entropy Loss:
    $$\\mathcal{L}_{sentiment} = -\\sum_{c=1}^{3} y_c \\log(\\hat{y}_c)$$
  - **Toxic Comment Head**: Independent Sigmoid activations over 6 labels with Binary Cross-Entropy Loss:
    $$\\mathcal{L}_{toxic} = -\\frac{1}{6}\\sum_{j=1}^{6} [y_j \\log(\\sigma(z_j)) + (1 - y_j)\\log(1 - \\sigma(z_j))]$$
  - **Joint Multi-Task Loss**:
    $$\\mathcal{L}_{total} = \\alpha \\mathcal{L}_{sentiment} + \\beta \\mathcal{L}_{toxic}$$

---

## 2. Quickstart

\`\`\`bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quickstart verification
python quickstart.py

# 3. Train on multi-task corpus
python -m luna.train
\`\`\`

---

## 3. Directory Layout

\`\`\`
luna/
├── __init__.py           # Package exports
├── config.py             # LunaConfig dataclass
├── tokenizer.py          # Subword WordPiece tokenizer
├── model.py              # PyTorch bidirectional transformer backbone
├── heads.py              # Sentiment & Toxic classification heads
├── dataset.py            # PyTorch Dataset & Synthetic benchmark
├── train.py              # Training loop with AdamW & metric logging
└── eval.py               # Evaluation pipeline & attention extraction
quickstart.py             # Standalone test script
requirements.txt          # Python dependencies
README.md                 # Architecture documentation
\`\`\`
`
  }
];
