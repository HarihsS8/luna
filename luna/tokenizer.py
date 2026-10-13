"""
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
            if len(w) > 4:
                vocab["##" + w[2:]] = idx
                idx += 1

        return vocab

    def tokenize(self, text: str) -> List[str]:
        """Splits raw text into WordPiece tokens."""
        text = text.strip()
        if not text:
            return []

        raw_tokens = re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)
        output_tokens: List[str] = []

        for token in raw_tokens:
            if token in self.vocab:
                output_tokens.append(token)
                continue

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
        """Encodes text into input_ids, token_type_ids, and attention_mask."""
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
