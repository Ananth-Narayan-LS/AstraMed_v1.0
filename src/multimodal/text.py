"""
Small, dependency-free text encoder for symptoms / clinical-history text.

RSNA has NO such text, so this module is NOT used in any trained Sprint 8 model.
It exists so genuine text can be plugged in later, and it is unit-tested with
clearly labelled synthetic strings only. Empty text is represented by an all-zero
mask, never by invented content.
"""
from __future__ import annotations

from collections import Counter

import torch
import torch.nn as nn

PAD_ID, UNK_ID = 0, 1


class SimpleTokenizer:
    """Lower-case word tokenizer with a vocabulary fitted on TRAINING text only."""

    def __init__(self, max_len: int = 64):
        if max_len <= 0:
            raise ValueError("max_len must be positive.")
        self.max_len = max_len
        self.vocab: dict[str, int] | None = None

    @staticmethod
    def _words(text) -> list[str]:
        return str(text or "").lower().replace(",", " ").replace(".", " ").split()

    def fit(self, train_texts, max_vocab: int = 5000, min_freq: int = 1) -> "SimpleTokenizer":
        counts = Counter(w for text in train_texts for w in self._words(text))
        words = [w for w, c in counts.most_common(max_vocab) if c >= min_freq]
        self.vocab = {w: i + 2 for i, w in enumerate(words)}
        return self

    @property
    def vocab_size(self) -> int:
        if self.vocab is None:
            raise RuntimeError("Tokenizer must be fitted on training text first.")
        return len(self.vocab) + 2

    def encode(self, text) -> tuple[list[int], list[float]]:
        """Return (token ids, mask) padded/truncated to max_len. Empty text -> mask all 0."""
        if self.vocab is None:
            raise RuntimeError("Tokenizer must be fitted on training text first.")
        ids = [self.vocab.get(w, UNK_ID) for w in self._words(text)][: self.max_len]
        mask = [1.0] * len(ids)
        pad = self.max_len - len(ids)
        return ids + [PAD_ID] * pad, mask + [0.0] * pad

    def encode_batch(self, texts) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.encode(t) for t in texts]
        ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
        mask = torch.tensor([e[1] for e in encoded], dtype=torch.float32)
        return ids, mask


class ClinicalTextEncoder(nn.Module):
    """Embedding + masked mean pooling + projection. Output is zero where text is empty."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.projection = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.ReLU())

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if token_ids.shape != token_mask.shape:
            raise ValueError("token_ids and token_mask must have the same shape.")
        embedded = self.embedding(token_ids)
        weights = token_mask.unsqueeze(-1)
        pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        has_text = (token_mask.sum(dim=1, keepdim=True) > 0).float()
        return self.projection(pooled) * has_text
