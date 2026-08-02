from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def encode_texts(self, texts: list[str]) -> torch.Tensor: ...

    def encode_media(self, paths: list[str], prompts: list[str] | None = None) -> torch.Tensor:
        labels = prompts or [Path(path).stem for path in paths]
        return self.encode_texts([f"media {Path(path).name} {label}" for path, label in zip(paths, labels)])


class HashEmbedder(Embedder):
    """Deterministic dependency-light embedder for tests and no-model development."""

    def __init__(self, dim: int = 128):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self._dim, dtype=torch.float32)
        tokens = text.lower().split() or [""]
        for position, token in enumerate(tokens):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            for offset in range(0, 16, 2):
                index = int.from_bytes(digest[offset:offset + 2], "little") % self._dim
                sign = 1.0 if digest[offset] % 2 == 0 else -1.0
                vec[index] += sign / math.sqrt(position + 1.0)
        return F.normalize(vec.unsqueeze(0), dim=-1).squeeze(0)

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty((0, self._dim), dtype=torch.float32)
        return torch.stack([self._vector(text) for text in texts])


class TransformersTextEmbedder(Embedder):
    """Optional frozen text embedding adapter preserving a stable API."""

    def __init__(self, model_name: str, device: str = "auto"):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install mkg-sure[real] for transformers backends") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        resolved = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.device = torch.device(resolved)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self._dim = int(getattr(self.model.config, "hidden_size", 1024))

    @property
    def dim(self) -> int:
        return self._dim

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        outputs = self.model(**batch)
        hidden = outputs.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled.float(), dim=-1).cpu()


def make_embedder(name: str, dim: int = 128, device: str = "auto") -> Embedder:
    return HashEmbedder(dim) if name in {"hash", "dummy"} else TransformersTextEmbedder(name, device=device)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0)).item())
