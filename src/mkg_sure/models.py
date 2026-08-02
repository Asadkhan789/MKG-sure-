from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class ShortPathEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.question = nn.Linear(input_dim, hidden_dim)
        self.path = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, question: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        q = self.question(question)
        p = self.path(path)
        attention = torch.sigmoid((q * p).sum(-1, keepdim=True) / math.sqrt(q.shape[-1]))
        return self.output(attention * p + (1.0 - attention) * q)


class UtilityHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def forward(self, question_hidden: torch.Tensor, path_hidden: torch.Tensor) -> torch.Tensor:
        features = torch.cat([question_hidden, path_hidden, question_hidden * path_hidden, torch.abs(question_hidden - path_hidden)], dim=-1)
        return self.mlp(features).squeeze(-1)


class Stage1Model(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = ShortPathEncoder(input_dim, hidden_dim)
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.utility = UtilityHead(hidden_dim)

    def forward(self, question: torch.Tensor, path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        path_hidden = self.encoder(question, path)
        question_hidden = self.q_proj(question)
        return self.utility(question_hidden, path_hidden), path_hidden


class SourceProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, source_types: int = 8):
        super().__init__()
        self.type_embedding = nn.Embedding(source_types, hidden_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, sources: torch.Tensor, source_type_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(sources), dim=-1) + self.type_embedding(source_type_ids)


class TwoStreamInjection(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, context: torch.Tensor, paths: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.q(context), self.k(paths).transpose(-1, -2)) / math.sqrt(context.shape[-1])
        attended = torch.softmax(scores, dim=-1) @ self.v(paths)
        enriched = self.norm(context + attended)
        q = question.unsqueeze(1).expand(-1, context.shape[1], -1)
        gate = torch.sigmoid(self.gate(torch.cat([context, enriched, q], dim=-1)))
        return gate * enriched + (1.0 - gate) * context


class SufficiencyHead(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class ToyFrozenAnswerHead(nn.Module):
    """Frozen differentiable stand-in used only by the smoke Stage-2 test."""

    def __init__(self, hidden_dim: int, max_options: int = 5, seed: int = 17):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(max_options, hidden_dim, generator=generator) / math.sqrt(hidden_dim)
        self.register_buffer("weight", weight)

    def forward(self, pooled: torch.Tensor, option_count: int) -> torch.Tensor:
        return pooled @ self.weight[:option_count].transpose(0, 1)
