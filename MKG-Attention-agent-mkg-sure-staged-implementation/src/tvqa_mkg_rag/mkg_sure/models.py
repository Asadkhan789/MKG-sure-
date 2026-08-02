from __future__ import annotations

import math

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - clear optional dependency error
    raise ImportError("MKG-Sure trainable heads require PyTorch") from exc


class ShortPathEncoder(nn.Module):
    """Question-conditioned attention over ordered path elements."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.question = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, question: Tensor, path_elements: Tensor, mask: Tensor | None = None) -> Tensor:
        # question: [batch, input_size], path_elements: [batch, length, input_size]
        query = self.question(question).unsqueeze(1)
        keys = self.key(path_elements)
        values = self.value(path_elements)
        logits = (query * keys).sum(dim=-1) / math.sqrt(keys.size(-1))
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (weights * values).sum(dim=1)
        return self.norm(self.dropout(pooled))


class UtilityHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, question: Tensor, path: Tensor) -> Tensor:
        features = torch.cat([question, path, question * path, torch.abs(question - path)], dim=-1)
        return self.network(features).squeeze(-1)


class SourceProjector(nn.Module):
    def __init__(self, source_size: int, reader_size: int, source_types: int = 6) -> None:
        super().__init__()
        self.projection = nn.Linear(source_size, reader_size)
        self.type_embedding = nn.Embedding(source_types, reader_size)
        self.norm = nn.LayerNorm(reader_size)

    def forward(self, source_embeddings: Tensor, source_type_ids: Tensor) -> Tensor:
        return self.norm(self.projection(source_embeddings) + self.type_embedding(source_type_ids))


class TwoStreamInjection(nn.Module):
    """Cross-attend reader states to selected path/source representations."""

    def __init__(self, hidden_size: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.Sigmoid())

    def forward(self, reader_states: Tensor, evidence_states: Tensor, question_state: Tensor) -> Tensor:
        attended, _ = self.attention(reader_states, evidence_states, evidence_states, need_weights=False)
        updated = self.norm(reader_states + attended)
        repeated_question = question_state.unsqueeze(1).expand(-1, reader_states.size(1), -1)
        gate = self.gate(torch.cat([reader_states, updated, repeated_question], dim=-1))
        return gate * updated + (1.0 - gate) * reader_states


class SufficiencyHead(nn.Module):
    """Predict sufficiency from coverage, connectivity, grounding and utility."""

    def __init__(self, hidden_size: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def utility_loss(
    logits: Tensor,
    helpful_targets: Tensor,
    utilities: Tensor,
    qids: Tensor,
    tie_margin: float = 0.05,
    pointwise_weight: float = 1.0,
    pairwise_weight: float = 1.0,
) -> Tensor:
    pointwise = nn.functional.binary_cross_entropy_with_logits(logits, helpful_targets.float())
    pair_losses: list[Tensor] = []
    for qid in torch.unique(qids):
        indices = torch.nonzero(qids == qid, as_tuple=False).flatten()
        for left_pos in range(indices.numel()):
            for right_pos in range(indices.numel()):
                left = indices[left_pos]
                right = indices[right_pos]
                difference = utilities[left] - utilities[right]
                if difference > tie_margin:
                    pair_losses.append(-nn.functional.logsigmoid(logits[left] - logits[right]))
    pairwise = torch.stack(pair_losses).mean() if pair_losses else logits.new_zeros(())
    return pointwise_weight * pointwise + pairwise_weight * pairwise
