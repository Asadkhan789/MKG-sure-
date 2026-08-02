from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .config import MKGSureConfig
from .io_utils import set_seed
from .models import SourceProjector, Stage1Model, SufficiencyHead, ToyFrozenAnswerHead, TwoStreamInjection
from .schemas import SufficiencyExample, UtilityLabel


def train_stage1(labels: list[UtilityLabel], config: MKGSureConfig, checkpoint: Path) -> tuple[Stage1Model, dict[str, float]]:
    if not labels:
        raise ValueError("Stage 1 requires utility labels")
    set_seed(config.training.seed)
    input_dim = len(labels[0].question_vector)
    model = Stage1Model(input_dim, config.training.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    q = torch.tensor([row.question_vector for row in labels], dtype=torch.float32)
    p = torch.tensor([row.path_vector for row in labels], dtype=torch.float32)
    y = torch.tensor([float(row.helpful) for row in labels], dtype=torch.float32)
    utilities = torch.tensor([row.utility for row in labels], dtype=torch.float32)
    last_loss = 0.0
    for _ in range(config.training.stage1_epochs):
        model.train()
        logits, _ = model(q, p)
        pointwise = F.binary_cross_entropy_with_logits(logits, y)
        order = torch.argsort(utilities)
        low, high = order[:len(order)//2], order[len(order)//2:]
        pair_count = min(len(low), len(high))
        pairwise = -F.logsigmoid(logits[high[:pair_count]] - logits[low[-pair_count:]]).mean() if pair_count else torch.zeros((), dtype=torch.float32)
        loss = config.training.pointwise_weight * pointwise + config.training.pairwise_weight * pairwise
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        last_loss = float(loss.item())
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim, "hidden_dim": config.training.hidden_dim}, checkpoint)
    with torch.no_grad():
        logits, _ = model(q, p)
        accuracy = float(((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).float().mean().item())
    return model, {"loss": last_loss, "helpfulness_accuracy": accuracy, "examples": float(len(labels))}


def load_stage1(checkpoint: Path) -> Stage1Model:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Stage1Model(payload["input_dim"], payload["hidden_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def train_stage2(examples: list[SufficiencyExample], config: MKGSureConfig, checkpoint: Path) -> dict[str, float]:
    if not examples:
        raise ValueError("Stage 2 requires sufficiency examples")
    set_seed(config.training.seed)
    dim = config.training.hidden_dim
    source_projector = SourceProjector(dim, dim)
    injection = TwoStreamInjection(dim)
    sufficiency = SufficiencyHead()
    frozen_reader = ToyFrozenAnswerHead(dim)
    frozen_reader.requires_grad_(False)
    parameters = list(source_projector.parameters()) + list(injection.parameters()) + list(sufficiency.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    features = torch.tensor([row.features for row in examples], dtype=torch.float32)
    labels = torch.tensor([row.label for row in examples], dtype=torch.float32)
    generator = torch.Generator().manual_seed(config.training.seed)
    context = torch.randn(len(examples), 3, dim, generator=generator)
    paths = torch.randn(len(examples), 2, dim, generator=generator)
    question = torch.randn(len(examples), dim, generator=generator)
    source_type_ids = torch.zeros((len(examples), 2), dtype=torch.long)
    last_loss = 0.0
    for _ in range(config.training.stage2_epochs):
        projected = source_projector(paths, source_type_ids)
        fused = injection(context, projected, question)
        answer_logits = frozen_reader(fused.mean(dim=1), 2)
        answer_target = labels.long().clamp(0, 1)
        generation_loss = F.cross_entropy(answer_logits, answer_target)
        suff_logits = sufficiency(features)
        suff_loss = F.binary_cross_entropy_with_logits(suff_logits, labels)
        loss = config.training.generation_weight * generation_loss + config.training.sufficiency_weight * suff_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        last_loss = float(loss.item())
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"source_projector": source_projector.state_dict(), "injection": injection.state_dict(), "sufficiency": sufficiency.state_dict(), "hidden_dim": dim}, checkpoint)
    with torch.no_grad():
        probs = torch.sigmoid(sufficiency(features))
        accuracy = float(((probs >= 0.5) == (labels >= 0.5)).float().mean().item())
    return {"loss": last_loss, "sufficiency_accuracy": accuracy, "examples": float(len(examples))}


def load_sufficiency_head(checkpoint: Path) -> SufficiencyHead:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    head = SufficiencyHead(); head.load_state_dict(payload["sufficiency"])
    return head.eval()
